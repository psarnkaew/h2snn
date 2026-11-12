"""NSD -- Neural Sheaf Diffusion (Bodnar et al., NeurIPS 2022).

A homogeneous sheaf model, so it takes the same two adaptations as ChebNetII, both
recorded on the run record:

1. **Graph reduction** via the shared
   :class:`~src.baselines.bundles.HomogeneousView` -- HGB's summed, symmetrised,
   self-looped, untyped graph.
2. **Input projection** via :class:`~src.baselines.encoders.TypeProjectedInput`, since NSD
   consumes a single dense feature matrix.

Their published settings cover WebKB and the Wikipedia heterophilic graphs
(``neural-sheaf-diffusion/exp/scripts/``), not ACM/DBLP/IMDB, so every cell here is tuned.

``--model`` selects the restriction-map family; ``DiagSheaf`` is their default and the
cheapest, with ``BundleSheaf`` (orthogonal) and ``GeneralSheaf`` (dense d x d) available in
the search space. Note ``GeneralSheaf`` scales as ``d^2`` per edge and is the one to drop
first if a sweep runs out of memory.

Graph classification needs one extra step. NSD is a transductive full-graph model: it
takes ``edge_index`` and ``graph_size`` at construction and reshapes by the latter
throughout. PyG batching hands it a different graph on every step. Their own
``update_edge_index`` (models/sheaf_base.py:51-54) exists for exactly this, but it
holds ``graph_size`` fixed, so :func:`_rebind_graph` sets the node count alongside it.
That is safe because no parameter is shaped by ``graph_size`` -- ``epsilons`` is
``(final_d, 1)`` and is broadcast with ``.tile`` (disc_models.py:50, 90), and the sheaf
learners are per-edge. Only the Laplacian builder's index tables depend on it, and
those are rebuilt per batch by their own code path.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, to_undirected

from src.baselines.adapters._homogeneous_lp import fit_homogeneous_link_prediction
from src.baselines.bundles import (
    HGB_DEFAULT_FEATS_TYPE,
    GraphBundle,
    LinkBundle,
    NodeBundle,
)
from src.baselines.encoders import TypeProjectedInput
from src.baselines.heads import SharedGraphHead
from src.baselines.budget import budget_kwargs
from src.baselines import search_grids as grids
from src.baselines.protocol import (
    GRAPH_CLASSIFICATION,
    LINK_PREDICTION,
    NODE_CLASSIFICATION,
    BaselineParams,
    UnsupportedCell,
)
from src.baselines.published import load_published_params
from src.baselines.runtime import pushd_sys_path
from src.baselines.training import (
    node_classification_criterion,
    train_full_batch_node_classification,
    train_link_prediction,
    train_graph_classification,
)
from src.utils.misc import seed_everything

NAME = "nsd"
VENDOR = "nsd"
SUPPORTED_DATASETS = frozenset(HGB_DEFAULT_FEATS_TYPE)
# YouTube, Amazon removed (spec 1.5); the derived tasks wired in Part P P2.
SUPPORTED_LP_DATASETS = frozenset({"lastfm", "pubmed", "acm_pa", "dblp_ap", "imdb_ma", "freebase_bb"})
SUPPORTED_GC_DATASETS = frozenset({"mutag", "proteins", "enzymes", "nci1"})

#: Restriction-map families, from neural-sheaf-diffusion/exp/run.py:151-157.
SHEAF_MODELS = ("DiagSheaf", "BundleSheaf", "GeneralSheaf")


@contextlib.contextmanager
def _emit_logits() -> Iterator[None]:
    """Make NSD's forward return logits instead of log-probabilities.

    Every NSD variant ends in ``F.log_softmax`` (models/disc_models.py:96 and siblings).
    The campaign's shared loss and metric path expects raw logits, and log-probabilities
    are incoherent under the BCE loss multi-label IMDB requires.

    Their ``forward`` is forty lines of sheaf construction, so copying it out to drop one
    trailing call would be far more fragile than neutralising that call. ``F.log_softmax``
    is resolved as an attribute at call time, so patching it to the identity for the
    duration of one forward is exact and reversible. Nothing else about the model changes,
    and ``argmax`` is identical either way.
    """

    original = F.log_softmax
    F.log_softmax = lambda tensor, *args, **kwargs: tensor  # type: ignore[assignment]
    try:
        yield
    finally:
        F.log_softmax = original  # type: ignore[assignment]


def _vendor_args(
    params: BaselineParams,
    *,
    input_dim: int,
    output_dim: int,
    graph_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """The dict NSD's SheafDiffusion base reads (models/sheaf_base.py:15-47)."""

    return {
        "d": int(params.get("d", 2)),
        "layers": int(params.get("layers", 2)),
        "hidden_channels": int(params.get("hidden_channels", 32)),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "graph_size": graph_size,
        "device": device,
        "dropout": float(params.get("dropout", 0.0)),
        "input_dropout": float(params.get("input_dropout", 0.0)),
        "left_weights": bool(params.get("left_weights", True)),
        "right_weights": bool(params.get("right_weights", True)),
        "add_lp": bool(params.get("add_lp", False)),
        "add_hp": bool(params.get("add_hp", False)),
        "normalised": bool(params.get("normalised", True)),
        "deg_normalised": bool(params.get("deg_normalised", False)),
        "linear": bool(params.get("linear", False)),
        "use_act": bool(params.get("use_act", True)),
        "second_linear": bool(params.get("second_linear", False)),
        "sparse_learner": bool(params.get("sparse_learner", False)),
        "edge_weights": bool(params.get("edge_weights", True)),
        "sheaf_act": str(params.get("sheaf_act", "tanh")),
        # The current H2S-NN rejects matrix_exp, but that constraint is ours, not NSD's;
        # their own default parameterisation stays available here.
        "orth": str(params.get("orth", "householder")),
        "max_t": float(params.get("max_t", 1.0)),
        "adjoint": False,
    }


def _rebind_graph(sheaf: Any, edge_index: torch.Tensor, num_nodes: int) -> None:
    """Point a constructed NSD model at a different graph.

    ``SheafDiffusion.update_edge_index`` rebuilds the Laplacian builder for a new
    edge set but leaves ``graph_size`` alone, and the builder's
    ``create_with_new_edge_index`` reuses its own ``size``. Both have to move together
    for a minibatch, so the node count is set on each before their rebuild runs. No
    parameter is shaped by it (see this module's docstring), so nothing is reallocated
    and the optimiser state stays valid across batches.
    """

    sheaf.graph_size = num_nodes
    sheaf.laplacian_builder.size = num_nodes
    sheaf.update_edge_index(edge_index)


class _NSDEncoder(torch.nn.Module):
    """Per-type projection feeding one of NSD's sheaf diffusion models."""

    def __init__(
        self,
        in_dims: list[int],
        edge_index: torch.Tensor,
        args: dict[str, Any],
        model_name: str,
    ) -> None:
        super().__init__()
        self.input_projection = TypeProjectedInput(in_dims, args["input_dim"])

        # Constructed inside the vendor context: NSD's laplacian builders subclass PyG's
        # MessagePassing, whose __init__ resolves annotations through
        # sys.modules[cls.__module__]. See pushd_sys_path's docstring.
        with pushd_sys_path(VENDOR):
            from models.disc_models import (  # type: ignore[import-not-found]
                DiscreteBundleSheafDiffusion,
                DiscreteDiagSheafDiffusion,
                DiscreteGeneralSheafDiffusion,
            )

            builders = {
                "DiagSheaf": DiscreteDiagSheafDiffusion,
                "BundleSheaf": DiscreteBundleSheafDiffusion,
                "GeneralSheaf": DiscreteGeneralSheafDiffusion,
            }
            self.sheaf = builders[model_name](edge_index, args)

    def forward(self, features_list: list[torch.Tensor]) -> torch.Tensor:
        x = self.input_projection(features_list)
        with _emit_logits():
            return self.sheaf(x)


class NSDAdapter:
    name = NAME

    def supports(self, task: str, dataset: str) -> bool:
        if task == NODE_CLASSIFICATION:
            return dataset in SUPPORTED_DATASETS
        if task == LINK_PREDICTION:
            return dataset in SUPPORTED_LP_DATASETS
        if task == GRAPH_CLASSIFICATION:
            return dataset in SUPPORTED_GC_DATASETS
        return False

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task not in (NODE_CLASSIFICATION, LINK_PREDICTION, GRAPH_CLASSIFICATION):
            raise UnsupportedCell(NAME, task, dataset, "unsupported task")
        space = {
            "lr": grids.LR,
            "weight_decay": grids.WEIGHT_DECAY,
            "hidden_channels": {"values": [32, 64]},
            "dropout": grids.DROPOUT,
            "input_dropout": grids.INPUT_DROP,  # shared axis (spec v1.15 §6)
            "d": {"values": [2, 3, 4, 5]},
            "layers": grids.SHEAF_LAYERS,
            "sheaf_model": {"values": list(SHEAF_MODELS)},
            "orth": {"values": ["householder", "cayley", "euler"]},
            "add_lp": {"values": [True, False]},
            "add_hp": {"values": [True, False]},
            "feats_type": {"values": [0, 2, 3]},
        }
        if task == LINK_PREDICTION:
            # HGB offers both decoders and lets each method choose, so it is
            # validation-selected here as for every other link-prediction baseline.
            space["decoder_type"] = {"values": ["dot", "distmult"]}
        if task == GRAPH_CLASSIFICATION:
            # feats_type is HGB's per-node-type feature fill-in; TU graphs carry one
            # uniform feature matrix, so there is nothing for it to select.
            space.pop("feats_type", None)
            space["embed_dim"] = {"values": [32, 64, 128]}
            # The Laplacian builder is rebuilt once per batch, so depth costs more
            # here than in the full-graph tasks.
            space["layers"] = {"values": [2, 3]}
        return space

    def published_params(self, task: str, dataset: str) -> BaselineParams | None:
        # exp/scripts/ covers WebKB and Wikipedia graphs only.
        return load_published_params(NAME, task, dataset)

    def fit_task(
        self,
        bundle: NodeBundle,
        params: BaselineParams,
        *,
        task: str,
        seed: int,
        device: torch.device,
    ) -> dict[str, float]:
        if task == GRAPH_CLASSIFICATION:
            return self._fit_graph_classification(
                bundle, params, seed=seed, device=device
            )
        if task == LINK_PREDICTION:
            model_name = str(params.get("sheaf_model", "DiagSheaf"))

            def build(view: Any, edge_index: torch.Tensor, embed_dim: int) -> torch.nn.Module:
                # output_dim is the embedding width rather than a class count: NSD's final
                # lin2 becomes the projection the link decoder scores pairs in.
                args = _vendor_args(
                    params,
                    input_dim=int(params.get("hidden_channels", 32)),
                    output_dim=embed_dim,
                    graph_size=view.num_nodes,
                    device=device,
                )
                return _NSDEncoder(view.in_dims, edge_index, args, model_name)

            record = fit_homogeneous_link_prediction(
                NAME, bundle, params, seed=seed, device=device, build_encoder=build
            )
            record["sheaf_model"] = model_name
            return record
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, bundle.dataset, "unsupported task")

        seed_everything(seed, device)
        feats_type = int(
            params.get("feats_type", HGB_DEFAULT_FEATS_TYPE[bundle.dataset])
        )
        view = bundle.homogeneous(feats_type=feats_type)
        features = [
            feature.coalesce().to(device) if feature.is_sparse else feature.to(device)
            for feature in view.features_list
        ]
        # NSD builds its own Laplacian and requires exact symmetric pairs, no diagonal.
        edge_index = view.edge_index_without_self_loops.to(device)

        model_name = str(params.get("sheaf_model", "DiagSheaf"))
        if model_name not in SHEAF_MODELS:
            raise ValueError(f"sheaf_model must be one of {SHEAF_MODELS}.")

        # NSD's hidden width is hidden_channels * d, and its input projection expects
        # input_dim; the per-type projection maps every node type into that width.
        args = _vendor_args(
            params,
            input_dim=int(params.get("hidden_channels", 32)),
            output_dim=bundle.num_classes,
            graph_size=view.num_nodes,
            device=device,
        )
        model = _NSDEncoder(view.in_dims, edge_index, args, model_name).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params.get("lr", 1e-2)),
            weight_decay=float(params.get("weight_decay", 5e-4)),
        )

        data = bundle.data
        labels = data.target_y.to(device)
        if bundle.is_multilabel:
            labels = labels.float()

        record = train_full_batch_node_classification(
            model=model,
            forward=lambda: model(features),
            labels=labels,
            train_idx=data.target_train_mask.nonzero(as_tuple=False).view(-1).to(device),
            val_idx=data.target_val_mask.nonzero(as_tuple=False).view(-1).to(device),
            test_idx=data.target_test_mask.nonzero(as_tuple=False).view(-1).to(device),
            optimizer=optimizer,
            criterion=node_classification_criterion(bundle.is_multilabel),
            is_multilabel=bundle.is_multilabel,
            **budget_kwargs(NODE_CLASSIFICATION, params),
            num_target=int(labels.shape[0]),
        )
        record["params_source"] = params.source
        record["feats_type"] = float(feats_type)
        record["sheaf_model"] = model_name
        record["input_projection"] = "per_type_linear"
        return record

    def _fit_graph_classification(
        self,
        bundle: GraphBundle,
        params: BaselineParams,
        *,
        seed: int,
        device: torch.device,
    ) -> dict[str, float]:
        seed_everything(seed, device)

        model_name = str(params.get("sheaf_model", "DiagSheaf"))
        if model_name not in SHEAF_MODELS:
            raise ValueError(f"sheaf_model must be one of {SHEAF_MODELS}.")

        embed_dim = int(params.get("embed_dim", 64))
        args = _vendor_args(
            params,
            # No per-type projection here: TU node features are already one uniform
            # matrix, so the model reads them at their native width.
            input_dim=bundle.in_channels,
            output_dim=embed_dim,
            # Replaced per batch by _rebind_graph; the constructor only needs a value
            # large enough for the first Laplacian index tables it builds.
            graph_size=1,
            device=device,
        )

        # Built against a single self-loop-free edge so construction succeeds; the
        # first batch rebinds it to a real graph.
        placeholder = torch.zeros((2, 0), dtype=torch.long, device=device)
        with pushd_sys_path(VENDOR):
            from models.disc_models import (  # type: ignore[import-not-found]
                DiscreteBundleSheafDiffusion,
                DiscreteDiagSheafDiffusion,
                DiscreteGeneralSheafDiffusion,
            )

            builders = {
                "DiagSheaf": DiscreteDiagSheafDiffusion,
                "BundleSheaf": DiscreteBundleSheafDiffusion,
                "GeneralSheaf": DiscreteGeneralSheafDiffusion,
            }
            sheaf = builders[model_name](placeholder, args).to(device)

        head = SharedGraphHead(
            embed_dim,
            bundle.num_classes,
            dropout=float(params.get("dropout", 0.0)),
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(sheaf.parameters()) + list(head.parameters()),
            lr=float(params.get("lr", 1e-2)),
            weight_decay=float(params.get("weight_decay", 5e-4)),
        )

        def embed_batch(batch: Any) -> torch.Tensor:
            # NSD builds its own sheaf Laplacian and expects exact symmetric pairs
            # with no diagonal, the same form the link-prediction path uses.
            edge_index, _ = remove_self_loops(batch.edge_index)
            edge_index = to_undirected(edge_index, num_nodes=batch.num_nodes)
            _rebind_graph(sheaf, edge_index, int(batch.num_nodes))
            with _emit_logits():
                return sheaf(batch.x.float())

        record = train_graph_classification(
            model=sheaf,
            head=head,
            embed_batch=embed_batch,
            loaders=bundle.loaders,
            optimizer=optimizer,
            device=device,
            **budget_kwargs(GRAPH_CLASSIFICATION, params),
        )
        record["params_source"] = params.source
        record["sheaf_model"] = model_name
        record["readout"] = "shared_mean_pool_linear"
        return record


ADAPTER = NSDAdapter()
