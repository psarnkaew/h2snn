"""HINormer (Mao et al., WWW 2023) -- graph transformer for heterogeneous information networks.

Uses the **authors' own implementation** (https://github.com/Ffffffffire/HINormer, linked as
[Code] from corresponding author Zemin Liu's homepage), not H2GB's copy.

That choice is deliberate and evidence-based. H2GB's ``hinormer_model.py`` has the entire
transformer stack commented out and computes the relation encoding ``r`` without ever
consuming it, so what runs there is a per-type projection, GCN layers, and a linear head --
weaker than the paper's own worst ablation (Fig. 2), and missing both stated contributions.
The official repository implements every checkpoint in the paper:

* per-type feature projection, ``fc_list``                        (§4.2, Eq. 9)
* GCN local structure encoder, ``GCNLayers``                      (§4.2, Eq. 10)
* relation encoder over one-hot types with learnable type-aware
  weights, ``RELayers`` seeded by ``type_emb[node_type]``         (§4.3, Eq. 12)
* transformer layers that actually run and consume ``r``          (§4.1, Eq. 8)
* attention bias ``score + beta * score_r``                       (§4.3, Eq. 13)
* GATv2-style attention, and no feed-forward block                (§4.1, Eq. 7)
* readout at sequence position 0, ``h[:, 0, :]``                  (Eq. 5)
* optional L2-normalised prediction                               (§4.4)

Protocol notes:

* The paper's split is 24:6:70 (§5.1) and it uses HGB's standard split for DBLP and IMDB --
  the same protocol this campaign enforces, so the canonical bundle applies unchanged.
* Their graph is ``adjM + adjM.T`` with self-loops *removed* (run.py), so the adapter feeds
  :attr:`~src.baselines.bundles.HomogeneousView.edge_index_without_self_loops`.
* The paper never evaluates ACM (DBLP, IMDB, Freebase, AMiner only), so ACM here is an
  extension beyond their reported results, not a reproduction.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
from pathlib import Path
from typing import Any

import torch

from src.baselines.bundles import HGB_DEFAULT_FEATS_TYPE, NodeBundle
from src.baselines.budget import budget_kwargs
from src.baselines import search_grids as grids
from src.baselines.protocol import (
    NODE_CLASSIFICATION,
    BaselineParams,
    UnsupportedCell,
)
from src.baselines.published import load_published_params
from src.baselines.runtime import pushd_sys_path
from src.baselines.runtime._dgl_compat import (
    resolve_dgl_device,
    restore_legacy_message_functions,
)
from src.baselines.training import (
    node_classification_criterion,
    train_full_batch_node_classification,
)
from src.utils.misc import seed_everything

NAME = "hinormer"
VENDOR = "hinormer"
SUPPORTED_DATASETS = frozenset(HGB_DEFAULT_FEATS_TYPE)
#: Card 1c F1 (author decision G-1): the original HINormer on the H2GB NC datasets, class B -- its own sequences
#: (run.py:100-127) and model (model.py:198-217, full-graph GCN then the sequences), the shared batch size and
#: 8,192-target epoch imposed on its training step (run.py:155 trains all targets in one forward). H2GB's own
#: `HINormerModel` is not used: its transformer and encoder are commented out (hinormer_model.py:221-222, 274-289).
H2GB_DATASETS = frozenset({"ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns"})
READS_BATCH_PLAN = True


def _build_node_sequences(
    graph: Any, num_target: int, len_seq: int
) -> torch.Tensor:
    """The D-hop context sequences of paper §4.1, as implemented in run.py:100-122.

    For each target node: start the sequence with the node itself, then repeatedly take
    the successors of the node currently at the read cursor, appending until the sequence
    is full. Sequences are built for target-type nodes only, which occupy the leading
    block of the flattened id space.
    """

    node_seq = torch.zeros(num_target, len_seq, dtype=torch.long)
    for node in range(num_target):
        cursor, read = 0, 0
        node_seq[node, cursor] = node
        cursor += 1
        start = int(node_seq[node, read].item())
        while cursor < len_seq:
            candidates = graph.successors(start).numpy().tolist()
            sampled = random.sample(candidates, max(len(candidates), 1)) if candidates else [start]
            for value in sampled:
                node_seq[node, cursor] = value
                cursor += 1
                if cursor == len_seq:
                    break
            read += 1
            start = int(node_seq[node, read].item())
    return node_seq


def build_node_sequences_for(graph: Any, nodes: torch.Tensor, len_seq: int) -> torch.Tensor:
    """`_build_node_sequences` (run.py:100-122) for the given node ids (row i = the sequence of `nodes[i]`), so the
    target type need not be the leading block (card 1c F1, H2GB graphs)."""

    node_seq = torch.zeros(int(nodes.numel()), len_seq, dtype=torch.long)
    for row, node in enumerate(nodes.tolist()):
        cursor, read = 0, 0
        node_seq[row, cursor] = node
        cursor += 1
        start = int(node_seq[row, read].item())
        while cursor < len_seq:
            candidates = graph.successors(start).numpy().tolist()
            sampled = random.sample(candidates, max(len(candidates), 1)) if candidates else [start]
            for value in sampled:
                node_seq[row, cursor] = value
                cursor += 1
                if cursor == len_seq:
                    break
            read += 1
            start = int(node_seq[row, read].item())
    return node_seq


#: Card 21 C1: directory of the on-disk sequence cache (ops sets it under the run's state directory; unset: no cache).
SEQUENCE_CACHE_ENV = "H2SNN_HINORMER_SEQ_CACHE"


def sequence_cache_key(dataset: str, graph: Any, nodes: torch.Tensor, len_seq: int) -> str:
    """What fixes `build_node_sequences_for`'s output: the dataset, the graph's edges, the target ids, `len_seq` and
    Python's `random` state at the call (its only source of randomness)."""
    digest = hashlib.sha256()
    source, destination = graph.edges()
    digest.update(repr((str(dataset), int(graph.num_nodes()), int(len_seq))).encode())
    for tensor in (source, destination, nodes):
        digest.update(tensor.detach().cpu().contiguous().to(torch.int64).numpy().tobytes())
    digest.update(pickle.dumps(random.getstate(), protocol=4))
    return digest.hexdigest()


def cached_node_sequences(graph: Any, nodes: torch.Tensor, len_seq: int, *, dataset: str,
                          cache_dir: str | Path | None = None) -> tuple[torch.Tensor, str]:
    """Card 21 C1: `build_node_sequences_for`, cached on disk by `sequence_cache_key`. A hit returns the stored
    sequences and sets Python's `random` state to the one the uncached call left (so every later draw is unchanged);
    a miss computes, then stores both. `cache_dir` None: `$H2SNN_HINORMER_SEQ_CACHE`, unset = no cache. Returns
    (sequences, "hit" | "miss" | "off")."""
    cache_dir = cache_dir if cache_dir is not None else os.environ.get(SEQUENCE_CACHE_ENV)
    if not cache_dir:
        return build_node_sequences_for(graph, nodes, len_seq), "off"
    path = Path(cache_dir) / f"hinormer_seq_{sequence_cache_key(dataset, graph, nodes, len_seq)}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            stored = pickle.load(handle)
        random.setstate(stored["random_state_after"])
        return stored["node_seq"].to(torch.long), "hit"
    node_seq = build_node_sequences_for(graph, nodes, len_seq)
    compact = node_seq.to(torch.int32) if node_seq.numel() == 0 or int(node_seq.max()) < 2 ** 31 else node_seq
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with tmp.open("wb") as handle:
        pickle.dump({"node_seq": compact, "random_state_after": random.getstate()}, handle, protocol=4)
    os.replace(tmp, path)
    return node_seq, "miss"


def allow_isolated_nodes(model: Any) -> Any:
    """Let each `GraphConv` of `model.GCNLayers` accept nodes with no in-edge (card 1d H5, author decision 1a).

    HINormer builds its graph as `adjM + adjM.T` with self-loops removed (run.py:95-96) and never adds them back, so a
    node with no non-self edge has in-degree 0; HGB Freebase has 45 such book nodes. DGL's `GraphConv` (model.py:188,
    default `allow_zero_in_degree=False`) refuses that graph. With the flag such a node receives no message (its degree
    normalisation is clamped at 1); on a graph without such nodes the outputs are unchanged. The graph is not changed
    and no self-loop is added."""
    for layer in model.GCNLayers:
        layer.set_allow_zero_in_degree(True)  # DGL 2.4: sets GraphConv._allow_zero_in_degree
    return model


def build_model(graph: Any, num_classes: int, in_dims: list[int], num_types: int, params: BaselineParams) -> Any:
    """Their `HINormer` (model.py) with the adapter's defaults, isolated nodes allowed (`allow_isolated_nodes`)."""
    with pushd_sys_path(VENDOR):
        from model import HINormer  # type: ignore[import-not-found]

        return allow_isolated_nodes(HINormer(graph, num_classes, in_dims, int(params.get("hidden_dim", 256)),
                                             int(params.get("num_layers", 2)), int(params.get("num_gnns", 4)),
                                             int(params.get("num_heads", 2)), float(params.get("dropout", 0.5)),
                                             temper=float(params.get("temperature", 1.0)), num_type=num_types,
                                             beta=float(params.get("beta", 1.0))))


class HINormerAdapter:
    name = NAME

    def supports(self, task: str, dataset: str) -> bool:
        return task == NODE_CLASSIFICATION and (dataset in SUPPORTED_DATASETS or dataset in H2GB_DATASETS)

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, dataset, "node classification only")
        if dataset in H2GB_DATASETS:
            from src.configs.config import MINIBATCH_BATCH_SIZES

            return {"lr": grids.LR, "weight_decay": grids.WEIGHT_DECAY, "hidden_dim": {"values": [64, 128, 256]},
                    "dropout": grids.DROPOUT, "num_layers": {"values": [2, 3, 4, 5]},
                    "num_gnns": {"values": [3, 4, 5]}, "num_heads": {"values": [2, 8]},
                    "beta": {"values": [0.1, 0.2, 0.5, 1.0, 2.0]}, "len_seq": grids.LEN_SEQ,
                    "temperature": grids.TEMPERATURE, "batch_size": {"values": list(MINIBATCH_BATCH_SIZES)}}
        # Ranges follow the paper's Appendix C: lr 1e-4, d=256, n_h=2, L in {2,3,4,5},
        # beta in {0.1,0.2,0.5,1,2}, S in [10,200], K_s=K_h 3-5 by dataset.
        return {
            "lr": grids.LR,
            "weight_decay": grids.WEIGHT_DECAY,
            "hidden_dim": {"values": [64, 128, 256]},
            "dropout": grids.DROPOUT,
            "num_layers": {"values": [2, 3, 4, 5]},
            "num_gnns": {"values": [3, 4, 5]},
            "num_heads": {"values": [2, 8]},
            "beta": {"values": [0.1, 0.2, 0.5, 1.0, 2.0]},
            "len_seq": grids.LEN_SEQ,
            "temperature": grids.TEMPERATURE,
            "feats_type": {"values": [0, 2, 3]},
        }

    def published_params(self, task: str, dataset: str) -> BaselineParams | None:
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
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, bundle.dataset, "node classification only")
        if bundle.dataset in H2GB_DATASETS:
            return self._fit_h2gb(bundle, params, seed=seed, device=device)

        device = resolve_dgl_device(device)
        seed_everything(seed, device)

        feats_type = int(
            params.get("feats_type", HGB_DEFAULT_FEATS_TYPE[bundle.dataset])
        )
        view = bundle.homogeneous(feats_type=feats_type)
        features = [
            feature.coalesce().to(device) if feature.is_sparse else feature.to(device)
            for feature in view.features_list
        ]

        import dgl

        # HINormer targets DGL 0.9.0 and calls the pre-rename fn.copy_src.
        restore_legacy_message_functions()

        # run.py builds dgl.DGLGraph(adjM + adjM.T) then removes self-loops and does NOT
        # add them back, so the self-loop-free view is the right one.
        source, destination = view.edge_index_without_self_loops
        graph = dgl.graph(
            (source.cpu(), destination.cpu()), num_nodes=view.num_nodes
        )

        data = bundle.data
        num_target = int(data.target_y.shape[0])
        len_seq = int(params.get("len_seq", 50))
        node_seq = _build_node_sequences(graph, num_target, len_seq).to(device)

        graph = graph.to(device)
        node_type = view.node_type.to(device)
        type_emb = torch.eye(len(view.node_type_names)).to(device)

        with pushd_sys_path(VENDOR):
            from model import HINormer  # type: ignore[import-not-found]

            model = HINormer(
                graph,
                bundle.num_classes,
                view.in_dims,
                int(params.get("hidden_dim", 256)),
                int(params.get("num_layers", 2)),
                int(params.get("num_gnns", 4)),
                int(params.get("num_heads", 2)),
                float(params.get("dropout", 0.5)),
                temper=float(params.get("temperature", 1.0)),
                num_type=len(view.node_type_names),
                beta=float(params.get("beta", 1.0)),
            ).to(device)
            allow_isolated_nodes(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params.get("lr", 1e-4)),
            weight_decay=float(params.get("weight_decay", 0.0)),
        )

        labels = data.target_y.to(device)
        if bundle.is_multilabel:
            labels = labels.float()
        use_l2norm = bool(params.get("l2norm", True))

        record = train_full_batch_node_classification(
            model=model,
            forward=lambda: model(features, node_seq, type_emb, node_type, use_l2norm),
            labels=labels,
            train_idx=data.target_train_mask.nonzero(as_tuple=False).view(-1).to(device),
            val_idx=data.target_val_mask.nonzero(as_tuple=False).view(-1).to(device),
            test_idx=data.target_test_mask.nonzero(as_tuple=False).view(-1).to(device),
            optimizer=optimizer,
            criterion=node_classification_criterion(bundle.is_multilabel),
            is_multilabel=bundle.is_multilabel,
            **budget_kwargs(NODE_CLASSIFICATION, params),
        )
        record["params_source"] = params.source
        record["feats_type"] = float(feats_type)
        record["len_seq"] = float(len_seq)
        record["device"] = device.type
        record["implementation"] = "official (Ffffffffire/HINormer)"
        return record


    def _fit_h2gb(self, bundle: NodeBundle, params: BaselineParams, *, seed: int,
                  device: torch.device) -> dict[str, float]:
        """Card 1c F1: their sequences and model on the bundle's HGB reduction (attributes kept, identity for
        featureless types); batches of B targets' sequences per step (`minibatch.train_index_minibatch`)."""
        import dgl

        from src.baselines.minibatch import minibatch_plan, train_index_minibatch

        device = resolve_dgl_device(device)
        seed_everything(seed, device)
        view = bundle.homogeneous(feats_type=0)
        features = [f.coalesce().to(device) if f.is_sparse else f.to(device) for f in view.features_list]
        restore_legacy_message_functions()
        source, destination = view.edge_index_without_self_loops  # as run.py: no self-loops added back
        graph = dgl.graph((source.cpu(), destination.cpu()), num_nodes=view.num_nodes)
        data = bundle.data
        node_seq, seq_cache = cached_node_sequences(graph, data.target_indices.cpu(), int(params.get("len_seq", 50)),
                                                    dataset=bundle.dataset)  # card 21 C1
        graph = graph.to(device)
        node_type = view.node_type.to(device)
        type_emb = torch.eye(len(view.node_type_names)).to(device)
        model = build_model(graph, bundle.num_classes, view.in_dims, len(view.node_type_names), params).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("lr", 1e-4)),
                                      weight_decay=float(params.get("weight_decay", 0.0)))
        use_l2norm = bool(params.get("l2norm", True))
        plan = minibatch_plan(bundle.dataset, int(params.get("batch_size", 256)))
        record = train_index_minibatch(
            model=model,
            forward_rows=lambda idx: model(features, node_seq[idx.cpu()].to(device), type_emb, node_type, use_l2norm),
            labels=data.target_y.to(device), train_idx=data.target_train_mask.nonzero().view(-1),
            val_idx=data.target_val_mask.nonzero().view(-1), test_idx=data.target_test_mask.nonzero().view(-1),
            optimizer=optimizer, criterion=torch.nn.functional.cross_entropy, plan=plan, base_seed=seed,
            report=params.report, run_seed=bundle.metadata.get("seed"))  # card 1f U5.3 (PDNS)
        record["params_source"] = params.source
        record["batch_size"] = float(plan.batch_size)
        record["implementation"] = "official (Ffffffffire/HINormer), card 1c F1"
        record["sequence_cache"] = seq_cache
        return record


ADAPTER = HINormerAdapter()
