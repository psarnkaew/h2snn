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

import random
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


class HINormerAdapter:
    name = NAME

    def supports(self, task: str, dataset: str) -> bool:
        return task == NODE_CLASSIFICATION and dataset in SUPPORTED_DATASETS

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, dataset, "node classification only")
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


ADAPTER = HINormerAdapter()
