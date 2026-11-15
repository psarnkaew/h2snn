"""Part M3b B4: label-leak audit of a node-classification message graph (F18 style).

Spec F18 (v1.10) audits a link-prediction task by (i) an attribute inventory and (ii) a fingerprint probe against
a control. For a node-classification label the same two parts read:

(i) **Inventory**: every node type and edge type of the message graph; a node type or relation whose name
    marks it as the label's source (OAG: `venue`, `PV_*`, `H2GB/datasets/oag_dataset.py:138-150`) fails.
(ii) **Fingerprint probe**: a relation fingerprints the label when the neighbours it gives a target node identify
    the target's label. For every relation with a target-type end, each node at the other end ("hub"; for a
    target-target relation, the source node) collects its labelled target neighbours; over the hubs with at
    least two of them, the *majority share* is (sum over hubs of the largest same-label count) / (sum of the
    labelled-neighbour counts). A venue node has share 1.0 (each paper has one venue). Control: the same
    statistic after permuting the labels among the labelled targets (seeded), the share a relation has by
    chance. A relation **carries the label** when its share is >= `CARRIER_SHARE` (0.95, fixed before the audit
    was run on any real data) — a stop for that dataset (card B4).

Edges are read as stored on the graph H2GB trains on (`dataset[0]`, which is what H2S-NN's `build_dataset`
and the baseline bundles flatten); reverse copies are audited as their own relation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

#: A relation whose majority share is at least this carries the label (fixed in advance, B4).
CARRIER_SHARE = 0.95
#: Name fragments that mark the label's source on OAG (venue nodes, paper-venue relations).
LABEL_SOURCE_NAMES: tuple[str, ...] = ("venue", "PV_")


@dataclass(frozen=True, slots=True)
class RelationProbe:
    relation: tuple[str, str, str]
    hubs: int                 # hubs with >= 2 labelled target neighbours
    neighbours: int           # labelled target neighbours summed over those hubs
    covered_targets: int      # distinct labelled targets adjacent to such a hub
    majority_share: float
    control_share: float      # the same with permuted labels

    @property
    def carries_label(self) -> bool:
        return self.hubs > 0 and self.majority_share >= CARRIER_SHARE


def majority_share(hub: torch.Tensor, label: torch.Tensor) -> tuple[float, int, int, torch.Tensor]:
    """`hub[i]` and `label[i]` of each (hub, labelled target) incidence. Returns (share, hubs, incidences,
    mask of incidences on hubs with >= 2 labelled neighbours)."""
    if hub.numel() == 0:
        return float("nan"), 0, 0, torch.zeros(0, dtype=torch.bool)
    num_labels = int(label.max()) + 1
    key = hub.long() * num_labels + label.long()
    _, key_inverse, key_counts = torch.unique(key, return_inverse=True, return_counts=True)
    hubs, hub_inverse, hub_counts = torch.unique(hub, return_inverse=True, return_counts=True)
    keep_hub = hub_counts >= 2
    keep = keep_hub[hub_inverse]
    if not bool(keep.any()):
        return float("nan"), 0, 0, keep
    # largest same-label count per hub
    pair_hub = torch.zeros(key_counts.numel(), dtype=torch.long)
    pair_hub[key_inverse] = hub_inverse
    best = torch.zeros(hubs.numel(), dtype=torch.long).scatter_reduce(0, pair_hub, key_counts, "amax")
    share = float(best[keep_hub].sum()) / float(hub_counts[keep_hub].sum())
    return share, int(keep_hub.sum()), int(hub_counts[keep_hub].sum()), keep


def probe_relations(data: "HeteroData", target: str, seed: int = 0) -> list[RelationProbe]:
    y = data[target].y
    if y.dim() != 1:
        raise ValueError("the probe reads single-label targets")
    labelled = y >= 0
    generator = torch.Generator().manual_seed(seed)
    permuted = y.clone()
    idx = torch.nonzero(labelled).flatten()
    permuted[idx] = y[idx][torch.randperm(idx.numel(), generator=generator)]
    results = []
    for relation in data.edge_types:
        src, _, dst = relation
        if target not in (src, dst):
            continue
        edge_index = data[relation].edge_index
        # target end and hub end (a target-target relation: the source is the hub)
        tgt_row, hub_row = (1, 0) if dst == target else (0, 1)
        targets, hubs = edge_index[tgt_row], edge_index[hub_row]
        keep = labelled[targets]
        targets, hubs = targets[keep], hubs[keep]
        share, n_hubs, n_inc, on_hub = majority_share(hubs, y[targets])
        control, _, _, _ = majority_share(hubs, permuted[targets])
        covered = int(torch.unique(targets[on_hub]).numel()) if on_hub.numel() else 0
        results.append(RelationProbe(tuple(map(str, relation)), n_hubs, n_inc, covered, share, control))
    return results


def inventory(data: "HeteroData") -> dict[str, list]:
    """Node types, relations and attributes; names that mark the label source."""
    flagged = [str(t) for t in data.node_types if any(n.lower() in str(t).lower() for n in LABEL_SOURCE_NAMES)]
    flagged += ["|".join(map(str, r)) for r in data.edge_types
                if any(n.lower() in "|".join(map(str, r)).lower() for n in LABEL_SOURCE_NAMES)]
    return {"node_types": [str(t) for t in data.node_types],
            "edge_types": ["|".join(map(str, r)) for r in data.edge_types],
            "attributes": {str(t): sorted(data[t].keys()) for t in data.node_types},
            "label_source_names_present": flagged}


def audit(data: "HeteroData", target: str, seed: int = 0) -> dict:
    """Inventory + probe; `passed` is False when a label-source name is present or a relation carries the label."""
    inv = inventory(data)
    probes = probe_relations(data, target, seed)
    carriers = ["|".join(p.relation) for p in probes if p.carries_label]
    return {"inventory": inv, "probes": probes, "carriers": carriers,
            "passed": not inv["label_source_names_present"] and not carriers}


__all__ = ["CARRIER_SHARE", "LABEL_SOURCE_NAMES", "RelationProbe", "audit", "inventory", "majority_share",
           "probe_relations"]
