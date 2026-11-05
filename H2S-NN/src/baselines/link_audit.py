"""What the link-prediction encoder can already see about a held-out edge.

HGB's link-prediction split is not edge-disjoint in the way "transductive split" usually
implies, and every model trained under it inherits the same two effects. Both are
properties of the benchmark, not of this campaign, so the campaign reproduces them rather
than repairing them -- repairing them would make our numbers incomparable to every
published HGB result. What the campaign does add is a measurement, stamped on each link
run record, so a claim about link prediction can state the exposure instead of inheriting
it silently.

**Reverse counterparts.** The message-passing graph is HGB's ``gen_edge_list(dl,
reverse=True)`` reduction (HGB/LP/benchmark/methods/GNN/utils.py:15): every training edge
is inserted in both directions. A held-out edge ``(u, v)`` whose reverse ``(v, u)`` stayed
in training therefore has its endpoints adjacent in the graph the encoder reads. On
Amazon this is about a fifth of the held-out edges, because its two relations (co-view,
co-purchase) are close to symmetric.

**Forward duplicates.** HGB splits each relation independently
(``data_loader.get_train_valid_pos``), so the same directed pair can be held out under one
relation and kept under another. Those pairs are visible to the encoder no matter how the
graph is reduced.

The counts are reported per split, over *unique directed pairs*, against the directed
training edge list the bundle actually carries.
"""

from __future__ import annotations

from typing import Any

import torch


def _pair_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    source, destination = edge_index.detach().cpu()
    return set(zip(source.tolist(), destination.tolist()))


def link_split_overlap(data: Any) -> dict[str, float]:
    """Count how much of each held-out split the training graph already exposes.

    Returns, for ``val`` and ``test``: the number of unique held-out directed pairs, how
    many of those pairs are themselves training edges (forward duplicates), and how many
    have their reverse in training (reverse counterparts, visible once the graph is
    symmetrised). Fractions are included so a reader does not have to divide.
    """

    training = _pair_set(data.edge_index)
    overlap: dict[str, float] = {}
    for split in ("val", "test"):
        held_out = _pair_set(getattr(data, f"{split}_pos_edge_index"))
        forward = len(held_out & training)
        reverse = len({(v, u) for u, v in held_out} & training)
        total = max(len(held_out), 1)
        overlap[f"{split}_unique_pairs"] = float(len(held_out))
        overlap[f"{split}_forward_in_train"] = float(forward)
        overlap[f"{split}_reverse_in_train"] = float(reverse)
        overlap[f"{split}_forward_in_train_fraction"] = forward / total
        overlap[f"{split}_reverse_in_train_fraction"] = reverse / total
    return overlap


def record_link_split_overlap(record: dict[str, float], data: Any) -> None:
    """Stamp :func:`link_split_overlap` onto a finished link run record, in place."""

    for key, value in link_split_overlap(data).items():
        record[f"split_overlap/{key}"] = value


__all__ = ["link_split_overlap", "record_link_split_overlap"]
