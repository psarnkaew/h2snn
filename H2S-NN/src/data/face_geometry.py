"""Geometric preconditions a face group must satisfy for the 2-complex construction.

The path transport (`src.models.map_operators.PathTransportOperator`) assigns one
transport `T_w` per node *position* along a path, and relies on every hop matching its
sign: `face_edge_sign = +1` means the hop's fact is stored `v_{j-1} -> v_j` (step
`(F_src)^-1 F_tgt`), `-1` means it is stored `v_j -> v_{j-1}` (step `(F_tgt)^-1 F_src`).
Both preconditions are properties of path building, not of the data, so they are checked
here rather than assumed.
"""

from __future__ import annotations

import torch


def validate_face_orientation(
    face_node_index: torch.Tensor,
    face_edge_index: torch.Tensor,
    edge_index: torch.Tensor,
    template_name: str,
    face_edge_sign: torch.Tensor | None = None,
) -> None:
    """Assert every hop is stored as its sign says, signs are +-1, and paths are simple.

    A hop with sign +1 must be stored `v_{j-1} -> v_j`, a hop with sign -1 must be stored
    `v_j -> v_{j-1}`; anything else would silently produce a wrong transport.

    Args:
        face_node_index: [num_faces, L+1] node sequence per instance.
        face_edge_index: [num_faces, L] path edge ids (closure edge excluded).
        edge_index: [2, num_edges] global edge list.
        template_name: used only in error messages.
        face_edge_sign: [num_faces, L] hop signs in {+1, -1}; None means all +1.
    """
    if face_node_index.numel() == 0:
        return

    expected_src = face_node_index[:, :-1]
    expected_tgt = face_node_index[:, 1:]
    if face_edge_index.shape != expected_src.shape:
        raise ValueError(
            f"face group '{template_name}': face_edge_index has shape "
            f"{tuple(face_edge_index.shape)} but the node sequence implies "
            f"{tuple(expected_src.shape)} path edges."
        )

    if face_edge_sign is not None:
        if face_edge_sign.shape != face_edge_index.shape:
            raise ValueError(
                f"face group '{template_name}': face_edge_sign has shape "
                f"{tuple(face_edge_sign.shape)}, expected {tuple(face_edge_index.shape)}."
            )
        if not bool(((face_edge_sign == 1) | (face_edge_sign == -1)).all()):
            raise ValueError(f"face group '{template_name}': face_edge_sign must be +1 or -1.")
        reversed_hop = face_edge_sign < 0
        expected_src, expected_tgt = (
            torch.where(reversed_hop, expected_tgt, expected_src),
            torch.where(reversed_hop, expected_src, expected_tgt),
        )

    actual_src = edge_index[0][face_edge_index]
    actual_tgt = edge_index[1][face_edge_index]
    if not bool((actual_src == expected_src).all() and (actual_tgt == expected_tgt).all()):
        raise ValueError(
            f"face group '{template_name}': some hop's fact is not stored in the direction "
            "its face_edge_sign states (+1: v_{j-1} -> v_j, -1: v_j -> v_{j-1}), so the "
            "transport step would be wrong."
        )

    if not is_simple(face_node_index):
        raise ValueError(
            f"face group '{template_name}': a face instance revisits a node. The transport "
            "construction assigns one map per node position, so a repeat would need two "
            "conflicting maps (equivalently, trivial holonomy around the boundary). Filter "
            "enumeration to simple paths."
        )


def validate_relation_sequence_constant(
    relation_index: torch.Tensor,
    template_name: str,
) -> None:
    """Assert every instance in a face group walks the same relation sequence.

    A face group materialises one schema `(t_0, r_1, ..., r_L, t_L)`, so `(r_1..r_L)` is
    fixed and `relation_index` should be one row repeated. The `concat` face encoder relies
    on that: it reads row 0 as the group's schema identity and broadcasts it, which is exact
    only if the column is constant. Like the orientation check above, this is a property of
    enumeration rather than of the data, so it is asserted rather than assumed.

    Args:
        relation_index: [num_faces, L] relation id per path position.
        template_name: used only in error messages.
    """
    if relation_index.numel() == 0:
        return
    if not bool((relation_index == relation_index[0]).all()):
        raise ValueError(
            f"face group '{template_name}': relation ids vary across instances, so the "
            "group mixes schemas. The concat face encoder takes row 0 as the group's schema "
            "identity, which would then be wrong for the other instances. Split the group "
            "per relation sequence, or use model.face_encoder='lstm'."
        )


def is_simple(face_node_index: torch.Tensor) -> bool:
    """True when no instance repeats a node, endpoints included.

    Any repeat (endpoints included) would need two conflicting transports at one node.
    Both endpoint and interior repeats are rejected.
    """
    if face_node_index.numel() == 0:
        return True
    ordered, _ = torch.sort(face_node_index, dim=1)
    return not bool((ordered[:, 1:] == ordered[:, :-1]).any())
