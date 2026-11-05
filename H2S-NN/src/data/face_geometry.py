"""Geometric preconditions a face group must satisfy for the 2-complex construction.

The transport construction in `src.models.sheaf_maps.compose_face_transport` assigns one
transport `T_w` per node *position* along a face's boundary, and relies on every boundary
edge being traversed in its stored `src -> tgt` direction (which is what makes the uniform
`+1` incidence sign correct). Both preconditions are properties of enumeration, not of the
data, so they are checked here rather than assumed.
"""

from __future__ import annotations

import torch


def validate_face_orientation(
    face_node_index: torch.Tensor,
    face_edge_index: torch.Tensor,
    edge_index: torch.Tensor,
    template_name: str,
) -> None:
    """Assert every path edge is traversed along its stored direction, and paths are simple.

    Uniform `face_edge_sign = +1` is only the correct boundary map because all three
    enumerators walk forward through adjacency built from each relation's stored
    `src -> dst`. If reverse traversal is ever introduced (e.g. by symmetrising adjacency),
    the incidence signs must alternate; this check makes that fail loudly instead of
    silently producing a wrong `delta1`.

    Args:
        face_node_index: [num_faces, L+1] node sequence per instance.
        face_edge_index: [num_faces, L] path edge ids (closure edge excluded).
        edge_index: [2, num_edges] global edge list.
        template_name: used only in error messages.
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

    actual_src = edge_index[0][face_edge_index]
    actual_tgt = edge_index[1][face_edge_index]
    if not bool((actual_src == expected_src).all() and (actual_tgt == expected_tgt).all()):
        raise ValueError(
            f"face group '{template_name}': some boundary edge is not traversed along its "
            "stored src->tgt direction, so uniform face_edge_sign=+1 is no longer a valid "
            "boundary map. Alternate the incidence signs, or keep enumeration forward-only."
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

    A repeat at the endpoints (`v_0 == v_L`) degenerates the closure edge into a self-loop;
    an interior repeat over-determines the transport. Both are rejected.
    """
    if face_node_index.numel() == 0:
        return True
    ordered, _ = torch.sort(face_node_index, dim=1)
    return not bool((ordered[:, 1:] == ordered[:, :-1]).any())
