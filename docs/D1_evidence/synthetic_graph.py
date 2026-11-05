"""Random paper/author heterogeneous graph with PAP (L=2) and PAPAP (L=4) path faces.

Used only by the D1 verification scripts in this scratchpad. Mirrors the PreparedH2SData
contract used by tests/test_models.py::_make_prepared_data, but large enough that every
relation, both path lengths, and many instances per relation sequence occur.
"""

from __future__ import annotations

import torch

from src.data.structures import FaceGroup, PreparedH2SData


def _walks(
    out_edges: dict[int, list[tuple[int, int]]],
    starts: list[int],
    length: int,
    max_faces: int,
    generator: torch.Generator,
) -> tuple[list[list[int]], list[list[int]]]:
    seen: set[tuple[int, ...]] = set()
    nodes_out: list[list[int]] = []
    edges_out: list[list[int]] = []
    attempts = 0
    while len(nodes_out) < max_faces and attempts < 50 * max_faces:
        attempts += 1
        start = starts[int(torch.randint(len(starts), (1,), generator=generator))]
        nodes = [start]
        edges: list[int] = []
        ok = True
        for _ in range(length):
            options = out_edges[nodes[-1]]
            if not options:
                ok = False
                break
            edge_id, nxt = options[int(torch.randint(len(options), (1,), generator=generator))]
            if nxt in nodes:
                ok = False
                break
            nodes.append(nxt)
            edges.append(edge_id)
        if not ok or tuple(nodes) in seen:
            continue
        seen.add(tuple(nodes))
        nodes_out.append(nodes)
        edges_out.append(edges)
    return nodes_out, edges_out


def make_random_pap_data(
    seed: int = 0,
    num_papers: int = 24,
    num_authors: int = 16,
    authors_per_paper: int = 3,
    max_faces: int = 40,
    feature_dim: int = 6,
    num_classes: int = 3,
) -> PreparedH2SData:
    generator = torch.Generator().manual_seed(seed)
    num_nodes = num_papers + num_authors
    src: list[int] = []
    dst: list[int] = []
    rel: list[int] = []
    for paper in range(num_papers):
        authors = torch.randperm(num_authors, generator=generator)[:authors_per_paper]
        for author in authors.tolist():
            author_id = num_papers + author
            src.append(paper)
            dst.append(author_id)
            rel.append(0)  # paper -> author
            src.append(author_id)
            dst.append(paper)
            rel.append(1)  # author -> paper
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rel, dtype=torch.long)

    out_edges: dict[int, list[tuple[int, int]]] = {node: [] for node in range(num_nodes)}
    for edge_id, (u, v) in enumerate(zip(src, dst)):
        out_edges[u].append((edge_id, v))

    papers = list(range(num_papers))
    groups: list[FaceGroup] = []
    closures: list[torch.Tensor] = []
    for name, length, type_seq in (
        ("PAP", 2, ("paper", "author", "paper")),
        ("PAPAP", 4, ("paper", "author", "paper", "author", "paper")),
    ):
        node_seqs, edge_seqs = _walks(out_edges, papers, length, max_faces, generator)
        face_node_index = torch.tensor(node_seqs, dtype=torch.long)
        face_edge_index = torch.tensor(edge_seqs, dtype=torch.long)
        closure = torch.stack([face_node_index[:, -1], face_node_index[:, 0]], dim=1)
        groups.append(
            FaceGroup(
                template_name=name,
                node_type_sequence=type_seq,
                face_node_index=face_node_index,
                face_edge_index=face_edge_index,
                relation_index=edge_type[face_edge_index],
                face_edge_sign=torch.ones(face_edge_index.shape),
                closure_edge_index=closure,
            )
        )
        closures.append(closure.t())

    x = torch.randn(num_nodes, feature_dim, generator=generator)
    node_type = torch.cat(
        [torch.zeros(num_papers, dtype=torch.long), torch.ones(num_authors, dtype=torch.long)]
    )
    target_indices = torch.arange(num_papers)
    target_y = torch.randint(num_classes, (num_papers,), generator=generator)
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    y[:num_papers] = target_y
    split = torch.arange(num_papers) % 3
    train_t, val_t, test_t = split == 0, split == 1, split == 2
    pad = torch.zeros(num_authors, dtype=torch.bool)
    return PreparedH2SData(
        x=x,
        y=y,
        node_type=node_type,
        node_type_names=("paper", "author"),
        node_slices={"paper": (0, num_papers), "author": (num_papers, num_nodes)},
        feature_dims={"paper": feature_dim, "author": feature_dim},
        edge_index=edge_index,
        edge_type=edge_type,
        relation_names=("paper__to__author", "author__to__paper"),
        relation_tuples=(("paper", "to", "author"), ("author", "to", "paper")),
        train_mask=torch.cat([train_t, pad]),
        val_mask=torch.cat([val_t, pad]),
        test_mask=torch.cat([test_t, pad]),
        target_node_type="paper",
        target_indices=target_indices,
        target_y=target_y,
        target_train_mask=train_t,
        target_val_mask=val_t,
        target_test_mask=test_t,
        num_classes=num_classes,
        is_multilabel=False,
        face_groups=tuple(groups),
        closure_edge_index=torch.cat(closures, dim=1),
        metrics=torch.zeros(8),
    )
