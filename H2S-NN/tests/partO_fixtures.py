"""Synthetic graphs with ACM's schema for the Part O tests (card `docs/H2S-NN_partO_card.md`).

`acm_like_hetero` builds a `HeteroData` with ACM's node types and HGB's eight ACM relations under the
names PyG's `HGBDataset` gives them (`paper__cite__paper`, `paper__ref__paper` = its reverse,
`paper__to__author` / `author__to__paper`, subject, term), labels and masks on `paper`. No real
data are read. `patched_acm_loader` makes `build_dataset` return such a graph for
`dataset_name: acm`, so the real data pipeline (fact layer, WISP or schema draw, labels) runs on it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import torch

# ACM in HGB (VERIFIED-RUN in Part O: `data/NC/acm` through `_load_hetero_data`): node counts.
ACM_NODE_COUNTS: dict[str, int] = {"paper": 3025, "author": 5959, "subject": 56, "term": 1902}


def _pairs(generator: torch.Generator, num_src: int, num_dst: int, per_src: int,
           offset_avoid_self: bool = False) -> torch.Tensor:
    """`[2, E]` distinct (src, dst) pairs: `per_src` distinct targets per source (no self-pairs
    when `offset_avoid_self`)."""
    rows, cols = [], []
    for s in range(num_src):
        choices = torch.randperm(num_dst, generator=generator)
        if offset_avoid_self:
            choices = choices[choices != s]
        picked = choices[:per_src]
        rows.append(torch.full((picked.numel(),), s, dtype=torch.long))
        cols.append(picked)
    return torch.stack([torch.cat(rows), torch.cat(cols)])


def acm_like_hetero(num_papers: int = 24, num_authors: int = 14, num_subjects: int = 4,
                    num_terms: int = 10, cites_per_paper: int = 2, authors_per_paper: int = 2,
                    terms_per_paper: int = 3, num_classes: int = 3, feature_dim: int = 6,
                    seed: int = 0):
    """ACM-shaped `HeteroData`: every paper cites `cites_per_paper` others, has
    `authors_per_paper` authors, one subject and `terms_per_paper` terms; reverse relations stored
    as in HGB (`ref` is the transpose of `cite`)."""
    from torch_geometric.data import HeteroData

    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    counts = {"paper": num_papers, "author": num_authors, "subject": num_subjects,
              "term": num_terms}
    for node_type, count in counts.items():
        data[node_type].x = torch.randn(count, feature_dim, generator=generator)
        data[node_type].num_nodes = count
    data["paper"].y = torch.randint(num_classes, (num_papers,), generator=generator)
    # every class present among the papers (the label code reads max + 1 as the class count)
    data["paper"].y[:num_classes] = torch.arange(num_classes)
    order = torch.randperm(num_papers, generator=generator)
    train = torch.zeros(num_papers, dtype=torch.bool)
    test = torch.zeros(num_papers, dtype=torch.bool)
    train[order[: num_papers * 3 // 5]] = True
    test[order[num_papers * 3 // 5:]] = True
    data["paper"].train_mask = train
    data["paper"].test_mask = test

    cite = _pairs(generator, num_papers, num_papers, cites_per_paper, offset_avoid_self=True)
    writes = _pairs(generator, num_papers, num_authors, authors_per_paper)
    subject = torch.stack([torch.arange(num_papers),
                           torch.randint(num_subjects, (num_papers,), generator=generator)])
    terms = _pairs(generator, num_papers, num_terms, terms_per_paper)
    data["paper", "cite", "paper"].edge_index = cite
    data["paper", "ref", "paper"].edge_index = cite.flip(0)
    data["paper", "to", "author"].edge_index = writes
    data["author", "to", "paper"].edge_index = writes.flip(0)
    data["paper", "to", "subject"].edge_index = subject
    data["subject", "to", "paper"].edge_index = subject.flip(0)
    data["paper", "to", "term"].edge_index = terms
    data["term", "to", "paper"].edge_index = terms.flip(0)
    return data


@contextlib.contextmanager
def patched_acm_loader(hetero) -> Iterator[None]:
    """`build_dataset` returns `hetero` for every configuration (used with `dataset_name: acm`)."""
    from unittest import mock

    with mock.patch("src.data.dataset._load_hetero_data", lambda config: hetero):
        yield


def save_hetero(hetero, path: Path) -> None:
    torch.save(hetero, path)


def load_hetero(path: Path):
    return torch.load(path, weights_only=False)
