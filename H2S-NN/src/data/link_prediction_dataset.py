from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.configs.config import TrainConfig
from src.data.fact_layer import (
    FactLayer,
    apply_fact_layer,
    build_fact_layer,
    held_out_fact_leaks,
    inverse_relations_typed,
)
from src.data.graph_faces import build_typed_path_face_groups
from src.data.hetero_flatten import flatten_edges, stack_node_features
from src.data.link_pred_structures import PreparedLinkPredData
from src.data.path_sampler import build_free_path_groups

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


_LOCAL_HGB_LINK_DATASETS = {
    "pubmed": {
        "folder": "PubMed",
        "node_type_names": {
            0: "pubmed_type_0",
            1: "disease",
            2: "pubmed_type_2",
            3: "pubmed_type_3",
        },
    },
    "youtube": {
        "folder": "youtube",
        "node_type_names": {
            0: "user",
        },
        # All five user-user interaction types are targeted, because all five appear in
        # link.dat.test: relation 0 holds 876 of its 131,007 test edges, so targeting
        # relation 0 alone scored 0.7% of the benchmark -- and its smallest relation by a
        # factor of ~40. Metrics are macro-averaged per relation, which is what every HGB
        # link-prediction runner reports on a multi-relation target.
        "target_relation_ids": (0, 1, 2, 3, 4),
    },
    "lastfm": {
        "folder": "LastFM",
        "node_type_names": {
            0: "user",
            1: "artist",
            2: "tag",
        },
    },
    "amazon": {
        "folder": "amazon",
        "node_type_names": {
            0: "product",
        },
        # HGB's official Amazon LP benchmark pools both relations (rel_0 =
        # co-view, rel_1 = co-purchase — both "product-product" per info.dat)
        # into one target, trained/scored jointly with relation id as a decoder
        # feature (matches Simple-HGN's run_dist.py). Every other local-HGB
        # dataset here targets exactly one relation id, inferred from
        # target_relation instead of listed explicitly.
        "target_relation_ids": (0, 1),
    },
    # Derived tasks (spec v1.7 §6; task card Part J, J1.2): the HGB node-classification releases
    # with the target relation split by fact 81/9/10 by scripts/lp/build_derived_lp.py. Targets:
    # acm_pa ("paper", "rel_2", "author"), dblp_ap ("author", "rel_0", "paper"), imdb_ma
    # ("movie", "rel_2", "actor"); configs name them with data.task_type: link_prediction.
    "acm_pa": {
        "folder": "lp_derived/acm_pa",
        "node_type_names": {0: "paper", 1: "author", 2: "subject", 3: "term"},
    },
    "dblp_ap": {
        "folder": "lp_derived/dblp_ap",
        "node_type_names": {0: "author", 1: "paper", 2: "term", 3: "venue"},
    },
    "imdb_ma": {
        "folder": "lp_derived/imdb_ma",
        "node_type_names": {0: "movie", 1: "director", 2: "actor", 3: "keyword"},
    },
    # Freebase BOOK-and-BOOK (spec v1.8 F10; task card J4.3): target ("book", "rel_0", "book"),
    # split 81/9/10 by unordered pair by scripts/lp/build_derived_lp.py. Self-loops stay in the
    # message graph as stored facts but are never positives (nor negatives: they are known).
    "freebase_bb": {
        "folder": "lp_derived/freebase_bb",
        "node_type_names": {0: "book", 1: "film", 2: "music", 3: "sports", 4: "people",
                            5: "location", 6: "organization", 7: "business"},
        "exclude_self_loop_positives": True,
    },
}


@dataclass(slots=True)
class _LocalHGBNodes:
    data: "HeteroData"
    raw_to_type_name: dict[int, str]
    raw_to_local: dict[int, int]
    type_names_by_id: dict[int, str]


@dataclass(slots=True)
class _LocalHGBLinkData:
    data: "HeteroData"
    train_edge_index_by_relation: dict[tuple[str, str, str], torch.Tensor]
    valid_edge_index_by_relation: dict[tuple[str, str, str], torch.Tensor]
    test_edge_index_by_relation: dict[tuple[str, str, str], torch.Tensor]
    # Every relation of link.dat before the split: rho is computed on this graph (D7,
    # decision 2), never on the split message graph.
    link_edge_index_by_relation: dict[tuple[str, str, str], torch.Tensor]
    # HGB's node.dat ids are raw and global (shared across node types — e.g.
    # LastFM's users are raw ids 0-1891, artists 1892-19523, not each restarting
    # at 0), while our own flattened graph offsets per-type LOCAL ids. Needed to
    # remap the official _ini test file's head/tail ids, which are in this same
    # raw space, before adding our own type offset.
    raw_to_local: dict[int, int]


def _load_link_pred_hetero_data(config: TrainConfig) -> "HeteroData":
    dataset_name = config.data.dataset_name.lower()
    root = Path(config.data.root)

    if dataset_name == "amazonbook":
        from torch_geometric.datasets import AmazonBook

        dataset = AmazonBook(root=str(root / "amazonbook"))
        return dataset[0]
    raise ValueError(f"Unsupported link-prediction dataset '{config.data.dataset_name}'.")


def _resolve_target_relation_ids(
    spec: dict,
    target_relation: tuple[str, str, str],
) -> tuple[int, ...]:
    """Which raw link.dat relation ids get pooled into this dataset's target.

    Explicit for datasets whose official benchmark spans several relation ids
    (Amazon); otherwise inferred from the single `rel_<id>` name in target_relation.
    """
    explicit = spec.get("target_relation_ids")
    if explicit is not None:
        return tuple(explicit)
    return (int(target_relation[1].removeprefix("rel_")),)


def _load_official_ini_test_labels(
    root: Path,
    folder: str,
    relation_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Load HGB's official fixed test triples (head, tail, label) for one relation,
    from the provider-shipped `<folder>_ini` sibling directory, if present.

    This is the literal file Simple-HGN's own LP code
    (HGB/LP/benchmark/methods/baseline/run_new.py and run_dist.py) reads in place
    of regenerating 2-hop negatives whenever it exists — using it here makes test
    metrics exactly reproducible against HGB's published Table 4, rather than an
    approximation via our own 2-hop sampler.
    """
    label_path = root / f"{folder}_ini" / f"{folder}_ini_{relation_id}_label.txt"
    if not label_path.exists():
        return None
    import numpy as np

    array = np.loadtxt(label_path, dtype=np.int64)
    head = torch.from_numpy(np.ascontiguousarray(array[0])).long()
    tail = torch.from_numpy(np.ascontiguousarray(array[1])).long()
    label = torch.from_numpy(np.ascontiguousarray(array[2])).long()
    return head, tail, label


def _offset_edge_index(edge_index: torch.Tensor, src_offset: int, dst_offset: int) -> torch.Tensor:
    offset = edge_index.long().clone()
    offset[0] += src_offset
    offset[1] += dst_offset
    return offset


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return set(map(tuple, edge_index.t().tolist()))


def _unique_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.long().reshape(2, 0)
    return torch.unique(edge_index.long().t(), dim=0).t().contiguous()


def _cat_edge_indices(parts: list[torch.Tensor]) -> torch.Tensor:
    non_empty = [part.long() for part in parts if part.numel() > 0]
    if not non_empty:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.cat(non_empty, dim=1)


def _hgb_relation_name(relation_id: int) -> str:
    return f"rel_{relation_id}"


def _parse_hgb_feature_vector(raw: str) -> list[float]:
    raw = raw.strip()
    if not raw:
        return []
    return [float(value) for value in raw.split(",")]


def _load_local_hgb_nodes(
    dataset_dir: Path,
    node_type_names: dict[int, str],
) -> _LocalHGBNodes:
    from torch_geometric.data import HeteroData

    raw_ids_by_type: dict[int, list[int]] = defaultdict(list)
    features_by_raw_id: dict[int, list[float]] = {}

    with (dataset_dir / "node.dat").open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) not in (3, 4):
                raise ValueError(f"Could not parse HGB node row: {line[:80]!r}")
            raw_id = int(parts[0])
            node_type_id = int(parts[2])
            raw_ids_by_type[node_type_id].append(raw_id)
            if len(parts) == 4:
                features = _parse_hgb_feature_vector(parts[3])
                if features:
                    features_by_raw_id[raw_id] = features

    data = HeteroData()
    raw_to_type_name: dict[int, str] = {}
    raw_to_local: dict[int, int] = {}
    type_names_by_id: dict[int, str] = {}

    for node_type_id in sorted(raw_ids_by_type):
        type_name = node_type_names.get(node_type_id, f"type_{node_type_id}")
        type_names_by_id[node_type_id] = type_name
        raw_ids = sorted(raw_ids_by_type[node_type_id])
        local_by_raw = {raw_id: idx for idx, raw_id in enumerate(raw_ids)}
        for raw_id, local_id in local_by_raw.items():
            raw_to_type_name[raw_id] = type_name
            raw_to_local[raw_id] = local_id

        data[type_name].num_nodes = len(raw_ids)
        feature_count = sum(raw_id in features_by_raw_id for raw_id in raw_ids)
        if feature_count == 0:
            continue
        if feature_count != len(raw_ids):
            raise ValueError(
                f"Node type '{type_name}' has partial features "
                f"({feature_count}/{len(raw_ids)} nodes)."
            )
        data[type_name].x = torch.tensor(
            [features_by_raw_id[raw_id] for raw_id in raw_ids],
            dtype=torch.float32,
        )

    return _LocalHGBNodes(
        data=data,
        raw_to_type_name=raw_to_type_name,
        raw_to_local=raw_to_local,
        type_names_by_id=type_names_by_id,
    )


def _read_local_hgb_link_file(
    path: Path,
    nodes: _LocalHGBNodes,
) -> dict[tuple[str, str, str], torch.Tensor]:
    rows_by_relation: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    relation_meta_by_id: dict[int, tuple[str, str, str]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"Could not parse HGB link row: {line[:80]!r}")
            src_raw = int(parts[0])
            dst_raw = int(parts[1])
            relation_id = int(parts[2])
            relation = (
                nodes.raw_to_type_name[src_raw],
                _hgb_relation_name(relation_id),
                nodes.raw_to_type_name[dst_raw],
            )
            previous = relation_meta_by_id.setdefault(relation_id, relation)
            if previous != relation:
                raise ValueError(
                    f"Relation id {relation_id} has inconsistent endpoint types: "
                    f"{previous} vs {relation}."
                )
            rows_by_relation[relation].append(
                (nodes.raw_to_local[src_raw], nodes.raw_to_local[dst_raw])
            )

    edge_index_by_relation: dict[tuple[str, str, str], torch.Tensor] = {}
    for relation, rows in rows_by_relation.items():
        if rows:
            edge_index = torch.tensor(rows, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_index_by_relation[relation] = _unique_edge_index(edge_index)
    return edge_index_by_relation


def _split_edge_index_hgb_by_head(
    edge_index: torch.Tensor,
    val_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split HGB local LP positives while keeping at least one edge per head in train."""
    if edge_index.numel() == 0:
        empty = edge_index.long().reshape(2, 0)
        return empty, empty

    pairs = sorted(set((int(src), int(dst)) for src, dst in edge_index.t().tolist()))
    rng = random.Random(seed)
    train: list[tuple[int, int]] = []
    valid: list[tuple[int, int]] = []
    valid_candidates: list[tuple[int, int]] = []
    last_src: int | None = None

    for pair in pairs:
        src, _ = pair
        if src != last_src:
            train.append(pair)
            last_src = src
            continue
        valid_candidates.append(pair)
        if rng.random() < val_ratio:
            valid.append(pair)
        else:
            train.append(pair)

    if not valid and valid_candidates:
        fallback = valid_candidates[-1]
        if fallback in train:
            train.remove(fallback)
        valid.append(fallback)

    return _pairs_to_edge_index(train), _pairs_to_edge_index(valid)


def _pairs_to_edge_index(rows: list[tuple[int, int]]) -> torch.Tensor:
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long).t().contiguous()


def _relation_id(relation: tuple[str, str, str]) -> int:
    return int(relation[1].removeprefix("rel_"))


def _file_validation_facts(
    relation: tuple[str, str, str],
    partner: tuple[str, str, str] | None,
    file_valid: dict[tuple[str, str, str], torch.Tensor],
) -> set[tuple[int, int]]:
    """The validation facts of `relation` named by `link.dat.valid`, in `relation`'s
    orientation: its own rows plus its rho-partner's rows flipped (both must agree when both
    are listed); for a self-inverse relation, each unordered pair once as (min, max)."""
    own = _edge_set(file_valid[relation]) if relation in file_valid else set()
    if partner is not None and partner != relation and partner in file_valid:
        flipped = {(v, u) for u, v in _edge_set(file_valid[partner])}
        if own and own != flipped:
            raise ValueError(
                f"link.dat.valid lists different facts for {relation} and its inverse {partner}."
            )
        own = flipped
    if partner == relation:
        if any((v, u) not in own for u, v in own):
            raise ValueError(
                f"link.dat.valid lists a self-inverse relation {relation} in one orientation only."
            )
        own = {(min(u, v), max(u, v)) for u, v in own}
    return own


def _split_validation(
    link_edges: dict[tuple[str, str, str], torch.Tensor],
    rho: dict,
    relations: list[tuple[str, str, str]],
    file_valid: dict[tuple[str, str, str], torch.Tensor] | None,
    val_ratio: float,
    split_seed: int,
) -> tuple[dict, dict]:
    """(train, validation) facts of every relation, split by FACT (D7, decision 2).

    rho comes from link.dat before splitting; a relation tied to a lower-id partner takes the
    partner's split flipped; a self-inverse relation is split over unordered pairs and both
    orientations go to the same side; a relation without a partner is split on its own rows.
    Validation is `link.dat.valid` when the file exists (spec v1.8 F9: every relation listed in
    it takes exactly those facts; a relation absent from it keeps all its facts in training),
    otherwise the carve `_split_edge_index_hgb_by_head` (seed `split_seed + id * 9973`). Both
    return their pairs in the same sorted order, so a file written from the carve reproduces it.
    """
    if file_valid is not None and set(file_valid) - set(relations):
        raise ValueError(
            f"link.dat.valid names relations absent from link.dat: "
            f"{sorted(set(file_valid) - set(relations))}."
        )
    train_edges: dict[tuple[str, str, str], torch.Tensor] = {}
    valid_edges: dict[tuple[str, str, str], torch.Tensor] = {}
    for relation in sorted(relations, key=_relation_id):
        partner = rho.get(relation)
        if partner is not None and partner != relation \
                and _relation_id(partner) < _relation_id(relation):
            continue  # filled from its lower-id partner below
        edges = link_edges.get(relation, torch.empty((2, 0), dtype=torch.long))
        if partner == relation:
            edges = edges[:, edges[0] <= edges[1]]
        if file_valid is None:
            train_edge_index, valid_edge_index = _split_edge_index_hgb_by_head(
                edge_index=edges,
                val_ratio=val_ratio,
                seed=split_seed + _relation_id(relation) * 9973,
            )
        else:
            facts = _file_validation_facts(relation, partner, file_valid)
            pairs = [tuple(pair) for pair in edges.t().tolist()]
            missing = facts - set(pairs)
            if missing:
                raise ValueError(
                    f"{len(missing)} link.dat.valid fact(s) of {relation} are not in link.dat "
                    "(HGB format: link.dat holds training and validation facts)."
                )
            train_edge_index = _pairs_to_edge_index([p for p in pairs if p not in facts])
            valid_edge_index = _pairs_to_edge_index(sorted(facts))
        if partner == relation:
            train_edge_index = _with_reversed(train_edge_index)
            valid_edge_index = _with_reversed(valid_edge_index)
        train_edges[relation] = train_edge_index
        valid_edges[relation] = valid_edge_index
        if partner is not None and partner != relation:
            train_edges[partner] = train_edge_index.flip(0)
            valid_edges[partner] = valid_edge_index.flip(0)
    return train_edges, valid_edges


def _assert_reverse_partner_is_tied(
    rho: dict,
    relations: list[tuple[str, str, str]],
    target_relations: list[tuple[str, str, str]],
) -> None:
    """Guard (spec v1.8 F9; Part J J1.3 finding 2): when a relation r' != target with the
    target's endpoint types reversed exists, it may store the target's facts reversed; if rho
    does not tie them (rho(target) is None), a held-out fact can survive as its r' row and no
    fact-level check would see it. Raise instead of splitting r and r' independently."""
    for target in target_relations:
        src_type, _, dst_type = target
        reverse_typed = [r for r in relations
                         if r != target and r[0] == dst_type and r[2] == src_type]
        if reverse_typed and rho.get(target) is None:
            raise ValueError(
                f"Target relation {target} has reverse-typed relation(s) {reverse_typed} but no "
                "exact inverse (rho is None): a held-out fact could leak into the message graph "
                "as its reverse row. Fix link.dat (every fact stored in both orientations, or "
                "none) before splitting."
            )


def _load_local_hgb_link_data(config: TrainConfig) -> _LocalHGBLinkData:
    dataset_name = config.data.dataset_name.lower()
    spec = _LOCAL_HGB_LINK_DATASETS[dataset_name]
    dataset_dir = Path(config.data.root) / str(spec["folder"])
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Expected local HGB link-prediction data at '{dataset_dir}'."
        )

    nodes = _load_local_hgb_nodes(
        dataset_dir=dataset_dir,
        node_type_names=spec["node_type_names"],  # type: ignore[arg-type]
    )
    link_edges = _read_local_hgb_link_file(dataset_dir / "link.dat", nodes)
    test_edges = _read_local_hgb_link_file(dataset_dir / "link.dat.test", nodes)

    all_relations = sorted(set(link_edges) | set(test_edges))
    # Split by FACT (D7, decision 2): see `_split_validation`. Validation is the fixed
    # link.dat.valid when the release has one (spec v1.8 F9), otherwise the carve.
    type_sizes = {name: int(nodes.data[name].num_nodes) for name in nodes.data.node_types}
    rho = inverse_relations_typed(link_edges, type_sizes)
    target_relation = tuple(config.data.target_relation)
    target_relations = [
        (target_relation[0], _hgb_relation_name(rel_id), target_relation[2])
        for rel_id in _resolve_target_relation_ids(spec, target_relation)
    ]
    _assert_reverse_partner_is_tied(rho, all_relations, target_relations)
    valid_path = dataset_dir / "link.dat.valid"
    file_valid = _read_local_hgb_link_file(valid_path, nodes) if valid_path.exists() else None
    train_edges, valid_edges = _split_validation(
        link_edges, rho, all_relations, file_valid,
        val_ratio=config.data.link_val_ratio, split_seed=config.data.acm_split_seed,
    )
    for relation in all_relations:
        nodes.data[relation].edge_index = train_edges[relation]

    for relation in test_edges:
        if relation not in nodes.data.edge_types:
            nodes.data[relation].edge_index = torch.empty((2, 0), dtype=torch.long)

    return _LocalHGBLinkData(
        data=nodes.data,
        train_edge_index_by_relation=train_edges,
        valid_edge_index_by_relation=valid_edges,
        test_edge_index_by_relation=test_edges,
        link_edge_index_by_relation=link_edges,
        raw_to_local=nodes.raw_to_local,
    )


def _with_reversed(edge_index: torch.Tensor) -> torch.Tensor:
    """Both orientations of the facts of a self-inverse relation (self-loops once)."""
    non_loop = edge_index[:, edge_index[0] != edge_index[1]]
    return torch.cat([edge_index, non_loop.flip(0)], dim=1)


def _split_edge_index_hgb(
    edge_index: torch.Tensor,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split positives into HGB-style train/val/test partitions."""
    edge_index = _unique_edge_index(edge_index)
    num_edges = edge_index.size(1)
    if num_edges < 3:
        raise ValueError("Need at least 3 positive edges for an HGB link-prediction split.")

    num_test = int(round(num_edges * test_ratio))
    num_val = int(round(num_edges * val_ratio))
    num_test = max(1, min(num_test, num_edges - 2))
    num_val = max(1, min(num_val, num_edges - num_test - 1))

    generator = torch.Generator(device=edge_index.device)
    generator.manual_seed(seed)
    perm = torch.randperm(num_edges, generator=generator, device=edge_index.device)

    test_idx = perm[:num_test]
    val_idx = perm[num_test : num_test + num_val]
    train_idx = perm[num_test + num_val :]
    return edge_index[:, train_idx], edge_index[:, val_idx], edge_index[:, test_idx]


def _replace_relation_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    relation_id: int,
    replacement_edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = edge_type != relation_id
    kept_edge_index = edge_index[:, keep]
    kept_edge_type = edge_type[keep]
    replacement_edges = replacement_edges.to(edge_index.device).long()
    replacement_type = torch.full(
        (replacement_edges.size(1),),
        relation_id,
        dtype=edge_type.dtype,
        device=edge_type.device,
    )
    return (
        torch.cat([kept_edge_index, replacement_edges], dim=1),
        torch.cat([kept_edge_type, replacement_type], dim=0),
    )


def _known_tails_by_source(edge_index: torch.Tensor) -> dict[int, set[int]]:
    known: dict[int, set[int]] = defaultdict(set)
    for src, dst in edge_index.t().tolist():
        known[int(src)].add(int(dst))
    return known


def _sample_random_tail_negatives(
    reference_pos_edge_index: torch.Tensor,
    known_positive_edge_index: torch.Tensor,
    dst_range: tuple[int, int],
    seed: int,
    neg_sampling_ratio: int = 1,
) -> torch.Tensor:
    """Sample random non-edges while preserving HGB's per-head negative count."""
    rng = random.Random(seed)
    dst_start, dst_end = dst_range
    known_by_src = _known_tails_by_source(known_positive_edge_index.cpu())
    head_counts = Counter(int(src) for src in reference_pos_edge_index[0].cpu().tolist())
    sampled: list[tuple[int, int]] = []

    for src, count in sorted(head_counts.items()):
        needed = count * neg_sampling_ratio
        known_tails = known_by_src.get(src, set())
        max_available = (dst_end - dst_start) - len(
            [tail for tail in known_tails if dst_start <= tail < dst_end]
        )
        if max_available <= 0:
            continue
        for _ in range(needed):
            for _attempt in range(100):
                tail = rng.randrange(dst_start, dst_end)
                if tail not in known_tails and tail != src:
                    sampled.append((src, tail))
                    break
            else:
                candidates = [
                    tail
                    for tail in range(dst_start, dst_end)
                    if tail not in known_tails and tail != src
                ]
                if candidates:
                    sampled.append((src, rng.choice(candidates)))

    if not sampled:
        return torch.empty((2, 0), dtype=torch.long, device=reference_pos_edge_index.device)
    return torch.tensor(sampled, dtype=torch.long, device=reference_pos_edge_index.device).t().contiguous()


def _has_two_hop_type_path(
    relation_tuples: tuple[tuple[str, str, str], ...],
    src_type: str,
    dst_type: str,
) -> bool:
    type_neighbors: dict[str, set[str]] = defaultdict(set)
    for head_type, _, tail_type in relation_tuples:
        type_neighbors[head_type].add(tail_type)
        type_neighbors[tail_type].add(head_type)
    return any(dst_type in type_neighbors[mid_type] for mid_type in type_neighbors[src_type])


def _build_undirected_adjacency(edge_index: torch.Tensor) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for src, dst in edge_index.cpu().t().tolist():
        src_i, dst_i = int(src), int(dst)
        adjacency[src_i].add(dst_i)
        adjacency[dst_i].add(src_i)
    return adjacency


def _sample_hgb_two_hop_negatives(
    reference_pos_edge_index: torch.Tensor,
    adjacency_edge_index: torch.Tensor,
    known_positive_edge_index: torch.Tensor,
    dst_range: tuple[int, int],
    seed: int,
    neg_sampling_ratio: int = 1,
    allow_random_fallback: bool = True,
) -> torch.Tensor:
    """Sample HGB hard negatives from source-node two-hop neighborhoods.

    HGB groups candidates by head/source node, ranks positives and negatives for
    MRR by that head, and uses a 1:1 positive/negative ratio for evaluation.
    This helper mirrors that grouping and excludes known positives from all
    target-relation splits.
    """
    dst_start, dst_end = dst_range
    adjacency = _build_undirected_adjacency(adjacency_edge_index)
    known_by_src = _known_tails_by_source(known_positive_edge_index.cpu())
    head_counts = Counter(int(src) for src in reference_pos_edge_index[0].cpu().tolist())
    rng = random.Random(seed)
    sampled: list[tuple[int, int]] = []

    for src, count in sorted(head_counts.items()):
        two_hop: set[int] = set()
        for mid in adjacency.get(src, set()):
            two_hop.update(adjacency.get(mid, set()))
        candidates = sorted(
            tail
            for tail in two_hop
            if dst_start <= tail < dst_end
            and tail != src
            and tail not in known_by_src.get(src, set())
        )
        needed = count * neg_sampling_ratio
        if candidates:
            sampled.extend((src, rng.choice(candidates)) for _ in range(needed))
            continue
        if allow_random_fallback:
            fallback = _sample_random_tail_negatives(
                reference_pos_edge_index=torch.full(
                    (2, count),
                    src,
                    dtype=torch.long,
                    device=reference_pos_edge_index.device,
                ),
                known_positive_edge_index=known_positive_edge_index,
                dst_range=dst_range,
                seed=rng.randrange(2**31),
                neg_sampling_ratio=neg_sampling_ratio,
            )
            if fallback.numel() > 0:
                sampled.extend((int(h), int(t)) for h, t in fallback.cpu().t().tolist())

    if not sampled:
        return torch.empty((2, 0), dtype=torch.long, device=reference_pos_edge_index.device)
    return torch.tensor(sampled, dtype=torch.long, device=reference_pos_edge_index.device).t().contiguous()


def _sample_hgb_eval_negatives(
    reference_pos_edge_index: torch.Tensor,
    adjacency_edge_index: torch.Tensor,
    known_positive_edge_index: torch.Tensor,
    relation_tuples: tuple[tuple[str, str, str], ...],
    src_type: str,
    dst_type: str,
    dst_range: tuple[int, int],
    seed: int,
) -> torch.Tensor:
    if _has_two_hop_type_path(relation_tuples, src_type, dst_type):
        negatives = _sample_hgb_two_hop_negatives(
            reference_pos_edge_index=reference_pos_edge_index,
            adjacency_edge_index=adjacency_edge_index,
            known_positive_edge_index=known_positive_edge_index,
            dst_range=dst_range,
            seed=seed,
            neg_sampling_ratio=1,
            allow_random_fallback=True,
        )
    else:
        negatives = _sample_random_tail_negatives(
            reference_pos_edge_index=reference_pos_edge_index,
            known_positive_edge_index=known_positive_edge_index,
            dst_range=dst_range,
            seed=seed,
            neg_sampling_ratio=1,
        )

    if negatives.size(1) != reference_pos_edge_index.size(1):
        raise ValueError(
            "Could not build a 1:1 HGB negative set for link prediction: "
            f"needed {reference_pos_edge_index.size(1)}, got {negatives.size(1)}."
        )
    return negatives


def _remove_edges_from_message_passing(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    relation_id: int,
    edges_to_remove: torch.Tensor,
    total_num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop held-out val/test edges of one relation from the message-passing graph.

    Vectorized via a unique integer key per (src, dst) pair — a Python-loop
    membership check would be too slow for relations with millions of edges
    (e.g. AmazonBook's ~2.9M ratings).
    """
    if edges_to_remove.size(1) == 0:
        return edge_index, edge_type

    def _keys(index: torch.Tensor) -> torch.Tensor:
        return index[0].long() * total_num_nodes + index[1].long()

    relation_mask = edge_type == relation_id
    is_removed = torch.isin(_keys(edge_index), _keys(edges_to_remove)) & relation_mask
    keep = ~is_removed
    return edge_index[:, keep], edge_type[keep]


def _assert_no_message_passing_leak(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    relation_ids: list[int],
    held_out_edges: torch.Tensor,
) -> None:
    """Confirm held-out (val/test) edges are absent from the message-passing graph,
    across every relation_id that could carry them (a relation and its mirrored
    reverse, if one exists) — the exact check that would have caught the AmazonBook
    reverse-relation leak this function guards against."""
    if held_out_edges.size(1) == 0 or not relation_ids:
        return
    relation_mask = torch.isin(edge_type, torch.tensor(relation_ids, device=edge_type.device))
    relation_edges = edge_index[:, relation_mask]
    if _edge_set(relation_edges) & _edge_set(held_out_edges):
        raise ValueError(
            "Held-out edges leak into the message-passing graph via relation_ids "
            f"{relation_ids}."
        )


def _assert_no_leakage(
    train_pos_edge_index: torch.Tensor,
    val_pos_edge_index: torch.Tensor,
    test_pos_edge_index: torch.Tensor,
) -> None:
    train_set = _edge_set(train_pos_edge_index)
    val_set = _edge_set(val_pos_edge_index)
    test_set = _edge_set(test_pos_edge_index)
    if train_set & val_set:
        raise ValueError("Train/val positive edges overlap — validation would leak into training.")
    if train_set & test_set:
        raise ValueError("Train/test positive edges overlap — test would leak into training.")
    if val_set & test_set:
        raise ValueError("Val/test positive edges overlap.")


def _flatten_relation_edges(
    edges_by_relation: dict[tuple[str, str, str], torch.Tensor],
    relation_tuples: tuple[tuple[str, str, str], ...],
    node_slices,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global `(edge_index, edge_type)` of per-relation local edges, in `relation_tuples`
    order (relations without edges contribute nothing)."""
    parts, types = [], []
    for relation_id, relation in enumerate(relation_tuples):
        local = edges_by_relation.get(relation)
        if local is None or local.numel() == 0:
            continue
        parts.append(_offset_edge_index(local, node_slices[relation[0]][0],
                                        node_slices[relation[2]][0]))
        types.append(torch.full((local.size(1),), relation_id, dtype=torch.long))
    if not parts:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.cat(parts, dim=1), torch.cat(types)


def build_link_path_groups(
    config: TrainConfig,
    fact_layer: FactLayer,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    node_type: torch.Tensor,
    seed: int,
) -> tuple:
    """Path cells on the message graph: schema-free walks (D8, decision 4; `seed` is the
    draw's seed) or the pre-D7 DFS (`path_sampler: dfs`)."""
    if config.data.path_sampler == "dfs":
        return build_typed_path_face_groups(
            edge_index=edge_index,
            edge_type=edge_type,
            node_type=node_type,
            batch=None,
            graph_id=None,
            max_paths_per_anchor=config.data.face_sample_k,
            exclude_identity_paths=config.data.exclude_identity_metapaths,
            seed=seed,
        )
    if config.data.path_sampler != "walk":
        raise ValueError("link prediction supports path_sampler walk or dfs.")
    return build_free_path_groups(
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type,
        relation_names=fact_layer.kept_names,
        self_inverse=fact_layer.self_inverse,
        tied=fact_layer.mode == "tied",
        walks_per_anchor=config.data.face_sample_k,
        seed=seed,
        length=2,
    )


def assert_no_held_out_facts(
    fact_layer: FactLayer,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    face_groups: tuple,
    held_out_edge_index: torch.Tensor,
    held_out_relation: torch.Tensor,
) -> None:
    """No held-out fact, in either orientation, in the message graph or on any path cell
    (D7, T11). Path cells reference message-graph edges by id, so the facts on every path
    are checked through the edges they use."""
    leaks = held_out_fact_leaks(fact_layer, edge_index, edge_type, held_out_edge_index,
                                held_out_relation)
    if leaks:
        raise AssertionError(
            f"{leaks} held-out fact(s) present in the message graph (either orientation)."
        )
    for group in face_groups:
        used = group.face_edge_index.reshape(-1)
        if used.numel() and int(used.max()) >= edge_index.size(1):
            raise AssertionError(f"path group {group.template_name!r} uses a non-message edge.")
        path_leaks = held_out_fact_leaks(fact_layer, edge_index[:, used], edge_type[used],
                                         held_out_edge_index, held_out_relation)
        if path_leaks:
            raise AssertionError(
                f"{path_leaks} held-out fact(s) on path group {group.template_name!r}."
            )


def build_link_prediction_dataset(config: TrainConfig) -> PreparedLinkPredData:
    dataset_name = config.data.dataset_name.lower()
    local_hgb_data: _LocalHGBLinkData | None = None
    if dataset_name in _LOCAL_HGB_LINK_DATASETS:
        local_hgb_data = _load_local_hgb_link_data(config)
        data = local_hgb_data.data
    else:
        data = _load_link_pred_hetero_data(config)

    raw_x, node_type, node_type_names, node_slices, feature_dims = stack_node_features(data)
    edge_index, edge_type, relation_names, relation_tuples = flatten_edges(data, node_slices)
    if local_hgb_data is None:
        # AmazonBook: the stored graph before our re-split; rho is computed on it.
        pre_split_edge_index, pre_split_edge_type = edge_index, edge_type
    else:
        pre_split_edge_index, pre_split_edge_type = _flatten_relation_edges(
            local_hgb_data.link_edge_index_by_relation, relation_tuples, node_slices
        )

    src_type, relation_name, dst_type = config.data.target_relation
    target_relation = (src_type, relation_name, dst_type)
    src_offset = node_slices[src_type][0]
    dst_offset = node_slices[dst_type][0]
    num_src_nodes = node_slices[src_type][1] - node_slices[src_type][0]
    num_dst_nodes = node_slices[dst_type][1] - node_slices[dst_type][0]

    edge_store = data[src_type, relation_name, dst_type]
    train_pos_edge_index = _offset_edge_index(edge_store.edge_index, src_offset, dst_offset)

    if local_hgb_data is not None:
        root = Path(config.data.root)
        spec = _LOCAL_HGB_LINK_DATASETS[dataset_name]
        target_relation_ids = _resolve_target_relation_ids(spec, target_relation)
        target_relations = [
            (src_type, _hgb_relation_name(rel_id), dst_type) for rel_id in target_relation_ids
        ]
        for relation, rel_id in zip(target_relations, target_relation_ids):
            if relation not in local_hgb_data.train_edge_index_by_relation:
                raise ValueError(
                    f"Target relation {relation} (rel_id={rel_id}) is not present in "
                    f"local HGB dataset '{config.data.dataset_name}'. Available "
                    f"relations: {sorted(local_hgb_data.train_edge_index_by_relation)}"
                )

        def _offset_local_relation(
            relation: tuple[str, str, str],
            local_edge_index: torch.Tensor,
        ) -> torch.Tensor:
            return _offset_edge_index(
                local_edge_index,
                src_offset=node_slices[relation[0]][0],
                dst_offset=node_slices[relation[2]][0],
            )

        train_neg_candidates = None
        valid_global_parts = [
            _offset_local_relation(relation, local_edge_index)
            for relation, local_edge_index in local_hgb_data.valid_edge_index_by_relation.items()
        ]
        test_global_parts = [
            _offset_local_relation(relation, local_edge_index)
            for relation, local_edge_index in local_hgb_data.test_edge_index_by_relation.items()
        ]
        known_positive_parts: list[torch.Tensor] = []
        all_relations = (
            set(local_hgb_data.train_edge_index_by_relation)
            | set(local_hgb_data.valid_edge_index_by_relation)
            | set(local_hgb_data.test_edge_index_by_relation)
        )
        for relation in sorted(all_relations):
            if relation[0] != src_type or relation[2] != dst_type:
                continue
            known_positive_parts.extend(
                [
                    _offset_local_relation(
                        relation,
                        local_hgb_data.train_edge_index_by_relation.get(
                            relation,
                            torch.empty((2, 0), dtype=torch.long),
                        ),
                    ),
                    _offset_local_relation(
                        relation,
                        local_hgb_data.valid_edge_index_by_relation.get(
                            relation,
                            torch.empty((2, 0), dtype=torch.long),
                        ),
                    ),
                    _offset_local_relation(
                        relation,
                        local_hgb_data.test_edge_index_by_relation.get(
                            relation,
                            torch.empty((2, 0), dtype=torch.long),
                        ),
                    ),
                ]
            )

        known_positives_global = _unique_edge_index(_cat_edge_indices(known_positive_parts))
        hgb_adjacency_edge_index = _cat_edge_indices(
            [edge_index, *valid_global_parts, *test_global_parts]
        )
        dst_range = (dst_offset, dst_offset + num_dst_nodes)

        # HGB's node.dat ids are raw and global (e.g. LastFM's artists start at
        # raw id 1892, not 0) — the official _ini test file uses this same raw
        # space, so its head/tail columns must go through raw_to_local before
        # adding our own per-type flattened offset.
        raw_to_local_items = sorted(local_hgb_data.raw_to_local.items())
        raw_to_local_lookup = torch.zeros(raw_to_local_items[-1][0] + 1, dtype=torch.long)
        for raw_id, local_id in raw_to_local_items:
            raw_to_local_lookup[raw_id] = local_id

        train_pos_parts: list[torch.Tensor] = []
        train_pos_relation_parts: list[torch.Tensor] = []
        val_pos_parts: list[torch.Tensor] = []
        val_pos_relation_parts: list[torch.Tensor] = []
        val_neg_parts: list[torch.Tensor] = []
        val_neg_relation_parts: list[torch.Tensor] = []
        test_pos_parts: list[torch.Tensor] = []
        test_pos_relation_parts: list[torch.Tensor] = []
        test_neg_parts: list[torch.Tensor] = []
        test_neg_relation_parts: list[torch.Tensor] = []

        for local_idx, (relation, rel_id) in enumerate(zip(target_relations, target_relation_ids)):
            relation_train_pos = _offset_local_relation(
                relation, local_hgb_data.train_edge_index_by_relation[relation]
            )
            if spec.get("exclude_self_loop_positives", False):
                relation_train_pos = relation_train_pos[
                    :, relation_train_pos[0] != relation_train_pos[1]
                ]
            relation_val_pos = _offset_local_relation(
                relation,
                local_hgb_data.valid_edge_index_by_relation.get(
                    relation, torch.empty((2, 0), dtype=torch.long)
                ),
            )

            # HGB's official fixed test triples (Simple-HGN's run_new.py/run_dist.py
            # read this file in place of regenerating negatives) when the provider
            # shipped one; otherwise fall back to our own 2-hop-or-random sampler.
            official = _load_official_ini_test_labels(root, str(spec["folder"]), rel_id)
            if official is not None:
                head, tail, label = official
                head = raw_to_local_lookup[head] + src_offset
                tail = raw_to_local_lookup[tail] + dst_offset
                pos_mask = label == 1
                neg_mask = label == 0
                relation_test_pos = torch.stack([head[pos_mask], tail[pos_mask]])
                relation_test_neg = torch.stack([head[neg_mask], tail[neg_mask]])
            else:
                relation_test_pos = _offset_local_relation(
                    relation,
                    local_hgb_data.test_edge_index_by_relation.get(
                        relation, torch.empty((2, 0), dtype=torch.long)
                    ),
                )
                relation_test_neg = _sample_hgb_eval_negatives(
                    reference_pos_edge_index=relation_test_pos,
                    adjacency_edge_index=hgb_adjacency_edge_index,
                    known_positive_edge_index=known_positives_global,
                    relation_tuples=relation_tuples,
                    src_type=src_type,
                    dst_type=dst_type,
                    dst_range=dst_range,
                    seed=config.data.acm_split_seed + 202 + rel_id,
                )

            if relation_train_pos.size(1) == 0 or relation_test_pos.size(1) == 0:
                raise ValueError(
                    f"Target relation {relation} needs non-empty train and test "
                    "positive edges."
                )

            # Validation negatives are always plain uniform-random, matching HGB's
            # official get_valid_neg — 2-hop/fixed-file negatives are test-only.
            relation_val_neg = _sample_random_tail_negatives(
                reference_pos_edge_index=relation_val_pos,
                known_positive_edge_index=known_positives_global,
                dst_range=dst_range,
                seed=config.data.acm_split_seed + 101 + rel_id,
            )

            # Checked per relation (not pooled across target_relations) — Amazon's
            # rel_0/rel_1 share the same (product, product) node-type pair, so a
            # pooled node-pair check would false-positive on any pair connected by
            # both relations (e.g. co-viewed AND co-purchased), even though each
            # relation's own held-out edges are correctly excluded from its own
            # message-passing edge_index.
            relation_id_in_tuples = relation_tuples.index(relation)
            relation_held_out = torch.cat([relation_val_pos, relation_test_pos], dim=1)
            _assert_no_message_passing_leak(
                edge_index, edge_type, [relation_id_in_tuples], relation_held_out
            )
            # Also per-relation for the same reason: two different relations
            # (rel_0/rel_1) sharing a node-type pair can connect the same node
            # pair without that being train/val/test supervision leakage — each
            # relation gets its own independent split.
            _assert_no_leakage(relation_train_pos, relation_val_pos, relation_test_pos)

            train_pos_parts.append(relation_train_pos)
            train_pos_relation_parts.append(
                torch.full((relation_train_pos.size(1),), local_idx, dtype=torch.long)
            )
            val_pos_parts.append(relation_val_pos)
            val_pos_relation_parts.append(
                torch.full((relation_val_pos.size(1),), local_idx, dtype=torch.long)
            )
            val_neg_parts.append(relation_val_neg)
            val_neg_relation_parts.append(
                torch.full((relation_val_neg.size(1),), local_idx, dtype=torch.long)
            )
            test_pos_parts.append(relation_test_pos)
            test_pos_relation_parts.append(
                torch.full((relation_test_pos.size(1),), local_idx, dtype=torch.long)
            )
            test_neg_parts.append(relation_test_neg)
            test_neg_relation_parts.append(
                torch.full((relation_test_neg.size(1),), local_idx, dtype=torch.long)
            )

        train_pos_edge_index = torch.cat(train_pos_parts, dim=1)
        train_pos_edge_relation = torch.cat(train_pos_relation_parts, dim=0)
        val_pos_edge_index = torch.cat(val_pos_parts, dim=1)
        val_pos_edge_relation = torch.cat(val_pos_relation_parts, dim=0)
        val_neg_edge_index = torch.cat(val_neg_parts, dim=1)
        val_neg_edge_relation = torch.cat(val_neg_relation_parts, dim=0)
        test_pos_edge_index = torch.cat(test_pos_parts, dim=1)
        test_pos_edge_relation = torch.cat(test_pos_relation_parts, dim=0)
        test_neg_edge_index = torch.cat(test_neg_parts, dim=1)
        test_neg_edge_relation = torch.cat(test_neg_relation_parts, dim=0)
        num_target_relations = len(target_relation_ids)
        target_original_ids = torch.tensor(
            [relation_tuples.index(relation) for relation in target_relations], dtype=torch.long
        )
    elif dataset_name == "amazonbook":
        label_pos_edge_index = _offset_edge_index(
            edge_store.edge_label_index,
            src_offset,
            dst_offset,
        )
        all_pos_edge_index = _unique_edge_index(
            torch.cat([train_pos_edge_index, label_pos_edge_index], dim=1)
        )
        train_pos_edge_index, val_pos_edge_index, test_pos_edge_index = _split_edge_index_hgb(
            edge_index=all_pos_edge_index,
            val_ratio=config.data.link_val_ratio,
            test_ratio=config.data.link_test_ratio,
            seed=config.data.acm_split_seed,
        )
        train_neg_candidates = None
        known_positives_global = torch.cat(
            [train_pos_edge_index, val_pos_edge_index, test_pos_edge_index], dim=1
        )

        # Rebuild the target relation from train positives only so message passing
        # cannot see validation/test supervision.
        relation_id = relation_tuples.index((src_type, relation_name, dst_type))
        edge_index, edge_type = _replace_relation_edges(
            edge_index=edge_index,
            edge_type=edge_type,
            relation_id=relation_id,
            replacement_edges=train_pos_edge_index,
        )
        # Mirror the train-only target relation into reverse relations as well.
        reverse_candidates = [
            idx
            for idx, tup in enumerate(relation_tuples)
            if tup[0] == dst_type and tup[2] == src_type
        ]
        for reverse_relation_id in reverse_candidates:
            edge_index, edge_type = _replace_relation_edges(
                edge_index=edge_index,
                edge_type=edge_type,
                relation_id=reverse_relation_id,
                replacement_edges=train_pos_edge_index.flip(0),
            )

        hgb_adjacency_edge_index = torch.cat(
            [edge_index, val_pos_edge_index, test_pos_edge_index], dim=1
        )
        dst_range = (dst_offset, dst_offset + num_dst_nodes)
        # Validation negatives are always plain uniform-random, matching HGB's
        # official get_valid_neg — 2-hop negatives are reserved for test.
        val_neg_edge_index = _sample_random_tail_negatives(
            reference_pos_edge_index=val_pos_edge_index,
            known_positive_edge_index=known_positives_global,
            dst_range=dst_range,
            seed=config.data.acm_split_seed + 101,
        )
        test_neg_edge_index = _sample_hgb_eval_negatives(
            reference_pos_edge_index=test_pos_edge_index,
            adjacency_edge_index=hgb_adjacency_edge_index,
            known_positive_edge_index=known_positives_global,
            relation_tuples=relation_tuples,
            src_type=src_type,
            dst_type=dst_type,
            dst_range=dst_range,
            seed=config.data.acm_split_seed + 202,
        )

        held_out = torch.cat([val_pos_edge_index, test_pos_edge_index], dim=1)
        _assert_no_message_passing_leak(
            edge_index, edge_type, [relation_id], held_out
        )
        _assert_no_message_passing_leak(
            edge_index, edge_type, reverse_candidates, held_out.flip(0)
        )
        num_target_relations = 1
        target_original_ids = torch.tensor([relation_id], dtype=torch.long)
        train_pos_edge_relation = torch.zeros(train_pos_edge_index.size(1), dtype=torch.long)
        val_pos_edge_relation = torch.zeros(val_pos_edge_index.size(1), dtype=torch.long)
        val_neg_edge_relation = torch.zeros(val_neg_edge_index.size(1), dtype=torch.long)
        test_pos_edge_relation = torch.zeros(test_pos_edge_index.size(1), dtype=torch.long)
        test_neg_edge_relation = torch.zeros(test_neg_edge_index.size(1), dtype=torch.long)
    else:
        raise ValueError(f"Unsupported link-prediction dataset '{config.data.dataset_name}'.")

    if num_target_relations == 1:
        # Already checked per relation above for the multi-relation case (Amazon)
        # — a pooled, relation-agnostic check there would false-positive on any
        # node pair connected by more than one pooled relation.
        _assert_no_leakage(train_pos_edge_index, val_pos_edge_index, test_pos_edge_index)

    # Fact layer (D7): rho from the graph BEFORE the split (decision 2), applied to the
    # message graph, which holds training positives only. Negatives, splits and the
    # per-relation checks above are computed on the stored (untied) graph, so they are the
    # same under both fact layers.
    fact_layer = build_fact_layer(pre_split_edge_index, pre_split_edge_type, relation_names,
                                  raw_x.size(0), config.data.fact_layer)
    edge_index, edge_type = apply_fact_layer(fact_layer, edge_index, edge_type)
    relation_names = fact_layer.kept_names
    relation_tuples = tuple(relation_tuples[r] for r in fact_layer.kept)

    # Path cells on the message graph `edge_index`, which holds training positives only.
    face_groups = build_link_path_groups(config, fact_layer, edge_index, edge_type, node_type,
                                         seed=config.seed)
    held_out_edge_index = torch.cat([val_pos_edge_index, test_pos_edge_index], dim=1)
    held_out_relation = target_original_ids[
        torch.cat([val_pos_edge_relation, test_pos_edge_relation])
    ]
    assert_no_held_out_facts(fact_layer, edge_index, edge_type, face_groups,
                             held_out_edge_index, held_out_relation)

    return PreparedLinkPredData(
        x=raw_x,
        node_type=node_type,
        node_type_names=node_type_names,
        node_slices=node_slices,
        feature_dims=feature_dims,
        edge_index=edge_index,
        edge_type=edge_type,
        relation_names=relation_names,
        relation_tuples=relation_tuples,
        target_relation=config.data.target_relation,
        train_pos_edge_index=train_pos_edge_index,
        train_neg_candidates=train_neg_candidates,
        val_pos_edge_index=val_pos_edge_index,
        val_neg_edge_index=val_neg_edge_index,
        test_pos_edge_index=test_pos_edge_index,
        test_neg_edge_index=test_neg_edge_index,
        known_pos_edge_index=known_positives_global,
        num_target_relations=num_target_relations,
        train_pos_edge_relation=train_pos_edge_relation,
        val_pos_edge_relation=val_pos_edge_relation,
        val_neg_edge_relation=val_neg_edge_relation,
        test_pos_edge_relation=test_pos_edge_relation,
        test_neg_edge_relation=test_neg_edge_relation,
        neg_sampling_mode=config.data.neg_sampling_mode,
        neg_sampling_ratio=config.data.neg_sampling_ratio,
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_dst_nodes,
        src_node_offset=src_offset,
        dst_node_offset=dst_offset,
        face_groups=face_groups,
        fact_layer=fact_layer,
    )
