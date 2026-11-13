"""H2G-former's node-encoder stack as H2GB's (spec v1.19 T2; decisions D-P3, 2026-11-13; card Part P
P1b).

* **H2GB datasets** take their config's stack exactly (`configs/<ds>/<ds>-H2Gformer.yaml`,
  `dataset.node_encoder_name`; `docs/predefined_metapaths.md` §3): raw + label + MetaPath2Vec on
  ogbn-mag, mag-year, oag-chem, PDNS and H-Pokec; raw + label on IEEE-CIS-G, oag-cs and oag-eng.
  No search key.
* **Other cells:** HGB NC raw + label, LP raw, GC raw; plus the MetaPath2Vec channel when the
  tuned key `metapath_pe` is on. The key exists only where a walk schema exists: HGB NC (ACM,
  DBLP, IMDB, Freebase), the derived LP tasks and LastFM; PubMed (no HGB table) and GC have none.
* **Walk schema:** H2GB's MetaPath2Vec takes one cyclic schema per dataset; it is built as H2GB
  builds its schemas, the dataset's listed meta-paths (HGB's published table, paths starting at the
  target type; LP: the source graph's table at the head type; LastFM: UU, UAU, UATAU) concatenated
  in table order. A hop against a stored relation uses PyG's `rev_<relation>` edge type, as
  `T.ToUndirected` names it (H2GB's own schemas use `rev_*` types, `encoding_generator.py:263-274`).
* **MetaPath2Vec:** H2GB's own class (`H2GB/loader/embedding/metapath2vec.py`) and training loop
  settings of its heterogeneous academic graphs (`encoding_generator.py:256-270`, ogbn-mag:
  walk length 64, context 7, 30 walks per node, 8 negatives, batch 128, SparseAdam lr 0.01;
  5 epochs, `:520`; `dim_pe` 128 and 1 layer, `configs/ogbn-mag/ogbn-mag-H2Gformer.yaml:21-24`).
* **Label channel:** H2GB's `Hetero_Label` encoder (`hetero_label_encoder.py`). It reads the
  target type's `y` under `train_mask` (and also `val_mask` when `batch.split == 'test'`, l.42-44).
  The campaign rule is training labels only (decision 2026-11-13, route test): the adapter hands it
  a `y` whose non-training entries are a fixed placeholder, `val_mask` all False, and never the
  split 'test'.
"""

from __future__ import annotations

from typing import Any

import torch

from src.baselines.protocol import GRAPH_CLASSIFICATION, LINK_PREDICTION, NODE_CLASSIFICATION
from src.configs.config import LITERATURE_METAPATHS

RAW, LABEL, METAPATH = "Hetero_Raw", "Hetero_Label", "Hetero_Metapath"

#: H2GB datasets: their configs' `node_encoder_name` (docs/predefined_metapaths.md §3).
H2GB_STACKS: dict[str, tuple[str, ...]] = {
    "ogbn-mag": (RAW, LABEL, METAPATH), "mag-year": (RAW, LABEL, METAPATH),
    "oag-chem": (RAW, LABEL, METAPATH), "pdns": (RAW, LABEL, METAPATH),
    "h-pokec": (RAW, LABEL, METAPATH),
    "ieee-cis-g": (RAW, LABEL), "oag-cs": (RAW, LABEL), "oag-eng": (RAW, LABEL),
}

#: H2GB's MetaPath2Vec settings (ogbn-mag branch, encoding_generator.py:256-270, 520; config).
METAPATH2VEC_SETTINGS: dict[str, Any] = {
    "walk_length": 64, "context_size": 7, "walks_per_node": 30, "num_negative_samples": 8,
    "batch_size": 128, "lr": 0.01, "epochs": 5, "dim_pe": 128, "layers": 1,
}

#: The HGB release's relations in link-id order (= PyG HGBDataset's edge-type order and the
#: `rel_<id>` names of the LP-format derived tasks). VERIFIED-RUN (Part O, `flatten_edges`).
HGB_RELATIONS: dict[str, tuple[str, ...]] = {
    "acm": ("paper__cite__paper", "paper__ref__paper", "paper__to__author", "author__to__paper",
            "paper__to__subject", "subject__to__paper", "paper__to__term", "term__to__paper"),
    "dblp": ("author__to__paper", "paper__to__term", "paper__to__venue", "paper__to__author",
             "term__to__paper", "venue__to__paper"),
    "imdb": ("movie__to__director", "director__to__movie", "movie__>actorh__actor",
             "actor__to__movie", "movie__to__keyword", "keyword__to__movie"),
    # data/NC/freebase/raw/Freebase/info.dat LINK 0-35, PyG's names (type__word__type)
    "freebase": tuple(f"{a}__{w}__{b}" for a, w, b in (
        ("book", "and", "book"), ("book", "to", "film"), ("book", "on", "sports"),
        ("book", "on", "location"), ("book", "about", "organization"), ("film", "and", "film"),
        ("music", "in", "book"), ("music", "in", "film"), ("music", "and", "music"),
        ("music", "for", "sports"), ("music", "on", "location"), ("sports", "in", "film"),
        ("sports", "and", "sports"), ("sports", "on", "location"), ("people", "to", "book"),
        ("people", "to", "film"), ("people", "to", "music"), ("people", "to", "sports"),
        ("people", "and", "people"), ("people", "on", "location"),
        ("people", "in", "organization"), ("people", "in", "business"),
        ("location", "in", "film"), ("location", "and", "location"),
        ("organization", "in", "film"), ("organization", "to", "music"),
        ("organization", "to", "sports"), ("organization", "on", "location"),
        ("organization", "and", "organization"), ("organization", "for", "business"),
        ("business", "about", "book"), ("business", "about", "film"),
        ("business", "about", "music"), ("business", "about", "sports"),
        ("business", "on", "location"), ("business", "and", "business"))),
}

#: Derived LP task -> (source NC release, head type = the start type of its listed paths).
DERIVED_SOURCE: dict[str, str] = {"acm_pa": "acm", "dblp_ap": "dblp", "imdb_ma": "imdb",
                                  "freebase_bb": "freebase"}

EdgeType = tuple[str, str, str]

#: LastFM (LP): UU, UAU, UATAU over rel_0 user-artist, rel_1 user-user, rel_2 artist-tag
#: (data/LP/LastFM/info.dat; spec v1.19 T2, D-P3).
LASTFM_PATHS: tuple[tuple[str, tuple[EdgeType, ...]], ...] = (
    ("UU", (("user", "rel_1", "user"),)),
    ("UAU", (("user", "rel_0", "artist"), ("artist", "rev_rel_0", "user"))),
    ("UATAU", (("user", "rel_0", "artist"), ("artist", "rel_2", "tag"),
               ("tag", "rev_rel_2", "artist"), ("artist", "rev_rel_0", "user"))),
)


def _hop(name: str, sign: str, rename: dict[str, str] | None = None) -> EdgeType:
    source, relation, destination = name.split("__")
    if rename is not None:
        relation = rename[name]
    if sign == "+":
        return (source, relation, destination)
    return (destination, f"rev_{relation}", source)


def listed_paths(dataset: str) -> tuple[tuple[str, tuple[EdgeType, ...]], ...]:
    """(name, edge types) of every listed meta-path of the cell's dataset, in table order, as edge
    types of the graph H2G-former sees (NC: PyG HGBDataset names; LP: the `rel_<id>` names)."""
    key = dataset.lower()
    if key == "lastfm":
        return LASTFM_PATHS
    if key in DERIVED_SOURCE:
        source = DERIVED_SOURCE[key]
        names = HGB_RELATIONS[source]
        rename = {name: f"rel_{index}" for index, name in enumerate(names)}
        return tuple((t.name, tuple(_hop(n, s, rename) for n, s in t.hops))
                     for t in LITERATURE_METAPATHS[source])
    if key in LITERATURE_METAPATHS:
        return tuple((t.name, tuple(_hop(n, s) for n, s in t.hops))
                     for t in LITERATURE_METAPATHS[key])
    raise KeyError(f"no listed meta-paths for {dataset!r}")


def walk_schema(dataset: str) -> tuple[EdgeType, ...]:
    """The cyclic MetaPath2Vec schema: the listed paths concatenated in table order."""
    schema = tuple(hop for _, hops in listed_paths(dataset) for hop in hops)
    if schema[0][0] != schema[-1][2]:
        raise ValueError(f"{dataset}: the concatenated schema is not cyclic.")
    for a, b in zip(schema, schema[1:]):
        if a[2] != b[0]:
            raise ValueError(f"{dataset}: hops {a} and {b} do not meet.")
    return schema


def has_walk_schema(task: str, dataset: str) -> bool:
    key = dataset.lower()
    if task == NODE_CLASSIFICATION:
        return key in LITERATURE_METAPATHS
    if task == LINK_PREDICTION:
        return key == "lastfm" or key in DERIVED_SOURCE
    return False


def encoder_stack(task: str, dataset: str, metapath_pe: bool | None = None) -> tuple[str, ...]:
    """The node-encoder stack of one cell (spec v1.19 T2)."""
    key = dataset.lower()
    if task == NODE_CLASSIFICATION and key in H2GB_STACKS:
        if metapath_pe is not None:
            raise ValueError(f"{dataset} takes H2GB's config stack; metapath_pe is not a key.")
        return H2GB_STACKS[key]
    base = {NODE_CLASSIFICATION: (RAW, LABEL), LINK_PREDICTION: (RAW,),
            GRAPH_CLASSIFICATION: (RAW,)}[task]
    if not has_walk_schema(task, key):
        if metapath_pe:
            raise ValueError(f"{task} {dataset} has no walk schema; metapath_pe cannot be on.")
        return base
    return base + ((METAPATH,) if metapath_pe else ())


def with_reverse_edges(edge_index_dict: dict[EdgeType, torch.Tensor]
                       ) -> dict[EdgeType, torch.Tensor]:
    """The walk graph of MetaPath2Vec: every relation and its `rev_` copy (as T.ToUndirected)."""
    out = dict(edge_index_dict)
    for (source, relation, destination), index in edge_index_dict.items():
        out.setdefault((destination, f"rev_{relation}", source), index.flip(0))
    return out


def metapath_dim_pe(hidden_dim: int) -> int:
    """Width of the MetaPath2Vec channel at model width `hidden_dim` (decision D-P6, author
    2026-11-13): `min(128, hidden_dim // 2)`, so H2GB's composed encoder always keeps a raw channel
    of `hidden_dim - dim_pe > 0` (composed_encoders.py:54-58). Reaches H2GB's configs: PDNS 64/32,
    Pokec and oag-chem 256/128, ogbn-mag and mag-year 512/128."""
    if hidden_dim < 2:
        raise ValueError(f"hidden_dim must be >= 2, got {hidden_dim}.")
    return min(METAPATH2VEC_SETTINGS["dim_pe"], hidden_dim // 2)


def metapath2vec_embeddings(edge_index_dict: dict[EdgeType, torch.Tensor],
                            num_nodes_dict: dict[str, int], schema: tuple[EdgeType, ...],
                            device: torch.device, seed: int, epochs: int | None = None,
                            dim_pe: int | None = None) -> dict[str, torch.Tensor]:
    """H2GB's MetaPath2Vec trained with its settings on the walk graph; `{node type: [N, dim_pe]}`
    for every node type the schema visits (H2GB's `pestat_Hetero_Metapath`, master_loader.py:266-281).
    `dim_pe` is the channel width (`metapath_dim_pe`; H2GB's 128 when None); `epochs` overrides
    H2GB's 5 only in tests."""
    from src.baselines.runtime import pushd_sys_path

    with pushd_sys_path("h2gb"):
        from H2GB.loader.embedding.metapath2vec import MetaPath2Vec  # type: ignore[import-not-found]

    s = METAPATH2VEC_SETTINGS
    walk_graph = with_reverse_edges(edge_index_dict)
    missing = [hop for hop in schema if hop not in walk_graph]
    if missing:
        raise KeyError(f"schema hops {missing} are not edge types of the graph.")
    torch.manual_seed(seed)
    model = MetaPath2Vec({k: v.cpu() for k, v in walk_graph.items()},
                         embedding_dim=s["dim_pe"] if dim_pe is None else int(dim_pe),
                         metapath=list(schema),
                         walk_length=s["walk_length"], context_size=s["context_size"],
                         walks_per_node=s["walks_per_node"],
                         num_negative_samples=s["num_negative_samples"],
                         num_nodes_dict=num_nodes_dict, sparse=True).to(device)
    loader = model.loader(batch_size=s["batch_size"], shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=s["lr"])
    model.train()
    for _ in range(s["epochs"] if epochs is None else epochs):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
    weight = model.embedding.weight.detach()
    return {node_type: weight[model.start[node_type]:model.end[node_type]].clone()
            for node_type in model.start}


def training_label_input(y: torch.Tensor, train_mask: torch.Tensor, placeholder: int = 0
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(`y`, `train_mask`, `val_mask`) for H2GB's label encoder with training labels only: every
    non-training entry of `y` is `placeholder` (the encoder masks it anyway), `val_mask` is all
    False (so even a 'test' split could not add validation labels)."""
    labels = y.clone()
    labels[~train_mask] = placeholder
    return labels, train_mask.clone(), torch.zeros_like(train_mask)


def configure_encoders(cfg: Any, stack: tuple[str, ...]) -> None:
    """Set graphgym's encoder config for `stack` (H2GB's own config keys). With MetaPath2Vec,
    H2GB's composed encoder gives the raw channel `dim_hidden - dim_pe` dimensions
    (composed_encoders.py:54-58); `dim_pe = metapath_dim_pe(dim_hidden)` keeps it positive for
    every width (D-P6)."""
    cfg.dataset.node_encoder_name = "+".join(stack)
    cfg.posenc_Hetero_Label.enable = LABEL in stack
    cfg.posenc_Hetero_Label.dim_pe = 0
    cfg.posenc_Hetero_Metapath.enable = METAPATH in stack
    cfg.posenc_Hetero_Metapath.dim_pe = metapath_dim_pe(int(cfg.gt.dim_hidden))
    cfg.posenc_Hetero_Metapath.layers = METAPATH2VEC_SETTINGS["layers"]


__all__ = ["DERIVED_SOURCE", "H2GB_STACKS", "HGB_RELATIONS", "LABEL", "LASTFM_PATHS",
           "METAPATH", "METAPATH2VEC_SETTINGS", "RAW", "configure_encoders", "encoder_stack",
           "metapath_dim_pe",
           "has_walk_schema", "listed_paths", "metapath2vec_embeddings",
           "training_label_input", "walk_schema", "with_reverse_edges"]
