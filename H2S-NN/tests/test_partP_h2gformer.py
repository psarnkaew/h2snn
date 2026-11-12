"""Part P, P1b (`docs/H2S-NN_partP_card.md`; spec v1.19 T2, D-P3): H2G-former's encoder as H2GB's.

- the stack resolved per (task, dataset, metapath_pe) equals T2;
- each concatenated walk schema is cyclic, every hop is an edge type of the dataset's graph (incl.
  `rev_` copies), and it equals the listed meta-paths concatenated in table order (hand lists);
- the MetaPath2Vec settings equal H2GB's source and config;
- information route: poisoning validation/test labels leaves the model's output unchanged (the
  label channel sees training labels only); a stray 'test' split cannot add validation labels;
- MetaPath2Vec embeddings are produced on a tiny graph and consumed by the model.
Card mutations: all labels to the label channel -> the route test fails; one listed meta-path
dropped from a schema -> the schema test fails.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.baselines.adapters._h2gformer_encoders import (
    H2GB_STACKS,
    HGB_RELATIONS,
    LABEL,
    METAPATH,
    METAPATH2VEC_SETTINGS,
    RAW,
    encoder_stack,
    walk_schema,
    with_reverse_edges,
)
from src.baselines.protocol import GRAPH_CLASSIFICATION, LINK_PREDICTION, NODE_CLASSIFICATION

ROOT = Path(__file__).resolve().parent.parent
H2GB_ROOT = ROOT / "vendor" / "H2GB"


class StackTests(unittest.TestCase):
    def test_stack_per_cell_is_t2(self) -> None:
        nc, lp, gc = NODE_CLASSIFICATION, LINK_PREDICTION, GRAPH_CLASSIFICATION
        for dataset in ("acm", "dblp", "imdb", "freebase"):
            self.assertEqual(encoder_stack(nc, dataset, False), (RAW, LABEL))
            self.assertEqual(encoder_stack(nc, dataset, True), (RAW, LABEL, METAPATH))
        for dataset in ("acm_pa", "dblp_ap", "imdb_ma", "freebase_bb", "lastfm"):
            self.assertEqual(encoder_stack(lp, dataset, False), (RAW,))
            self.assertEqual(encoder_stack(lp, dataset, True), (RAW, METAPATH))
        self.assertEqual(encoder_stack(lp, "pubmed"), (RAW,))  # no HGB table: raw only
        with self.assertRaises(ValueError):
            encoder_stack(lp, "pubmed", True)
        for dataset in ("mutag", "proteins", "enzymes", "nci1"):
            self.assertEqual(encoder_stack(gc, dataset), (RAW,))
        full = {"ogbn-mag", "mag-year", "oag-chem", "pdns", "h-pokec"}
        for dataset in H2GB_STACKS:  # H2GB's configs, no key
            expected = (RAW, LABEL, METAPATH) if dataset in full else (RAW, LABEL)
            self.assertEqual(encoder_stack(nc, dataset), expected)
            with self.assertRaises(ValueError):
                encoder_stack(nc, dataset, True)

    def test_h2gb_stacks_equal_their_configs(self) -> None:
        names = {"ogbn-mag": "ogbn-mag", "mag-year": "mag-year", "oag-chem": "oag-chem",
                 "oag-cs": "oag-cs", "oag-eng": "oag-eng", "ieee-cis-g": "IEEE-CIS",
                 "h-pokec": "Pokec", "pdns": "PDNS"}
        for dataset, folder in names.items():
            configs = list((H2GB_ROOT / "configs").glob(f"{folder}*/*H2Gformer*.yaml"))
            if not configs:
                configs = list((H2GB_ROOT / "configs").glob(f"*{folder}*/*H2Gformer*.yaml"))
            with self.subTest(dataset=dataset):
                self.assertTrue(configs, f"no H2G-former config for {dataset}")
                text = configs[0].read_text(encoding="utf-8")
                found = re.search(r"node_encoder_name:\s*(\S+)", text).group(1)
                self.assertEqual(tuple(found.split("+")), H2GB_STACKS[dataset])


# Hand-written: HGB's table in order (docs/predefined_metapaths.md §2), PyG edge names.
P, A, S = "paper", "author", "subject"
HAND = {
    "acm": [(P, "to", A), (A, "to", P),                                     # PAP
            (P, "to", S), (S, "to", P),                                     # PSP
            (P, "cite", P), (P, "to", A), (A, "to", P),                     # PcPAP
            (P, "cite", P), (P, "to", S), (S, "to", P),                     # PcPSP
            (P, "ref", P), (P, "to", A), (A, "to", P),                      # PrPAP
            (P, "ref", P), (P, "to", S), (S, "to", P)],                     # PrPSP
    "imdb": [("movie", "to", "director"), ("director", "to", "movie"),
             ("movie", ">actorh", "actor"), ("actor", "to", "movie")],
    "lastfm": [("user", "rel_1", "user"),
               ("user", "rel_0", "artist"), ("artist", "rev_rel_0", "user"),
               ("user", "rel_0", "artist"), ("artist", "rel_2", "tag"),
               ("tag", "rev_rel_2", "artist"), ("artist", "rev_rel_0", "user")],
    "acm_pa": [(P, "rel_2", A), (A, "rel_3", P), (P, "rel_4", S), (S, "rel_5", P),
               (P, "rel_0", P), (P, "rel_2", A), (A, "rel_3", P),
               (P, "rel_0", P), (P, "rel_4", S), (S, "rel_5", P),
               (P, "rel_1", P), (P, "rel_2", A), (A, "rel_3", P),
               (P, "rel_1", P), (P, "rel_4", S), (S, "rel_5", P)],
}


def _edge_types(dataset: str) -> set[tuple[str, str, str]]:
    key = {"acm_pa": "acm", "dblp_ap": "dblp", "imdb_ma": "imdb", "freebase_bb": "freebase"}
    if dataset == "lastfm":
        stored = {("user", "rel_0", "artist"), ("user", "rel_1", "user"),
                  ("artist", "rel_2", "tag")}
    elif dataset in key:
        stored = {(n.split("__")[0], f"rel_{i}", n.split("__")[2])
                  for i, n in enumerate(HGB_RELATIONS[key[dataset]])}
    else:
        stored = {tuple(n.split("__")) for n in HGB_RELATIONS[dataset]}
    return set(with_reverse_edges({t: torch.zeros(2, 0, dtype=torch.long) for t in stored}))


class SchemaTests(unittest.TestCase):
    DATASETS = ("acm", "dblp", "imdb", "freebase", "acm_pa", "dblp_ap", "imdb_ma",
                "freebase_bb", "lastfm")

    def test_schemas_are_cyclic_edge_types_of_the_graph(self) -> None:
        for dataset in self.DATASETS:
            with self.subTest(dataset=dataset):
                schema = walk_schema(dataset)
                self.assertEqual(schema[0][0], schema[-1][2])
                self.assertTrue(all(a[2] == b[0] for a, b in zip(schema, schema[1:])))
                self.assertLessEqual(set(schema), _edge_types(dataset))

    def test_schemas_equal_the_table_concatenation(self) -> None:
        for dataset, expected in HAND.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(list(walk_schema(dataset)), expected)
        self.assertEqual(len(walk_schema("dblp")), 2 + 4 + 4)          # APA, APTPA, APVPA
        self.assertEqual(len(walk_schema("freebase")), 2 + 2 + 3 + 2 + 3 + 3 + 2)

    def test_hops_exist_on_the_real_graphs(self) -> None:
        from src.data.data_layout import REPO_DATA_ROOT, nc_root

        names = {"acm": "ACM", "dblp": "DBLP", "imdb": "IMDB", "freebase": "Freebase"}
        present = [d for d in names if (nc_root(REPO_DATA_ROOT) / d).exists()]
        if not present:
            self.skipTest("HGB node-classification data not present")
        from torch_geometric.datasets import HGBDataset

        for dataset in present:  # data read only
            with self.subTest(dataset=dataset):
                data = HGBDataset(root=str(nc_root(REPO_DATA_ROOT)), name=names[dataset])[0]
                graph = with_reverse_edges(dict(data.edge_index_dict))
                self.assertLessEqual(set(walk_schema(dataset)), set(graph))


class SettingsTests(unittest.TestCase):
    def test_metapath2vec_settings_equal_h2gbs(self) -> None:
        source = (H2GB_ROOT / "H2GB" / "loader" / "encoding_generator.py").read_text("utf-8")
        branch = source.split("elif dataset.name == 'ogbn-mag':", 2)[2].split("elif", 1)[0]
        for key in ("walk_length", "context_size", "walks_per_node", "num_negative_samples",
                    "batch_size", "lr"):
            with self.subTest(key=key):
                value = re.search(rf"{key} = ([0-9.e-]+)", branch).group(1)
                self.assertEqual(float(value), float(METAPATH2VEC_SETTINGS[key]))
        self.assertIn("for epoch in range(1, 6):", source)  # 5 epochs
        self.assertEqual(METAPATH2VEC_SETTINGS["epochs"], 5)
        config = (H2GB_ROOT / "configs" / "ogbn-mag" / "ogbn-mag-H2Gformer.yaml").read_text("utf-8")
        block = config.split("posenc_Hetero_Metapath:", 1)[1]
        self.assertEqual(int(re.search(r"dim_pe:\s*(\d+)", block).group(1)),
                         METAPATH2VEC_SETTINGS["dim_pe"])
        self.assertEqual(int(re.search(r"layers:\s*(\d+)", block).group(1)),
                         METAPATH2VEC_SETTINGS["layers"])


def _tiny(seed: int = 0):
    from src.baselines.protocol import BaselineParams
    from tests.partO_fixtures import acm_like_hetero

    hetero = acm_like_hetero(num_papers=30, num_authors=12, num_subjects=3, num_terms=6,
                             seed=seed)
    hetero["paper"].batch_size = int(hetero["paper"].num_nodes)
    y = hetero["paper"].y.clone()
    train = hetero["paper"].train_mask.clone()
    bundle = SimpleNamespace(
        dataset="acm", num_classes=int(y.max()) + 1, is_multilabel=False,
        data=SimpleNamespace(target_y=y, target_train_mask=train))
    params = BaselineParams({"hidden_dim": 160, "gt_layers": 1, "attn_heads": 2})
    return hetero, bundle, params


def _output(stack, bundle_y=None, split="val", seed=0, metapath_epochs=1, poison_pestat=False):
    from src.baselines.adapters.h2gformer import build_node_classification_model

    hetero, bundle, params = _tiny()
    if bundle_y is not None:
        bundle.data.target_y = bundle_y(bundle.data.target_y, bundle.data.target_train_mask)
    torch.manual_seed(seed)
    model = build_node_classification_model(hetero, "paper", bundle, params, stack,
                                            device=torch.device("cpu"), seed=seed,
                                            metapath_epochs=metapath_epochs)
    if poison_pestat:
        for node_type in hetero.node_types:
            if "pestat_Hetero_Metapath" in hetero[node_type]:
                hetero[node_type]["pestat_Hetero_Metapath"] = torch.randn_like(
                    hetero[node_type]["pestat_Hetero_Metapath"])
    model.eval()
    hetero.split = split
    with torch.no_grad():
        out = model(hetero)
    return out[0] if isinstance(out, tuple) else out


def _poison(y: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    poisoned = y.clone()
    poisoned[~train] = (y[~train] + 1) % (int(y.max()) + 1)
    return poisoned


class RouteTests(unittest.TestCase):
    def test_validation_and_test_labels_do_not_reach_the_label_channel(self) -> None:
        clean = _output((RAW, LABEL))
        poisoned = _output((RAW, LABEL), bundle_y=_poison)
        self.assertTrue(torch.equal(clean, poisoned))
        # even H2GB's 'test' branch (train | val labels) sees training labels only
        self.assertTrue(torch.equal(_output((RAW, LABEL), split="test"),
                                    _output((RAW, LABEL), bundle_y=_poison, split="test")))

    def test_the_label_channel_reads_training_labels(self) -> None:
        def flip_train(y, train):
            flipped = y.clone()
            flipped[train] = (y[train] + 1) % (int(y.max()) + 1)
            return flipped

        self.assertFalse(torch.equal(_output((RAW, LABEL)), _output((RAW, LABEL),
                                                                    bundle_y=flip_train)))


class MetapathChannelTests(unittest.TestCase):
    def test_narrow_hidden_width_is_refused_with_metapath(self) -> None:
        from src.baselines.adapters.h2gformer import build_node_classification_model
        from src.baselines.protocol import BaselineParams

        hetero, bundle, _ = _tiny()
        with self.assertRaisesRegex(ValueError, "hidden_dim > dim_pe"):
            build_node_classification_model(hetero, "paper", bundle, BaselineParams(
                {"hidden_dim": 128}), (RAW, LABEL, METAPATH), device=torch.device("cpu"),
                seed=0, metapath_epochs=1)

    def test_embeddings_are_produced_and_consumed(self) -> None:
        from src.baselines.adapters.h2gformer import attach_metapath_embeddings

        hetero, _, _ = _tiny()
        attach_metapath_embeddings(hetero, "acm", device=torch.device("cpu"), seed=0, epochs=1)
        self.assertTrue(bool((hetero["term"]["pestat_Hetero_Metapath"] == 0).all()))  # unvisited
        for node_type in ("paper", "author", "subject"):
            value = hetero[node_type]["pestat_Hetero_Metapath"]
            self.assertEqual(tuple(value.shape), (int(hetero[node_type].num_nodes),
                                                  METAPATH2VEC_SETTINGS["dim_pe"]))
            self.assertTrue(bool(torch.isfinite(value).all()))
        out = _output((RAW, LABEL, METAPATH))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertFalse(torch.equal(out, _output((RAW, LABEL, METAPATH), poison_pestat=True)))


if __name__ == "__main__":
    unittest.main()
