"""Part O, O3 (`docs/H2S-NN_partO_card.md`): the fixed-literature-schema row (spec v1.19 §6).

- the builder changes only `data.path_source` (diff test, also after the saved-config round trip);
  it refuses anything but a resolved v3-B anchored `full` WISP NC configuration, and any other
  difference; the row is still refused in tuning;
- walks per anchor: the tuned r split evenly over the listed paths (table order, at least one each),
  summing to r, incl. r not divisible by the number of paths; the sampler asks for exactly that;
- every drawn cell is an instance of a listed meta-path (both orientations), every listed path is
  drawn on the fixture, cells start at the target type;
- the evaluation draw is deterministic per base seed;
- the anchored forward is finite on an ACM-sized synthetic graph;
- (real HGB files, when present: data read only) the lists resolve on ACM, DBLP, IMDB and Freebase
  and every listed path has instances from the target nodes (the card's stop condition).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    import yaml
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch and yaml are required for these tests: {exc}") from exc

from src.configs.config import (
    LITERATURE_METAPATHS,
    TrainConfig,
    fixed_literature_schema_config,
    is_literature_row,
    literature_walk_split,
    reject_never_tuned_settings,
)
from src.data.data_layout import nc_root
from tests.partO_fixtures import ACM_NODE_COUNTS, acm_like_hetero, patched_acm_loader

REPO_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def _full(dataset: str = "acm", r: int = 16, l_max: int = 3, **top) -> TrainConfig:
    """A tuned v3-B `full` configuration (as the seed-0 sweep leaves it), small sizes, CPU."""
    return TrainConfig.from_dict({
        "device": "cpu", "seed": 3, **top,
        "data": {"dataset_name": dataset, "wisp_max_length": l_max, "wisp_walks_per_anchor": r},
        "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                  "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full"},
        "ablation": {"coupling": "symmetric"}})


def _flat(config: TrainConfig) -> dict:
    from src.configs.config import _flatten
    from src.utils.misc import dataclass_to_dict

    return _flatten(dataclass_to_dict(config))


class BuilderTests(unittest.TestCase):
    def test_builder_changes_only_the_path_source(self) -> None:
        from src.utils.misc import save_config

        for dataset in ("acm", "dblp", "imdb"):
            for l_max, r in ((2, 8), (5, 64), (4, 33)):
                with self.subTest(dataset=dataset, l_max=l_max, r=r):
                    full = _full(dataset, r=r, l_max=l_max)
                    row = fixed_literature_schema_config(full)
                    before, after = _flat(full), _flat(row)
                    self.assertEqual(set(before), set(after))
                    changed = {k for k in before if before[k] != after[k]}
                    # spec v1.20 U1 (D-O6): the path source and the row's own run name
                    self.assertEqual(changed, {"data.path_source", "exp_name"})
                    self.assertEqual(after["exp_name"], before["exp_name"] + "__literature_row")
                    self.assertEqual((before["data.path_source"], after["data.path_source"]),
                                     ("wisp", "schemas"))
                    self.assertTrue(is_literature_row(row.data))
                    self.assertFalse(is_literature_row(full.data))
                    self.assertEqual(row.data.metapath_templates, ())  # the list is the table
                    with tempfile.TemporaryDirectory() as tmp:  # what a server run reloads
                        save_config(row, Path(tmp) / "config.yaml")
                        loaded = TrainConfig.from_dict(yaml.safe_load(
                            (Path(tmp) / "config.yaml").read_text(encoding="utf-8")))
                    self.assertEqual(_flat(loaded), after)

    def test_builder_refuses_other_inputs(self) -> None:
        cases = {
            "arm": TrainConfig.from_dict({
                "data": {"dataset_name": "acm", "wisp_walks_per_anchor": 16},
                "model": {"propagation": "anchored", "arm": "covering_only"},
                "ablation": {"coupling": "symmetric"}}),
            "legacy propagation": TrainConfig.from_dict({"data": {"dataset_name": "acm"}}),
            "already a row": fixed_literature_schema_config(_full()),
            "graph classification": TrainConfig.from_dict({
                "data": {"dataset_name": "mutag", "wisp_walks_per_anchor": 8},
                "model": {"propagation": "anchored", "arm": "full"},
                "ablation": {"coupling": "symmetric"}}),
        }
        for name, config in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                fixed_literature_schema_config(config)

    def test_any_other_difference_raises(self) -> None:
        import src.configs.config as config_module

        real = config_module.TrainConfig.from_dict

        def drifting(raw):  # a resolution that would also change another setting
            config = real(raw)
            config.data.acm_split_seed += 1
            return config

        with mock.patch.object(config_module.TrainConfig, "from_dict", staticmethod(drifting)):
            with self.assertRaisesRegex(ValueError, "data.acm_split_seed"):
                fixed_literature_schema_config(_full())

    def test_row_configuration_rules(self) -> None:
        raw = {"data": {"dataset_name": "acm", "path_source": "schemas",
                        "wisp_walks_per_anchor": 16, "wisp_max_length": 3}}
        self.assertTrue(is_literature_row(TrainConfig.from_dict(raw).data))
        bad = {
            "explicit list": {"metapath_templates": [{"name": "PAP", "node_types":
                                                      ["paper", "author", "paper"]}]},
            "r below the number of paths": {"wisp_walks_per_anchor": 5},  # ACM lists 6
            "adaptive proposal": {"wisp_proposal": "adaptive"},
            "untied": {"fact_layer": "untied"},
        }
        for name, change in bad.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                TrainConfig.from_dict({"data": {**raw["data"], **change}})
        with self.assertRaises(ValueError):  # node classification only
            TrainConfig.from_dict({"data": {"dataset_name": "mutag", "path_source": "schemas",
                                            "wisp_walks_per_anchor": 8}})
        # the legacy schema row (r unset) keeps the old presets and its rule on l_max
        legacy = TrainConfig.from_dict({"data": {"dataset_name": "dblp",
                                                 "path_source": "schemas"}}).data
        self.assertFalse(is_literature_row(legacy))
        self.assertEqual([t.name for t in legacy.metapath_templates], ["APA", "APTPA"])
        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"data": {"dataset_name": "acm", "path_source": "schemas",
                                            "wisp_max_length": 3}})

    def test_row_is_refused_in_tuning(self) -> None:
        from src.utils.misc import dataclass_to_dict

        row = dataclass_to_dict(fixed_literature_schema_config(_full()))
        with self.assertRaisesRegex(ValueError, "fixed-literature-schema"):
            reject_never_tuned_settings(row)
        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"data": {"dataset_name": "acm"}},
                                        ["data.path_source"])


class SplitTests(unittest.TestCase):
    def test_split_sums_to_r_and_is_even(self) -> None:
        for k in range(1, 8):
            for r in range(k, 70):
                with self.subTest(r=r, k=k):
                    split = literature_walk_split(r, k)
                    self.assertEqual((len(split), sum(split)), (k, r))
                    self.assertGreaterEqual(min(split), 1)
                    self.assertLessEqual(max(split) - min(split), 1)
                    self.assertEqual(list(split), sorted(split, reverse=True))  # table order
        self.assertEqual(literature_walk_split(16, 6), (3, 3, 3, 3, 2, 2))
        self.assertEqual(literature_walk_split(64, 7), (10, 9, 9, 9, 9, 9, 9))
        self.assertEqual(literature_walk_split(8, 2), (4, 4))
        with self.assertRaises(ValueError):
            literature_walk_split(5, 6)


def _row_data(hetero, r: int = 16, seed: int = 3, dataset: str = "acm"):
    from src.data.dataset import build_dataset

    row = fixed_literature_schema_config(_full(dataset, r=r))
    row.seed = seed
    with patched_acm_loader(hetero):
        return build_dataset(row), row


class DrawTests(unittest.TestCase):
    def test_sampler_asks_for_the_split(self) -> None:
        import src.data.literature_paths as module

        calls = []
        real = module.sample_listed_walks  # spec v1.20 U1: WISP's hop rule (was D8's sample_walks)

        def recording(graph, hops, anchors, walks_per_anchor, mass, generator):
            calls.append((len(hops), walks_per_anchor, anchors.numel()))
            return real(graph, hops, anchors, walks_per_anchor, mass, generator)

        hetero = acm_like_hetero(seed=4)
        with mock.patch.object(module, "sample_listed_walks", recording):
            data, _ = _row_data(hetero, r=16)  # 16 over ACM's 6 paths: not divisible
        self.assertEqual([c[1] for c in calls], [3, 3, 3, 3, 2, 2])
        self.assertEqual(sum(c[1] for c in calls), 16)
        self.assertEqual([c[0] for c in calls],
                         [len(t.node_types) - 1 for t in LITERATURE_METAPATHS["acm"]])
        self.assertTrue(all(c[2] == hetero["paper"].num_nodes for c in calls))  # every target

    def test_every_cell_is_a_listed_meta_path(self) -> None:
        from src.analysis.schema_readout import literature_signatures

        data, _ = _row_data(acm_like_hetero(seed=5), r=16)
        listed = literature_signatures("acm", data.fact_layer, data.relation_tuples,
                                       data.node_type_names)
        allowed = set(listed.values())
        seen = set()
        paper = data.node_type_names.index("paper")
        for group in data.face_groups:
            self.assertIsNotNone(group.signature_index)
            signatures = [tuple(group.signature_table[i]) for i in group.signature_index.tolist()]
            self.assertTrue(set(signatures) <= allowed, set(signatures) - allowed)
            seen |= set(signatures)
            ends = data.node_type[group.face_node_index[:, [0, -1]]]
            self.assertTrue(bool((ends == paper).any(dim=1).all()), "cells start at the target")
            self.assertTrue(bool((group.path_weight > 0).all()))
        self.assertEqual(seen, allowed, "every listed meta-path is drawn on the fixture")

    def test_evaluation_draw_is_deterministic_per_base_seed(self) -> None:
        hetero = acm_like_hetero(seed=6)
        first, _ = _row_data(hetero, seed=11)
        second, _ = _row_data(hetero, seed=11)
        other, _ = _row_data(hetero, seed=12)

        def tensors(data):
            return [t for g in data.face_groups for t in (
                g.face_node_index, g.face_edge_index, g.face_edge_sign, g.relation_index,
                g.path_weight, g.signature_index)]

        self.assertTrue(all(torch.equal(a, b) for a, b in zip(tensors(first), tensors(second))))
        self.assertEqual(len(tensors(first)), len(tensors(second)))
        self.assertFalse(len(tensors(first)) == len(tensors(other)) and all(
            a.shape == b.shape and torch.equal(a, b)
            for a, b in zip(tensors(first), tensors(other))))


class AnchoredForwardTests(unittest.TestCase):
    def test_forward_is_finite_on_an_acm_sized_graph(self) -> None:
        from src.models.h2snn_model import build_model

        counts = ACM_NODE_COUNTS
        hetero = acm_like_hetero(num_papers=counts["paper"], num_authors=counts["author"],
                                 num_subjects=counts["subject"], num_terms=counts["term"],
                                 cites_per_paper=2, authors_per_paper=3, terms_per_paper=10,
                                 seed=7)
        data, row = _row_data(hetero, r=16)
        self.assertGreater(sum(g.num_faces for g in data.face_groups), 0)
        torch.manual_seed(0)
        model = build_model(row, data)
        model.train()
        output = model(data, epoch=0)
        self.assertEqual(output.logits.shape, (counts["paper"], data.num_classes))
        self.assertTrue(bool(torch.isfinite(output.logits).all()))
        loss = torch.nn.functional.cross_entropy(output.logits[data.target_train_mask],
                                                 data.target_y[data.target_train_mask])
        self.assertTrue(bool(torch.isfinite(loss)))
        model.eval()
        with torch.no_grad():
            self.assertTrue(bool(torch.isfinite(model(data).logits).all()))


def _real_layer(dataset: str):
    """Fact layer, kept tuples and node slices of a real HGB graph (data read only)."""
    from src.data.fact_layer import build_fact_layer
    from src.data.hetero_flatten import flatten_edges, stack_node_features

    if dataset == "freebase":  # no H2S-NN loader (Part O out-of-scope finding); PyG reads it
        from torch_geometric.datasets import HGBDataset

        hetero = HGBDataset(root=str(nc_root(REPO_DATA_ROOT)), name="Freebase")[0]
    else:
        from src.data.dataset import _load_hetero_data

        hetero = _load_hetero_data(TrainConfig.from_dict(
            {"data": {"dataset_name": dataset, "root": str(REPO_DATA_ROOT)}}))
    x, node_type, type_names, slices, _ = stack_node_features(hetero)
    edge_index, edge_type, names, tuples = flatten_edges(hetero, slices)
    layer = build_fact_layer(edge_index, edge_type, names, x.size(0), "tied")
    from src.data.fact_layer import apply_fact_layer

    edge_index, edge_type = apply_fact_layer(layer, edge_index, edge_type)
    return layer, tuple(tuples[r] for r in layer.kept), type_names, slices, node_type, \
        edge_index, edge_type


class RealListTests(unittest.TestCase):
    """The card's stop condition: a listed meta-path without an instance on its dataset."""

    TARGETS = {"acm": "paper", "dblp": "author", "imdb": "movie", "freebase": "book"}

    def test_lists_resolve_and_have_instances_on_the_real_graphs(self) -> None:
        from src.data.literature_paths import sample_listed_walks
        from src.data.schemas import resolve_schemas
        from src.data.wisp_sampler import build_action_graph

        present = [d for d in self.TARGETS if (nc_root(REPO_DATA_ROOT) / d).exists()]
        if not present:
            self.skipTest("HGB node-classification data not present")
        for dataset in present:
            with self.subTest(dataset=dataset):
                layer, tuples, types, slices, node_type, edge_index, edge_type = \
                    _real_layer(dataset)
                templates = LITERATURE_METAPATHS[dataset]
                schemas = resolve_schemas(templates, layer, tuples)
                self.assertEqual(len(schemas), len(templates))
                start, end = slices[self.TARGETS[dataset]]
                graph = build_action_graph(edge_index, edge_type, node_type.numel(),
                                           layer.self_inverse)
                generator = torch.Generator().manual_seed(0)
                for schema in schemas:
                    self.assertEqual(schema.node_types[0], self.TARGETS[dataset])
                    # the row's own hop rule (spec v1.20 U1; was D8's sample_walks)
                    walks = sample_listed_walks(graph, schema.hops, torch.arange(start, end), 4,
                                                1.0, generator)
                    self.assertGreater(walks.nodes.size(0), 0,
                                       f"{dataset} {schema.name}: no instance drawn")


if __name__ == "__main__":
    unittest.main()
