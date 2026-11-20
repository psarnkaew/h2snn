"""Card 8 C3: MAGNN adapter (`src/baselines/adapters/magnn.py`).

* "Finite and parameters move" per task class: NC HGB (tiny Freebase release, `run_Freebase.py` path: HGB's
  `get_adjlist_pkl` on HGB's `data_loader`, `MAGNN_nc_mb`, published settings shrunk), NC H2GB (synthetic graph, the
  `t–X–t` rule), LP small (tiny acm_pa task, `MAGNN_lp` through the shared loop's pair scorer), LP large (synthetic
  task, `build_large_lp`).
* Instance format equals HGB's `get_adjlist_pkl` on the tiny Freebase release (adjacency rows and instance arrays).
* LP held-out route: no meta-path instance of any schema of either endpoint side passes through a held-out
  (validation or test) fact; `assert_no_held_out` raises on a crafted instance that does.
* MAGNN's training mask (`parse_side`): an instance whose end hop is a positive pair of the step is dropped.

Card mutation: enumerate the LP instances on the full graph (message graph plus the test facts) ->
`test_no_instance_passes_a_held_out_fact` fails. The real data root is unreachable.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.baselines.protocol import LINK_PREDICTION, NODE_CLASSIFICATION, BaselineParams
from tests.partMB_fixtures import restore_real_data, unreachable_real_data


_ORIGINAL_OPTIMIZER = None


def setUpModule() -> None:
    global _ORIGINAL_OPTIMIZER
    from src.baselines.adapters import magnn

    unreachable_real_data()
    _ORIGINAL_OPTIMIZER = magnn._optimizer


def tearDownModule() -> None:
    restore_real_data()


class _Capture:
    """Wraps `magnn._optimizer` to keep a copy of the trained parameters before training."""

    def __init__(self) -> None:
        self.params: list[torch.Tensor] = []
        self.before: list[torch.Tensor] = []

    def __call__(self, parameters, args):
        self.params = list(parameters)
        self.before = [p.detach().clone() for p in self.params]
        return _ORIGINAL_OPTIMIZER(self.params, args)

    def moved(self) -> bool:
        return any(not torch.equal(b, p) for b, p in zip(self.before, self.params))


SMALL = {"hidden_dim": 8, "num_heads": 2, "attn_vec_dim": 8, "samples": 5, "dropout": 0.0}


class FreebaseNCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.baselines.bundles import build_bundle
        from tests.partP_freebase_fixture import write_freebase_release

        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        write_freebase_release(cls.root)
        cls.bundle = build_bundle(NODE_CLASSIFICATION, "freebase",
                                  overrides={"device": "cpu", "data": {"root": str(cls.root)}})

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_freebase_published_path_is_finite_and_moves(self) -> None:
        from src.baselines.adapters import magnn

        capture = _Capture()
        values = {**SMALL, "epochs": 1, "early_stopping": 1, "batch_size": 8, "optimizer": "adam", "feats_type": 2,
                  "lr": 0.005, "weight_decay": 0.001, "rnn_type": "RotatE0"}
        with mock.patch.object(magnn, "_optimizer", capture):
            record = magnn.ADAPTER.fit_task(self.bundle, BaselineParams(values, source="published"),
                                            task=NODE_CLASSIFICATION, seed=0, device=torch.device("cpu"))
        for key in ("test_macro_f1", "val_macro_f1"):
            self.assertTrue(math.isfinite(float(record[key])), key)
        self.assertTrue(capture.moved(), "no parameter moved")

    def test_our_instances_equal_hgbs_get_adjlist_pkl(self) -> None:
        """Our enumeration (used off HGB) gives HGB's format on HGB's own BB (link 0 and its reverse)."""
        from src.baselines.adapters.magnn import TypedGraph, magnn_lists, path_instances
        from src.baselines.bundles import hgb_raw_dir
        from src.baselines.runtime import pushd_sys_path

        with pushd_sys_path("hgb_magnn_nc", "hgb_nc"):
            from scripts.data_loader import data_loader  # type: ignore[import-not-found]
            from utils.data import get_adjlist_pkl  # type: ignore[import-not-found]

            dl = data_loader(str(hgb_raw_dir("freebase", self.root)))
            adj, idx = get_adjlist_pkl(dl, [0, -1], symmetric=True)
        matrix = dl.links["data"][0].tocoo()
        count = dl.nodes["count"][0]
        graph = TypedGraph(("book",), {"book": (0, count)}, (("book", "and", "book"),),
                           [(matrix.row.astype(np.int64), matrix.col.astype(np.int64))])
        paths = path_instances([graph.hop(("book", "and", "book"))[0], graph.hop(("book", "rev_and", "book"))[0]],
                               count)
        ours_adj, ours_idx = magnn_lists(paths, np.arange(count))
        for node in range(count):
            self.assertEqual(ours_adj[node], adj[node])
            theirs = idx.get(node, np.empty((0, 3)))
            self.assertTrue(np.array_equal(ours_idx[node], np.asarray(theirs).reshape(-1, 3)), node)


class H2GBNCTests(unittest.TestCase):
    def test_rule_schemas_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters.magnn import TypedGraph, build_nc_model, nc_parse, rule_schemas, schema_lists
        from tests.test_baselines_minibatch import synthetic_hetero

        data = synthetic_hetero()
        graph = TypedGraph.from_hetero(data)
        schemas = rule_schemas(graph, "a", id_base=0)
        self.assertEqual(sorted(s.name for s in schemas), ["a-a-a", "a-b-a"])
        lo, hi = graph.slices["a"]
        adjlists, indices = schema_lists(graph, schemas, np.arange(lo, hi))
        feats = [data[t].x.float() for t in data.node_types]
        torch.manual_seed(0)
        np.random.seed(0)
        model = build_nc_model(len(schemas), 2 * len(graph.type_pairs()), [s.etypes for s in schemas],
                               [f.size(1) for f in feats], 8, 3, 2, 8, "RotatE0", 0.0)
        batch = np.arange(lo, lo + 8)
        g_list, idx_list, mapped = nc_parse(adjlists, indices, batch, torch.device("cpu"), 5)
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        logits, _ = model((g_list, feats, graph.type_mask, idx_list, mapped))
        loss = torch.nn.functional.cross_entropy(logits, data["a"].y[:8])
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, model.parameters())))


class LPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from tests.test_partP_derived_lp import build_tiny_task, tiny_bundle

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        build_tiny_task(root, "acm_pa")
        cls.bundle = tiny_bundle(root, "acm_pa")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_lp_small_is_finite_and_moves(self) -> None:
        from src.baselines.adapters import magnn

        capture = _Capture()
        with mock.patch.object(magnn, "_optimizer", capture):
            record = magnn.ADAPTER.fit_task(self.bundle, BaselineParams({**SMALL, "epochs": 1, "early_stopping": 1}),
                                            task=LINK_PREDICTION, seed=0, device=torch.device("cpu"))
        self.assertTrue(math.isfinite(float(record["test_random_auroc"])))
        self.assertTrue(capture.moved(), "no parameter moved")

    def test_no_instance_passes_a_held_out_fact(self) -> None:
        from src.baselines.adapters.magnn import ADAPTER, run_arguments

        args = run_arguments(SMALL)
        args["feats_type"] = 3
        _, _, sides = ADAPTER.lp_setup(self.bundle, args, torch.device("cpu"))  # raises on a held-out instance
        self.assertTrue(sides[0] and sides[1])
        self.assertTrue(any(s.use_mask for s in sides[0] + sides[1]), "no schema runs through the target relation")

    def test_assert_no_held_out_raises_on_a_crafted_instance(self) -> None:
        from src.baselines.adapters.magnn import assert_no_held_out

        paths = np.array([[0, 5, 1], [2, 6, 3]])
        assert_no_held_out(paths, {(7, 8)})
        with self.assertRaises(AssertionError):
            assert_no_held_out(paths, {(6, 2)})  # the reverse orientation of hop (2, 6)

    def test_training_mask_drops_instances_through_a_step_positive(self) -> None:
        from src.baselines.adapters.magnn import parse_side

        adjlist = {0: "0 5 9"}
        indices = {0: np.array([[5, 1, 0], [9, 2, 0]])}  # far end, middle, centre
        _, kept, _ = parse_side(adjlist, indices, [0], None, None, torch.device("cpu"))
        self.assertEqual(kept.size(0), 2)
        _, kept, _ = parse_side(adjlist, indices, [0], None, {(0, 1)}, torch.device("cpu"))
        self.assertEqual(kept.tolist(), [[9, 2, 0]])


class LargeLPTests(unittest.TestCase):
    def test_lp_large_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters.magnn import ADAPTER
        from src.baselines.heads import SharedLinkHead
        from src.baselines.large_lp_minibatch import held_out_pairs, message_graph
        from src.baselines.large_lp_training import Sampler, step_scores
        from tests.test_baselines_large_lp import SAMPLERS, _lp_data, synthetic_task

        data, files = synthetic_task()
        graph = message_graph(data, "user", held_out_pairs(files))
        pos = torch.from_numpy(files["split_train"].T.copy())[:, :8]
        neg = torch.stack([pos[0], torch.randint(0, 40, (8,), generator=torch.Generator().manual_seed(1))])
        links = torch.cat([pos, neg], dim=1)
        torch.manual_seed(0)
        np.random.seed(0)
        mode, fanout = SAMPLERS["hgt, per-type lists seeded by endpoints (G-8)"]
        model, sampler = ADAPTER.build_large_lp(_lp_data(graph), BaselineParams(SMALL), torch.device("cpu"),
                                                Sampler(mode, fanout))
        head = SharedLinkHead("dot", num_relations=1, dim=model.embed_dim)
        params = list(model.modules.parameters()) + list(head.parameters())
        before = [p.detach().clone() for p in params]
        scores = step_scores(model, head, sampler, graph, ("user", "has_friend", "user"), "user", links,
                             torch.device("cpu"))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, torch.cat([torch.ones(8), torch.zeros(8)]))
        self.assertTrue(torch.isfinite(loss).item())
        optimizer = torch.optim.AdamW(params, lr=1e-2)
        loss.backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, params)))


if __name__ == "__main__":
    unittest.main()
