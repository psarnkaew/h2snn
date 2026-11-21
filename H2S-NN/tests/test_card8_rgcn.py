"""Card 8 C2: R-GCN adapter (`src/baselines/adapters/rgcn.py`).

* One "finite loss and parameters move" per loop class: HGB NC (tiny Freebase release, HGB's `EntityClassify` on HGB's
  `BaseRGCN`), H2GB NC (H2GB's `HomoGNNModel` with `layer_type: RGCN`, the published mag-year yaml shrunk), LP small
  (tiny acm_pa task, HGB's LP `RGCN`), LP large (synthetic task, `build_large_lp`).
* HGB's LP graph rule (`link_predict.py:141-162`) replayed literally with Python dicts equals `lp_relation_graph`.
* The adapter trains on the bundle's own masks (NC) and the bundle's LP data (split, negatives) unchanged.
The real data root is unreachable.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.baselines.protocol import LINK_PREDICTION, NODE_CLASSIFICATION, BaselineParams
from tests.partMB_fixtures import restore_real_data, unreachable_real_data


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def _moves(test: unittest.TestCase, modules: list[torch.nn.Module], loss_fn) -> None:
    params = [p for m in modules for p in m.parameters()]
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    loss = loss_fn()
    test.assertTrue(torch.isfinite(loss).item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    test.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, params)), "no parameter moved")


class _Freebase(unittest.TestCase):
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


class HGBNodeTests(_Freebase):
    def test_hgb_nc_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters.rgcn import entity_classify_model, nc_relation_graph

        data = self.bundle.data
        view = self.bundle.homogeneous(feats_type=3)
        graph, etype, norm = nc_relation_graph(data.edge_index, data.edge_type, int(view.num_nodes))
        torch.manual_seed(0)
        model = entity_classify_model(view.in_dims, 8, self.bundle.num_classes, len(data.relation_names), -1, 3, 0.0)
        start, end = view.node_slices[view.target_node_type]
        train = data.target_train_mask.nonzero().view(-1)
        features = [f.coalesce() if f.is_sparse else f for f in view.features_list]
        _moves(self, [model], lambda: torch.nn.functional.cross_entropy(
            model(graph, features, etype, norm)[start:end][train], data.target_y[train]))

    def test_fit_task_uses_the_bundle_masks(self) -> None:
        from src.baselines.adapters import rgcn

        seen = {}
        original = rgcn.train_full_batch_node_classification

        def spy(**kwargs):
            seen.update(kwargs)
            return original(**kwargs)

        with mock.patch.object(rgcn, "train_full_batch_node_classification", spy):
            record = rgcn.ADAPTER.fit_task(self.bundle, BaselineParams({"epochs": 1, "early_stopping": 1,
                                                                        "hidden_dim": 8}),
                                           task=NODE_CLASSIFICATION, seed=0, device=torch.device("cpu"))
        data = self.bundle.data
        for split in ("train", "val", "test"):
            self.assertTrue(torch.equal(seen[f"{split}_idx"].cpu(),
                                        getattr(data, f"target_{split}_mask").nonzero().view(-1)))
        self.assertTrue(math.isfinite(float(record["test_macro_f1"])))


class H2GBNodeTests(unittest.TestCase):
    def test_h2gb_rgcn_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters._h2gb_network import seed_predictions
        from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
        from src.baselines.h2gb_published import build_h2gb_network, resolve_published
        from src.baselines.minibatch import hgt_loader
        from src.baselines.published import load_published_params
        from src.baselines.runtime import pushd_sys_path
        from tests.test_baselines_minibatch import synthetic_hetero

        values = dict(load_published_params("rgcn", NODE_CLASSIFICATION, "mag-year").values)
        self.assertEqual(values["gnn.layer_type"], "RGCN")
        values.update({"gnn.dim_inner": 8, "gnn.layers_mp": 2})
        data = synthetic_hetero()
        batch = next(iter(hgt_loader(data, {t: [10 ** 6] * 2 for t in data.node_types}, "a", "train", 8,
                                     shuffle=False, steps=1)))
        ensure_pyg_submodules()
        with pushd_sys_path("h2gb"):
            cfg, _ = resolve_published("rgcn", "mag-year", data, "a", torch.device("cpu"), values)
            torch.manual_seed(0)
            network = build_h2gb_network(data, "synthetic", 3, cfg)
            self.assertEqual(type(network).__name__, "HomoGNNModel")
            size = int(batch["a"].batch_size)
            _moves(self, [network], lambda: torch.nn.functional.cross_entropy(
                seed_predictions(network, batch.clone(), "a"), batch["a"].y[:size]))


class LPGraphRuleTests(unittest.TestCase):
    def test_lp_graph_equals_hgbs_dict_rule(self) -> None:
        from src.baselines.adapters.rgcn import lp_relation_graph

        g = torch.Generator().manual_seed(0)
        n, L = 12, 3
        src = torch.randint(0, n, (40,), generator=g)
        dst = torch.randint(0, n, (40,), generator=g)
        rel = torch.randint(0, L, (40,), generator=g)
        # HGB's code (link_predict.py:141-150), literally, relations visited in id order
        edge2type = {}
        for k in range(L):
            for u, v in zip(src[rel == k].tolist(), dst[rel == k].tolist()):
                edge2type[(u, v)] = k
        for i in range(n):
            if (i, i) not in edge2type:
                edge2type[(i, i)] = L
        for k in range(L):
            for u, v in zip(src[rel == k].tolist(), dst[rel == k].tolist()):
                if (v, u) not in edge2type:
                    edge2type[(v, u)] = k + 1 + L
        graph, types, norm = lp_relation_graph(torch.stack([src, dst]), rel, n, L)
        u, v = graph.edges()
        ours = {(a, b): t for a, b, t in zip(u.tolist(), v.tolist(), types.tolist())}
        pairs = {(a, b) for a, b in zip(src.tolist(), dst.tolist()) if a != b}
        pairs |= {(b, a) for a, b in pairs} | {(i, i) for i in range(n)}
        self.assertEqual(set(ours), pairs)
        self.assertEqual(ours, {p: edge2type[p] for p in pairs})
        degree = torch.bincount(v, minlength=n).float()
        self.assertTrue(torch.allclose(norm.view(-1), 1.0 / degree[v]))


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

    def test_lp_small_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters.rgcn import _features, lp_relation_graph, lp_rgcn_model
        from src.baselines.heads import SharedLinkHead

        data = self.bundle.data
        view = self.bundle.homogeneous(feats_type=0)
        L = len(data.relation_names)
        graph, etype, norm = lp_relation_graph(data.edge_index, data.edge_type, int(view.num_nodes), L)
        torch.manual_seed(0)
        model = lp_rgcn_model(view.in_dims, 8, L + 1, 4, 2, 0.0)
        head = SharedLinkHead("distmult", num_relations=1, dim=8)
        features = _features(view, torch.device("cpu"))
        pos = data.train_pos_edge_index
        neg = torch.stack([pos[0], pos[1].flip(0)])

        def loss():
            h = model(graph, features, etype, norm)
            scores = torch.cat([head(h, pos), head(h, neg)])
            return torch.nn.functional.binary_cross_entropy_with_logits(
                scores, torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]))

        _moves(self, [model, head], loss)

    def test_fit_task_trains_on_the_bundle_lp_data(self) -> None:
        from src.baselines.adapters import rgcn

        seen = {}
        original = rgcn.train_link_prediction

        def spy(**kwargs):
            seen.update(kwargs)
            return original(**kwargs)

        with mock.patch.object(rgcn, "train_link_prediction", spy):
            record = rgcn.ADAPTER.fit_task(self.bundle, BaselineParams(
                {"epochs": 1, "early_stopping": 1, "hidden_dim": 8, "n_bases": 4}), task=LINK_PREDICTION, seed=0,
                device=torch.device("cpu"))
        data = self.bundle.data
        for name in ("train_pos_edge_index", "val_pos_edge_index", "val_neg_edge_index", "test_pos_edge_index"):
            self.assertTrue(torch.equal(getattr(seen["data"], name).cpu(), getattr(data, name)), name)
        self.assertEqual(seen["positives_per_step"], 8192)
        self.assertTrue(math.isfinite(float(record["test_random_auroc"])))


class LargeLPTests(unittest.TestCase):
    def test_lp_large_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters.rgcn import ADAPTER
        from src.baselines.heads import SharedLinkHead
        from src.baselines.large_lp_minibatch import held_out_pairs, message_graph
        from src.baselines.large_lp_training import Sampler, step_scores
        from tests.test_baselines_large_lp import _lp_data, synthetic_task

        data, files = synthetic_task()
        graph = message_graph(data, "user", held_out_pairs(files))
        pos = torch.from_numpy(files["split_train"].T.copy())[:, :8]
        neg = torch.stack([pos[0], torch.randint(0, 40, (8,), generator=torch.Generator().manual_seed(1))])
        links = torch.cat([pos, neg], dim=1)
        torch.manual_seed(0)
        model, sampler = ADAPTER.build_large_lp(_lp_data(graph), BaselineParams(
            {"hidden_dim": 8, "n_bases": 4, "num_layers": 2, "dropout": 0.0}), torch.device("cpu"),
            Sampler("link", [-1, -1]))
        head = SharedLinkHead("dot", num_relations=1, dim=model.embed_dim)

        def loss():
            scores = step_scores(model, head, sampler, graph, ("user", "has_friend", "user"), "user", links,
                                 torch.device("cpu"))
            out = torch.nn.functional.binary_cross_entropy_with_logits(
                scores, torch.cat([torch.ones(8), torch.zeros(8)]))
            return out + model.aux_loss()

        _moves(self, [model.modules, head], loss)


if __name__ == "__main__":
    unittest.main()
