"""Card 2 D4b: H2S-NN on the two large LP tasks, on a synthetic task (no real data; the data root is unreachable).

The synthetic H-Pokec U-U task (users and clubs; the target relation stored in both orientations, a parallel
user-user relation carrying held-out pairs, user-club facts) goes through the baselines' own path: `build_bundle`
gives the `LargeLinkBundle`, `load_large_lp` builds the message graph from H2GB's graph and the task files; only
H2GB's loader and the task-file reader are replaced by the synthetic graph and files.

(1) held-out route: no validation / test positive (either orientation, any relation) is a fact of any H2S-NN scope
    of the training, validation or test links. Card mutation: scopes on the full graph instead of the message graph
    -> this test fails (`docs/dryrun_evidence/d4b_mutation.py`).
(2) same data: H2S-NN's message graph is the bundle's (`torch.equal` on its facts in H2S-NN's layout) and its pairs
    are the bundle's (`torch.equal` on every split and negative set).
(3) one LP step through the trainer: the loss is finite and every trained parameter group moves.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

REPO = Path(__file__).resolve().parents[1]
TASK = "h-pokec_uu"


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def synthetic_task(seed: int = 0):
    """H2GB-like graph of the task's NC dataset and the task files (type-local pairs `[L, 2]`)."""
    from torch_geometric.data import HeteroData

    from src.data.large_lp import split_pairs, unordered_pairs

    g = torch.Generator().manual_seed(seed)
    n_user, n_club = 40, 8
    raw = torch.stack([torch.randint(0, n_user, (140,), generator=g), torch.randint(0, n_user, (140,), generator=g)])
    raw = raw[:, raw[0] != raw[1]]
    pairs, _ = unordered_pairs(raw)
    train, valid, test = split_pairs(pairs, seed=3, val_frac=0.15, test_frac=0.15)
    data = HeteroData()
    data["user"].x = torch.randn(n_user, 4, generator=g)
    data["club"].x = torch.randn(n_club, 3, generator=g)
    data["user", "has_friend", "user"].edge_index = torch.from_numpy(np.concatenate([pairs, pairs[:, ::-1]]).T.copy())
    extra = torch.from_numpy(np.concatenate([valid[:3], test[:3]]).T.copy())  # held-out pairs on another relation
    data["user", "messaged", "user"].edge_index = torch.cat([extra, extra.flip(0)], dim=1)
    uc = torch.stack([torch.randint(0, n_user, (50,), generator=g), torch.randint(0, n_club, (50,), generator=g)])
    data["user", "joins", "club"].edge_index = uc
    data["club", "rev_joins", "user"].edge_index = uc.flip(0)
    rng = np.random.default_rng(seed)

    def negatives(n: int) -> np.ndarray:
        return rng.integers(0, n_user, size=(n, 2)).astype(np.int64)

    files = {"split_train": train, "split_valid": valid, "split_test": test,
             "negatives_valid": negatives(len(valid)), "negatives_test_2hop": negatives(len(test)),
             "negatives_test_random": negatives(len(test))}
    return data, files


def synthetic_loaders(data, files):
    """The two readers `load_large_lp` calls, replaced by the synthetic graph and files."""
    return (mock.patch("src.data.h2gb_nc.load_h2gb_nc", lambda key, root, seed=None: data),
            mock.patch("src.baselines.large_lp_training.read_task_files", lambda root, task: dict(files)))


def config(tmp: str, epochs: int = 1):
    from src.configs.config import TrainConfig

    raw = yaml.safe_load((REPO / "configs" / "experiments" / f"{TASK}_h2snn_v3b.yaml").read_text())
    raw["data"].update({"wisp_walks_per_anchor": 3, "wisp_max_length": 3})
    raw["model"].update({"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                         "classifier_hidden_dim": 16, "solver_max_iterations": 200})
    raw.update({"device": "cpu", "epochs": epochs, "output_dir": tmp, "use_wandb": False})
    return TrainConfig.from_dict(raw)


def build(tmp: str, data, files, epochs: int = 1):
    from src.builders import build_experiment_components

    cfg = config(tmp, epochs)
    a, b = synthetic_loaders(data, files)
    with a, b:
        torch.manual_seed(0)
        return cfg, build_experiment_components(cfg)


def fact_pairs(graph, fact_ids: torch.Tensor) -> set[tuple[int, int]]:
    return {tuple(p) for p in graph.edge_index[:, fact_ids].t().tolist()}


class HeldOutRouteTests(unittest.TestCase):
    def test_no_held_out_positive_is_a_fact_of_any_scope(self) -> None:
        import src.trainers.large_lp_trainer as large
        from src.data.scopes import build_scope_batch

        data, files = synthetic_task()
        with tempfile.TemporaryDirectory() as tmp:
            cfg, (lp, model, *_rest) = build(tmp, data, files)
            scopes = large.link_scopes(model, lp, cfg)  # the trainer's scopes (the mutation replaces this)
            held = {tuple(p) for p in lp.held_out().t().tolist()}
            held |= {(v, u) for u, v in held}
            # the full graph does hold some held-out pairs (so the test can fail)
            full = {tuple(p) for p in (data["user", "has_friend", "user"].edge_index
                                       + lp.dst_range[0]).t().tolist()}
            self.assertTrue(full & held)
            checked = 0
            for seed, links in enumerate((torch.cat([lp.train_pos, lp.train_pos.flip(0)], 1), lp.val_pos, lp.val_neg,
                                          lp.test_pos, lp.test_2hop, lp.test_random)):
                endpoints = torch.unique(links.reshape(-1))
                cells = scopes.cells(endpoints, seed)
                batch = build_scope_batch(scopes.graph, endpoints, [cells.of(int(t)) for t in endpoints.tolist()])
                facts = fact_pairs(scopes.graph, batch.global_fact_id)
                self.assertEqual(facts & held, set(), f"links set {seed}")
                checked += len(facts)
            self.assertGreater(checked, 0)


class SameDataTests(unittest.TestCase):
    def test_message_graph_and_pairs_are_the_bundles(self) -> None:
        from src.baselines.bundles import LargeLinkBundle, build_bundle
        from src.baselines.large_lp_training import load_large_lp
        from src.baselines.protocol import LINK_PREDICTION
        from src.data.fact_layer import apply_fact_layer, build_fact_layer
        from src.data.hetero_flatten import flatten_edges, stack_node_features

        data, files = synthetic_task()
        with tempfile.TemporaryDirectory() as tmp:
            cfg, (lp, *_rest) = build(tmp, data, files)
            a, b = synthetic_loaders(data, files)
            with a, b:  # the baselines' path, as fit_large_lp
                bundle = build_bundle(LINK_PREDICTION, TASK, seed=cfg.seed)
                self.assertIsInstance(bundle, LargeLinkBundle)
                base = load_large_lp(bundle.dataset, bundle.metadata["data_root"])
        # message graph: the bundle's facts in H2S-NN's layout, then the same fact layer
        _, _, _, slices, _ = stack_node_features(base.graph)
        edge_index, edge_type, names, _ = flatten_edges(base.graph, slices)
        layer = build_fact_layer(edge_index, edge_type, names, sum(e - s for s, e in slices.values()),
                                 cfg.data.fact_layer)
        edge_index, edge_type = apply_fact_layer(layer, edge_index, edge_type)
        self.assertTrue(torch.equal(lp.graph.edge_index, edge_index))
        self.assertTrue(torch.equal(lp.graph.edge_type, edge_type))
        # pairs: every split and negative set, type-local
        start = lp.dst_range[0]
        for name in ("train_pos", "val_pos", "val_neg", "test_pos", "test_2hop", "test_random"):
            with self.subTest(pairs=name):
                self.assertTrue(torch.equal(getattr(lp, name) - start, getattr(base, name)))
        self.assertEqual(lp.dst_range, slices[base.node_type])

    def test_task_tables(self) -> None:
        from src.configs.config import DATASET_TASK_TYPES, LINK_PREDICTION_PRESETS, TrainConfig
        from src.data.large_lp import LARGE_LP_TASKS

        for task, spec in LARGE_LP_TASKS.items():
            with self.subTest(task=task):
                self.assertEqual(DATASET_TASK_TYPES[task], "link_prediction")
                self.assertEqual(LINK_PREDICTION_PRESETS[task]["target_relation"], spec.relation)
                raw = yaml.safe_load((REPO / "configs" / "experiments" / f"{task}_h2snn_v3b.yaml").read_text())
                cfg = TrainConfig.from_dict(raw)
                self.assertEqual((cfg.data.training_mode, tuple(cfg.data.target_relation)), ("minibatch", spec.relation))


class OneStepTests(unittest.TestCase):
    def test_one_lp_step_is_finite_and_parameters_move(self) -> None:
        data, files = synthetic_task()
        with tempfile.TemporaryDirectory() as tmp:
            cfg, (lp, model, _opt, _sch, logger, _ck, trainer) = build(tmp, data, files, epochs=1)
            from src.trainers.large_lp_trainer import LargeLinkTrainer

            self.assertIsInstance(trainer, LargeLinkTrainer)
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            try:
                trainer.fit(lp)
            finally:
                logger.close()
            loop = trainer.loop
            self.assertEqual(loop.epoch_steps(lp.train_pos.size(1)), 1)  # P < 8,192: one step = one epoch
            self.assertTrue(np.isfinite(loop.history[0]["loss"]))
            self.assertIs(loop.scopes.graph.edge_index, lp.graph.edge_index)  # scopes on the message graph
            moved = [n for n, p in model.named_parameters() if p.requires_grad and not torch.equal(p, before[n])]
            self.assertTrue(moved)
            for group in trainer.optimizer.param_groups:  # both groups (scalars and the rest) are trained
                self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()
                                    if any(p is q for q in group["params"])))


if __name__ == "__main__":
    unittest.main()
