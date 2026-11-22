"""Card 5 (`docs/H2S-NN_card5_card.md`) T1-T9: B1a for H2S-NN on the H2GB NC datasets, the read-only analysis hooks
(`src/analysis/mechanism_hooks.py`), the face knockout and the TSUBAME phase `h2snn`. Synthetic data only.

T1 B1a scoring (ACM-shaped graph registered as a subset dataset): subset only during training, no test id before the
   end, the restored best weights scored once on the whole validation and test splits, the record's keys.
T2 PDNS: the subset follows the run seed and equals `validation_subset`; no run seed raises.
T3 selection: `summarize_tuned_seed` ranks by `sweep_objective`; a subset-dataset record without the key raises.
T4 HGB unchanged: full-batch NC and the ACM mini-batch flag give the base commit's records bit for bit.
T5 energy: the hook equals 1/2 Y^T L Y of the dense reference; sum e_v + fact-cell = E; e_v = 0 at the Thm 4
   section; at a model's solution the same identities hold.
T6 knockout: empty = identical; all nodes = the no-cell solve; one node zeroes exactly its cells' beta_f; refused in
   training and tuning; full batch and mini-batch.
T7 transports: Q_e x = F_src^-1 F_tgt x; Q^T Q = I.
T8 cells per target: node ids, schema keys, w_f and beta_f equal the model's values of the same forward.
T9 cell list: 24 tuning lines in order, every line parses with the launcher's parser; `baselines-1` unchanged.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT.parent / "docs" / "card5_evidence"
F64 = torch.float64
SMALL = {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
         "classifier_hidden_dim": 16, "propagation": "anchored"}


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def _model(arm: str = "full", ablation: dict | None = None, state: dict | None = None):
    """A float64 model on the ACM-shaped graph, in eval mode (the warm start cleared per forward)."""
    from src.configs.config import TrainConfig
    from src.data.dataset import build_dataset
    from src.models.h2snn_model import build_model
    from tests.partO_fixtures import acm_like_hetero, patched_acm_loader

    cfg = TrainConfig.from_dict({"device": "cpu", "seed": 3,
                                 "data": {"dataset_name": "acm", "wisp_max_length": 3,
                                          "wisp_walks_per_anchor": 5},
                                 "model": {**SMALL, "arm": arm}, "ablation": ablation or {}})
    with patched_acm_loader(acm_like_hetero(seed=9)):
        data = build_dataset(cfg)
    data.x = data.x.to(F64)
    torch.manual_seed(0)
    model = build_model(cfg, data).to(F64)
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return cfg, data, model


def _logits(model, data) -> torch.Tensor:
    model._warm_start = None
    with torch.no_grad():
        return model(data).logits


# ----------------------------------------------------------------------------- T1-T3 (U1)
class B1aTests(unittest.TestCase):
    def test_T1_subset_only_during_training_then_one_whole_split_pass(self) -> None:
        from src.evaluation.validation_subset import subset_fingerprint, validation_subset
        from tests.test_partMB_trainer import trainer_for

        with mock.patch.dict("src.evaluation.validation_subset.SUBSET_SIZES", {"acm": 2}), \
                trainer_for(8, 16, epochs=3) as (cfg, graph, model, tr):
            calls: list[torch.Tensor] = []
            original = tr.evaluate

            def spy(eval_graph, split_mask, *args, **kwargs):
                out = original(eval_graph, split_mask, *args, **kwargs)
                calls.append(out[0].clone())
                return out
            tr.evaluate = spy
            tr.fit(graph)
            start = graph.node_slices[cfg.data.target_node_type][0]
            val_ids = graph.target_indices[graph.target_val_mask]
            test_ids = graph.target_indices[graph.target_test_mask]
            subset_local = validation_subset("acm", val_ids - start, cfg.seed)
            self.assertLess(subset_local.numel(), val_ids.numel())
            during, final_val, final_test = calls[:-2], calls[-2], calls[-1]
            self.assertEqual(len(during), 3)                       # eval_period 1, 3 epochs
            for ids in during:
                self.assertTrue(torch.equal(torch.sort(ids).values, subset_local + start))
                self.assertFalse(bool(torch.isin(ids, test_ids).any()))
            self.assertTrue(torch.equal(torch.sort(final_val).values, torch.sort(val_ids).values))
            self.assertTrue(torch.equal(torch.sort(final_test).values, torch.sort(test_ids).values))
            record = tr._best_test_metrics
            for key in ("val_subset_macro_f1", "val_subset_size", "val_subset_fingerprint", "dataset",
                        "val_macro_f1", "test_macro_f1", "best_epoch"):
                self.assertIn(key, record)
            self.assertEqual(record["dataset"], "acm")
            self.assertEqual(record["val_subset_size"], float(subset_local.numel()))
            self.assertEqual(record["val_subset_fingerprint"], subset_fingerprint(subset_local))
            saved = json.loads((cfg.exp_dir / "best_test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["val_subset_fingerprint"], record["val_subset_fingerprint"])
            best = torch.load(cfg.exp_dir / "checkpoints" / "best.pt", map_location="cpu",
                              weights_only=False)
            self.assertEqual(best["epoch"], record["best_epoch"])
            restored = {k: v for k, v in model.state_dict().items()}
            self.assertTrue(all(torch.equal(restored[k].cpu(), v) for k, v in best["state_dict"].items()))

    def test_T2_pdns_subset_follows_the_run_seed(self) -> None:
        from src.evaluation.validation_subset import validation_subset
        from tests.test_partMB_trainer import trainer_for

        with mock.patch.dict("src.evaluation.validation_subset.SUBSET_SIZES", {"pdns": 2}), \
                trainer_for(8, 16) as (cfg, graph, _, tr):
            start = graph.node_slices[cfg.data.target_node_type][0]
            local_val = graph.target_indices[graph.target_val_mask] - start
            subsets = []
            for seed in range(1, 7):
                tr.cfg = SimpleNamespace(data=SimpleNamespace(dataset_name="pdns",
                                                              target_node_type=cfg.data.target_node_type),
                                         seed=seed, checkpoint_metric="macro_f1")
                _, ids = tr.validation_subset_mask(graph)
                self.assertTrue(torch.equal(ids, validation_subset("pdns", local_val, seed)))
                subsets.append(tuple(ids.tolist()))
            self.assertGreater(len(set(subsets)), 1)
            tr.cfg = SimpleNamespace(data=SimpleNamespace(dataset_name="pdns",
                                                          target_node_type=cfg.data.target_node_type),
                                     seed=None, checkpoint_metric="macro_f1")
            with self.assertRaises(ValueError):
                tr.validation_subset_mask(graph)

    def test_T3_selection_by_the_subset_metric(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_h2snn_wandb_tuning import summarize_tuned_seed

        a = {"dataset": "ogbn-mag", "val_macro_f1": 0.9, "val_subset_macro_f1": 0.5, "trial_name": "a"}
        b = {"dataset": "ogbn-mag", "val_macro_f1": 0.6, "val_subset_macro_f1": 0.7, "trial_name": "b"}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "best.json"
            self.assertEqual(summarize_tuned_seed([a, b], monitor="macro_f1", out_path=out)["trial_name"], "b")
            with self.assertRaises(ValueError):
                summarize_tuned_seed([{"dataset": "ogbn-mag", "val_macro_f1": 0.9}], monitor="macro_f1",
                                     out_path=out)
            hgb = [{"dataset": "acm", "val_macro_f1": 0.6, "trial_name": "x"},
                   {"dataset": "acm", "val_macro_f1": 0.8, "trial_name": "y"}]
            self.assertEqual(summarize_tuned_seed(hgb, monitor="macro_f1", out_path=out)["trial_name"], "y")


class HgbUnchangedTests(unittest.TestCase):
    def test_T4_records_equal_the_base_commit(self) -> None:
        import importlib.util

        from tests.pinned_kernels import assert_kernel_path_pinned

        assert_kernel_path_pinned(self)
        spec = importlib.util.spec_from_file_location("capture_base", EVIDENCE / "capture_base.py")
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)  # type: ignore[union-attr]
        # Merge c12+c14 (author 2026-11-22): card 14 changed the random stream (counter-based SplitMix64), so the
        # H2S-NN parts were re-captured with the same capture_base.py on the merged code ec4c607; the compare is
        # unchanged.
        reference = json.loads((EVIDENCE / "base_capture_ec4c607.json").read_text(encoding="utf-8"))
        threads = torch.get_num_threads()
        try:
            now = capture.records()
        finally:
            torch.set_num_threads(threads)
        self.assertEqual(now["hgb_full_batch"], reference["hgb_full_batch"])
        self.assertEqual(now["acm_minibatch_flag"], reference["acm_minibatch_flag"])
        # T9 part. Ops card T7.4 (v3b 1646efd) moved HetSheaf and CSNN on ACM/DBLP NC to phase `baselines-early` on
        # purpose, which puts 4 `# in phase baselines-early` lines in the baselines-1 list; the list is compared with
        # its capture at 1646efd (same capture_base.py; docs/card7_evidence/capture_1646efd_*.txt shows that this is
        # the only difference from 894fd74). The model parts above are compared with ec4c607.
        # Card 8 C5 (author 2026-11-20) rebuilt the baselines-1 list on purpose (tune cells only; filled, published and
        # probe cells are comment lines), so it no longer equals the 1646efd capture; its correctness is tested by
        # test_card8_phases (coverage, disjointness, h2snn / baselines-early byte-identical) and test_tsubame_*. Every
        # tuning line it keeps for a method that existed at 1646efd (not R-GCN or MAGNN, card 8 C2/C3) is still a line
        # of the capture:
        cells = json.loads((EVIDENCE / "base_capture_1646efd.json").read_text(encoding="utf-8"))
        kept = [line for line in now["baselines_1_cell_list"].splitlines() if line and not line.startswith("#")
                and "--models rgcn " not in line and "--models magnn " not in line]
        # Card 15 (author 2026-11-22): LP acm_pa is replaced by acm_pp; the capture's acm_pa lines are compared as acm_pp.
        captured = {line.replace("acm_pa", "acm_pp") for line in cells["baselines_1_cell_list"].splitlines()}
        self.assertTrue(kept)
        self.assertEqual([line for line in kept if line not in captured], [])


# ----------------------------------------------------------------------------- T5 (energy)
class EnergyTests(unittest.TestCase):
    def test_T5_energy_terms_against_the_dense_reference_and_the_thm4_section(self) -> None:
        from src.analysis.mechanism_hooks import energy_terms
        from tests.test_sheaf_energy import Fixture
        from tests.test_td1_maps import thm4_production

        fx = Fixture(dtype=F64)
        op = fx.operator()
        L, _ = fx.dense_reference()
        g = torch.Generator().manual_seed(11)
        y = torch.randn(op.layout.rows, fx.d, generator=g, dtype=F64)
        parts = energy_terms(op, *op.layout.split(y))
        dense = 0.5 * float(y.reshape(-1) @ L @ y.reshape(-1))
        self.assertLessEqual(abs(float(parts.energy) - dense), 1e-10)
        self.assertLessEqual(abs(float(parts.node_energy.sum() + parts.fact_cell - parts.energy)), 1e-10)
        self.assertGreater(float(parts.fact_cell), 0.0)
        # e_v = 0 at the Thm 4 section (S6's T6 construction on the production operator)
        fx8 = Fixture(d=8)
        op4, labels, classes, _ = thm4_production(fx8)
        section = op4.initial_guess(classes[labels])
        sec = energy_terms(op4, *op4.layout.split(section))
        self.assertLessEqual(float(sec.node_energy.abs().max()), 1e-10)
        self.assertLessEqual(abs(float(sec.energy)), 1e-10)

    def test_T5_energy_at_a_model_solution(self) -> None:
        from src.analysis.mechanism_hooks import energy_at_solution

        _, data, model = _model()
        out = energy_at_solution(model, data)
        snap_op = None
        model.record_solve_state = True
        _logits(model, data)
        snap = model.last_solve_state
        model.record_solve_state = False
        reduced = snap.build(*snap.params)
        snap_op = reduced.operator
        y = snap.y_star
        x, _, z = reduced.layout.split(y)
        s = reduced.fact_states(y)
        inv_v, inv_e, inv_f = snap_op.inverse_sqrt_degree()
        full = torch.cat([x * inv_v.unsqueeze(-1), s * inv_e.unsqueeze(-1), z * inv_f.unsqueeze(-1)])
        quadratic = 0.5 * float((full * snap_op.apply(full)).sum())       # 1/2 Y^T L_hat Y
        self.assertLessEqual(abs(float(out["energy"]) - quadratic), 1e-10 * max(1.0, abs(quadratic)))
        self.assertLessEqual(abs(float(out["node_energy"].sum() + out["fact_cell"] - out["energy"])), 1e-10)
        self.assertAlmostEqual(float(out["energy_tilde"]), float(out["energy"] / x.square().sum()), delta=1e-12)
        self.assertEqual(out["node_energy"].numel(), data.num_nodes)


# ----------------------------------------------------------------------------- T6 (knockout)
class KnockoutTests(unittest.TestCase):
    def test_T6_full_batch(self) -> None:
        from src.analysis.mechanism_hooks import knockout

        cfg, data, model = _model()
        base = _logits(model, data)
        base_betas = [b.clone() for b in model.last_cell_betas]
        with knockout(model, ()):
            self.assertTrue(torch.equal(_logits(model, data), base))
        with knockout(model, range(data.num_nodes)):
            every = _logits(model, data)
            self.assertTrue(all(bool((b == 0).all()) for b in model.last_cell_betas))
        _, _, no_cells = _model(arm="no_path_cells", state=model.state_dict())
        torch.testing.assert_close(every, _logits(no_cells, data), atol=1e-12, rtol=0.0)
        groups = [g for g in data.face_groups if g.num_faces]
        node = int(groups[0].face_node_index[0, 1])                  # an intermediate of a cell
        with knockout(model, [node]):
            one = _logits(model, data)
            for group, beta, before in zip(groups, model.last_cell_betas, base_betas):
                through = (group.face_node_index == node).any(dim=1)
                self.assertTrue(bool((beta[through] == 0).all()))
                self.assertTrue(torch.equal(beta[~through], before[~through]))
        self.assertFalse(torch.equal(one, base))
        model.train()
        with knockout(model, [node]), self.assertRaises(ValueError):
            model(data)

    def test_T6_refused_in_training_and_tuning_configs(self) -> None:
        from src.configs.config import TrainConfig, reject_never_tuned_settings

        cfg = TrainConfig.from_dict({"evaluation": {"knockout_nodes": [3, 5]}})
        self.assertEqual(cfg.evaluation.knockout_nodes, (3, 5))
        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"evaluation": {"knockout_nodes": [3]}})
        with self.assertRaises(ValueError):
            reject_never_tuned_settings(None, {"knockout_nodes": {"values": [[1]]}})
        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"evaluation": {"knockout_nodes": [-1]}})
        reject_never_tuned_settings({"evaluation": {"knockout_nodes": []}})
        # a training config: the model refuses the first training forward
        from src.data.dataset import build_dataset
        from src.models.h2snn_model import build_model
        from tests.partO_fixtures import acm_like_hetero, patched_acm_loader

        train_cfg = TrainConfig.from_dict({"device": "cpu", "seed": 3,
                                           "data": {"dataset_name": "acm", "wisp_max_length": 3,
                                                    "wisp_walks_per_anchor": 5},
                                           "model": {**SMALL, "arm": "full"},
                                           "evaluation": {"knockout_nodes": [0]}})
        with patched_acm_loader(acm_like_hetero(seed=9)):
            data = build_dataset(train_cfg)
        model = build_model(train_cfg, data)
        model.train()
        with self.assertRaises(ValueError):
            model(data, epoch=0)

    def test_T6_mini_batch(self) -> None:
        from src.analysis.mechanism_hooks import knockout
        from tests.test_partMB_trainer import trainer_for

        with trainer_for(8, 16) as (cfg, graph, model, tr):
            model.eval()
            targets = graph.target_indices[graph.target_val_mask | graph.target_test_mask][:4]
            cells = tr.evaluation_cells(graph)
            _, batch = tr._batch(graph, targets, cells)
            with torch.no_grad():
                base = model(batch).logits
            base_betas = [b.clone() for b in model.last_cell_betas]
            groups = [g for g in batch.face_groups if g.num_faces]
            node = int(batch.global_node_id[groups[0].face_node_index[0, 1]])
            with knockout(model, [node]), torch.no_grad():
                out = model(batch).logits
                betas = model.last_cell_betas
            self.assertTrue(bool(torch.isfinite(out).all()))
            hit = 0
            for group, beta, before in zip(groups, betas, base_betas):
                through = (batch.global_node_id[group.face_node_index] == node).any(dim=1)
                hit += int(through.sum())
                self.assertTrue(bool((beta[through] == 0).all()))
                self.assertTrue(torch.equal(beta[~through], before[~through]))
            self.assertGreater(hit, 0)


# ----------------------------------------------------------------------------- T7, T8
class TransportAndCellTests(unittest.TestCase):
    def test_T7_transports_per_fact(self) -> None:
        from src.analysis.mechanism_hooks import transports_per_fact

        _, data, model = _model()
        facts = torch.arange(min(50, data.edge_index.size(1)))
        out = transports_per_fact(model, data, facts, chunk=16)
        with torch.no_grad():
            state = model.encode(data.x, data.node_type, data.edge_index, data.edge_type, data.face_groups,
                                 data.num_nodes).operator
        src, tgt = state.maps.src.select(facts), state.maps.tgt.select(facts)
        g = torch.Generator().manual_seed(2)
        x = torch.randn(facts.numel(), model.d, generator=g, dtype=F64)
        expected = src.apply_inverse(tgt.apply(x))                      # F_src^-1 F_tgt x
        torch.testing.assert_close((out["Q"] @ x.unsqueeze(-1)).squeeze(-1), expected, atol=1e-12, rtol=0.0)
        eye = torch.eye(model.d, dtype=F64).expand_as(out["Q"])
        torch.testing.assert_close(out["Q"].transpose(-2, -1) @ out["Q"], eye, atol=1e-12, rtol=0.0)
        torch.testing.assert_close(out["distance_to_identity"], (out["Q"] - eye).flatten(1).norm(dim=1),
                                   atol=1e-12, rtol=0.0)
        self.assertTrue(torch.equal(out["src"], data.edge_index[0, facts]))
        _, data2, s2 = _model(ablation={"fact_maps_identity": True})
        q2 = transports_per_fact(s2, data2, facts)["Q"]
        self.assertTrue(torch.equal(q2, torch.eye(s2.d, dtype=F64).expand_as(q2)))

    def test_T8_cells_per_target(self) -> None:
        from src.analysis.mechanism_hooks import cells_per_target
        from src.analysis.wisp_readouts import full_schema_keys

        _, data, model = _model()
        rows = cells_per_target(model, data)
        with torch.no_grad():
            state = model.encode(data.x, data.node_type, data.edge_index, data.edge_type, data.face_groups,
                                 data.num_nodes).operator
        betas = model.last_cell_betas
        groups = [g for g in data.face_groups if g.num_faces]
        self.assertEqual(len(rows), len(groups))
        targets = set(data.target_indices.tolist())
        for row, group, w, beta in zip(rows, groups, state.path_weights, betas):
            self.assertTrue(torch.equal(row["node_ids"], group.face_node_index))
            self.assertEqual(row["schema_keys"], full_schema_keys(group, data.node_type,
                                                                  data.fact_layer.self_inverse))
            torch.testing.assert_close(row["w"], w, atol=1e-12, rtol=0.0)
            torch.testing.assert_close(row["beta"], beta, atol=1e-12, rtol=0.0)
            self.assertTrue(all(int(a) in targets for a in row["anchor"]))


# ----------------------------------------------------------------------------- T9 (U4)
class CellListTests(unittest.TestCase):
    def test_T9_h2snn_phase(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import shlex

        import run_h2snn_wandb_tuning as launcher

        from scripts.tsubame.lines import runnable_lines
        from scripts.tsubame.make_cell_list import (H2SNN_CELLS, h2snn_final_seeds_list,
                                                    h2snn_tuning_list)

        out = Path("/OUT")
        tuning = runnable_lines("\n".join(h2snn_tuning_list(out)))
        seeds = runnable_lines("\n".join(h2snn_final_seeds_list(out)))
        self.assertEqual(len(tuning), 24)
        self.assertEqual(len(seeds), 24)
        order = [("node_classification", d) for d in ("acm", "dblp", "imdb", "freebase")]
        self.assertEqual(list(H2SNN_CELLS[:4]), order)
        self.assertEqual([t for t, _ in H2SNN_CELLS], ["node_classification"] * 12
                         + ["link_prediction"] * 8 + ["graph_classification"] * 4)
        for line, (task, dataset) in zip(tuning + seeds, list(H2SNN_CELLS) * 2):
            words = shlex.split(line)
            self.assertEqual(words[0], "scripts/run_h2snn_wandb_tuning.py")
            with mock.patch.object(sys, "argv", words):
                args = launcher.parse_args()                    # unknown flags would exit
            self.assertEqual(args.config, Path(f"configs/experiments/{dataset}_h2snn_v3b.yaml"))
            self.assertTrue((ROOT / args.config).exists(), args.config)
            self.assertEqual(args.exp_name, f"h2snn_{task}_{dataset}")
            self.assertEqual((args.num_runs, args.base_seed), (1, 0))
        for line in tuning:
            with mock.patch.object(sys, "argv", shlex.split(line)):
                args = launcher.parse_args()
            self.assertEqual((args.n_trials, args.wandb_project), (100, "h2snn_h2snn_t4"))
            self.assertIsNotNone(args.state_dir)
        for line in seeds:
            with mock.patch.object(sys, "argv", shlex.split(line)):
                self.assertEqual(launcher.parse_args().final_seeds, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
