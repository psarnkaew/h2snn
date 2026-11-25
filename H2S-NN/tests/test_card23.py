"""Card 23 (docs/H2S-NN_card23_card.md): the regularisation pack for the overfitting HGB NC cells.

(a) C1 input-feature dropout; (b) C4 averaged evaluation draws; (c) C5 EMA of the weights; (d) C6 smoothed
checkpoint selection; (e) the tuning space and the b0f13fb fingerprints; (f) C4-C6 refused outside the
full-batch NC trainer. Off by default, every option is bit-identical to b0f13fb (tests/test_v3b_golden.py and
the mb0 compare, run with this file). Synthetic data only.
"""

from __future__ import annotations

import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.partO_fixtures import acm_like_hetero, patched_acm_loader

ROOT = Path(__file__).resolve().parents[1]

# Card 23 fingerprint check (docs/card23_evidence/fp_b0f13fb.jsonl, job 19003): the base-config and sweep
# fingerprints of configs/experiments/acm_h2snn_v3b.yaml on b0f13fb43bcc77bdf15f73a2caa026140179e441.
B0F13FB_ACM_BASE_FINGERPRINT = "888ca7ac9fd8f8a738e3bf0012bc33968a7fa141dd4b3c8f63b7168a4e218aed"
B0F13FB_ACM_SWEEP_FINGERPRINT = "c84be876d4d53c131beee74c931e4e9b5c4c8b98a9112e309c5f33d16d2a094e"


def _raw(**overrides) -> dict:
    """The mb0 reference configuration (docs/partMB_evidence/mb0_reference.py) on the ACM-shaped fixture."""
    raw = {
        "device": "cpu", "seed": 3, "epochs": 3, "early_stopping_patience": 100,
        "checkpoint_metric": "macro_f1", "log_every": 100,
        "data": {"dataset_name": "acm", "wisp_max_length": 3, "wisp_walks_per_anchor": 5},
        "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                  "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full"},
        "ablation": {"coupling": "symmetric"},
    }
    for section, values in overrides.items():
        if isinstance(values, dict):
            raw.setdefault(section, {}).update(values)
        else:
            raw[section] = values
    return raw


def _build(raw: dict, output_dir: str):
    from src.configs.config import TrainConfig
    from src.data.dataset import build_dataset
    from src.models.h2snn_model import build_model

    raw = {**raw, "output_dir": output_dir, "exp_name": "card23"}
    cfg = TrainConfig.from_dict(raw)
    hetero = acm_like_hetero(seed=9)
    del hetero["term"].x  # one featureless type, as in mb0
    with patched_acm_loader(hetero):
        data = build_dataset(cfg)
    torch.manual_seed(0)
    model = build_model(cfg, data)
    return cfg, data, model


def _trainer(cfg, model):
    from src.trainers.trainer import build_trainer
    from src.utils.checkpointer import Checkpointer
    from src.utils.logger import ExperimentLogger

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    logger = ExperimentLogger(cfg.exp_dir)
    checkpointer = Checkpointer(cfg.exp_dir, metric="macro_f1", mode="max")
    return build_trainer(model=model, optimizer=optimizer, scheduler=None, cfg=cfg, logger=logger,
                         checkpointer=checkpointer), optimizer, logger


class InputDropoutTests(unittest.TestCase):
    """(a) C1."""

    def _encoder(self, p: float):
        from src.models.node_encoder import TypeSpecificNodeEncoder

        torch.manual_seed(0)
        return TypeSpecificNodeEncoder(input_dim=6, hidden_dim=8, num_node_types=2, dropout=0.0,
                                       type_input_dims=(6, 0), type_num_nodes=(2000, 50),
                                       type_offsets=(0, 2000), input_dropout=p)

    def _inputs(self):
        generator = torch.Generator().manual_seed(1)
        x = torch.randn(2050, 6, generator=generator)
        x[2000:] = 0.0
        node_type = torch.cat([torch.zeros(2000, dtype=torch.long), torch.ones(50, dtype=torch.long)])
        return x, node_type

    def test_half_of_the_raw_entries_are_zeroed_in_training(self) -> None:
        encoder = self._encoder(0.5)
        x, node_type = self._inputs()
        seen = []
        encoder.encoders[0][0].register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach()))
        encoder.train()
        torch.manual_seed(5)
        encoder(x, node_type)
        dropped = seen[0]
        zero_fraction = float((dropped == 0).float().mean())
        self.assertGreater(zero_fraction, 0.45)
        self.assertLess(zero_fraction, 0.55)
        kept = dropped != 0
        torch.testing.assert_close(dropped[kept], 2.0 * x[:2000][kept])  # inverted dropout scale 1/(1-p)

    def test_featureless_rows_untouched_and_eval_is_a_no_op(self) -> None:
        encoder = self._encoder(0.5)
        x, node_type = self._inputs()
        encoder.train()
        torch.manual_seed(5)
        with_dropout = encoder(x, node_type)
        encoder.input_dropout = 0.0
        without = encoder(x, node_type)
        self.assertTrue(torch.equal(with_dropout[2000:], without[2000:]))
        self.assertFalse(torch.equal(with_dropout[:2000], without[:2000]))
        encoder.eval()
        reference = encoder(x, node_type)
        encoder.input_dropout = 0.5
        self.assertTrue(torch.equal(encoder(x, node_type), reference))

    def test_p0_draws_no_random_numbers(self) -> None:
        encoder = self._encoder(0.0)
        x, node_type = self._inputs()
        encoder.train()
        state = torch.get_rng_state()
        encoder(x, node_type)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_model_config_reaches_the_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, model = _build(_raw(model={"input_dropout": 0.3}), tmp)
            self.assertEqual(model.node_encoder.input_dropout, 0.3)
            _, _, model = _build(_raw(), tmp)
            self.assertEqual(model.node_encoder.input_dropout, 0.0)


class EvalDrawTests(unittest.TestCase):
    """(b) C4."""

    def test_k1_builds_no_draw_and_uses_the_single_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, model = _build(_raw(epochs=1), tmp)
            trainer, _, logger = _trainer(cfg, model)
            trainer._eval_epoch_averaged = None  # would raise if called
            try:
                trainer.fit(data)
            finally:
                logger.close()
            self.assertEqual(trainer._extra_eval_draws, ())
            # the evaluation logits are the model's on the evaluation draw
            prepared = data.to("cpu")
            model.eval()
            with torch.no_grad():
                model._warm_start = None
                direct = model(prepared, epoch=-1).logits
                model._warm_start = None
                via_trainer = trainer._compute_metrics(model(prepared, epoch=-1), prepared,
                                                       prepared.target_val_mask)[1]
            from src.utils.misc import compute_classification_metrics
            mask = prepared.target_val_mask
            self.assertEqual(via_trainer["macro_f1"],
                             compute_classification_metrics(direct[mask], prepared.target_y[mask])["macro_f1"])

    def test_k3_probabilities_are_the_mean_of_three_single_draw_forwards(self) -> None:
        from src.configs.config import EVAL_DRAW_SEED_STRIDE
        from src.data.dataset import build_node_classification_paths

        self.assertEqual(EVAL_DRAW_SEED_STRIDE, 1_000_003)
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, model = _build(_raw(data={"eval_draws": 3}), tmp)
            trainer, _, logger = _trainer(cfg, model)
            logger.close()
            prepared = data.to("cpu")
            trainer.build_eval_draws(data, prepared)
            draws = (prepared, *trainer._extra_eval_draws)
            self.assertEqual(len(draws), 3)
            for k in (1, 2):  # draw k has seed seed + 1_000_003 k
                expected = build_node_classification_paths(
                    config=cfg, fact_layer=data.fact_layer, node_slices=data.node_slices,
                    node_type=data.node_type, node_type_names=data.node_type_names,
                    edge_index=data.edge_index, edge_type=data.edge_type,
                    relation_tuples=data.relation_tuples, seed=cfg.seed + 1_000_003 * k)
                for got, want in zip(draws[k].face_groups, expected):
                    self.assertTrue(torch.equal(got.face_node_index, want.face_node_index))
            self.assertFalse(all(torch.equal(a.face_node_index, b.face_node_index)
                                 for a, b in zip(draws[0].face_groups, draws[1].face_groups)))

            averaged = trainer.averaged_eval_logits(prepared)  # first call: every draw starts cold
            model.eval()
            singles = []
            with torch.no_grad():
                for draw in draws:
                    model._warm_start = None
                    singles.append(torch.softmax(model(draw, epoch=-1).logits, dim=-1))
            mean = torch.stack(singles).mean(dim=0)
            torch.testing.assert_close(averaged.exp(), mean, rtol=1e-5, atol=1e-7)
            self.assertFalse(torch.allclose(singles[0], mean))

            val, test = trainer._eval_epoch_averaged(prepared)
            mask = prepared.target_val_mask
            nll = -mean[mask].gather(1, prepared.target_y[mask].view(-1, 1)).log().mean()
            self.assertAlmostEqual(val["loss"], float(nll), places=5)
            self.assertIn("macro_f1", test)


class EmaTests(unittest.TestCase):
    """(c) C5."""

    def _fit(self, raw: dict, tmp: str):
        cfg, data, model = _build(raw, tmp)
        trainer, optimizer, logger = _trainer(cfg, model)
        steps = [[p.detach().clone() for p in model.parameters()]]
        step = optimizer.step

        def recorded_step(*args, **kwargs):
            out = step(*args, **kwargs)
            steps.append([p.detach().clone() for p in model.parameters()])
            return out

        optimizer.step = recorded_step  # type: ignore[method-assign]
        seen_in_eval: list[torch.Tensor] = []
        first = next(model.parameters())
        model.register_forward_pre_hook(
            lambda module, _args: None if module.training else seen_in_eval.append(first.detach().clone()))
        metrics, emas = [], []

        def callback(epoch, train, val, test):
            metrics.append((train, val, test))
            emas.append(None if trainer._ema is None else [e.clone() for e in trainer._ema])

        torch.manual_seed(7)
        try:
            trainer.fit(data, epoch_callback=callback)
        finally:
            logger.close()
        return trainer, model, steps, metrics, emas, seen_in_eval, cfg

    def test_beta0_is_today_step_by_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            today = self._fit(_raw(), tmp + "/a")
            beta0 = self._fit(_raw(optim={"ema_decay": 0.0}), tmp + "/b")
        self.assertIsNone(beta0[0]._ema)
        for got, want in zip(beta0[2], today[2], strict=True):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, want)))
        self.assertEqual(beta0[3], today[3])

    def test_beta09_closed_form_and_evaluation_reads_ema(self) -> None:
        beta = 0.9
        with tempfile.TemporaryDirectory() as tmp:
            trainer, model, steps, _, emas, seen, cfg = self._fit(_raw(optim={"ema_decay": beta}), tmp)
            self.assertEqual(len(steps), 4)  # theta_0 and three steps
            for index in range(len(steps[0])):
                t0, t1, t2, t3 = (s[index] for s in steps)
                closed = beta ** 3 * t0 + (1 - beta) * (beta ** 2 * t1 + beta * t2 + t3)
                torch.testing.assert_close(trainer._ema[index], closed, rtol=1e-5, atol=1e-6)
            # training continues on theta: the model holds theta_3 after fit
            self.assertTrue(all(torch.equal(p, s) for p, s in zip(model.parameters(), steps[-1])))
            # every evaluation forward of epoch e read theta_ema after step e + 1
            self.assertEqual(len(seen), 6)  # validation and test, three epochs
            for epoch in range(3):
                for forward in seen[2 * epoch: 2 * epoch + 2]:
                    self.assertTrue(torch.equal(forward, emas[epoch][0]))
            self.assertFalse(torch.equal(seen[-1], steps[-1][0]))
            # the best checkpoint holds theta_ema of the selected epoch
            best = torch.load(cfg.exp_dir / "checkpoints" / "best.pt", weights_only=False)
            first_name = next(name for name, _ in model.named_parameters())
            self.assertTrue(torch.equal(best["state_dict"][first_name], emas[trainer._best_epoch][0]))


class SmoothedSelectionTests(unittest.TestCase):
    """(d) C6, on a synthetic validation sequence."""

    SEQUENCE = (0.50, 0.80, 0.40, 0.70, 0.72, 0.71, 0.30)

    def _select(self, w: int) -> int:
        from src.trainers.trainer import Trainer
        from src.utils.checkpointer import Checkpointer

        class _Logger:
            def log(self, *_a, **_k): return None
            def info(self, *_a, **_k): return None
            def log_summary(self, *_a, **_k): return None

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(device="cpu", epochs=len(self.SEQUENCE), early_stopping_patience=100,
                                  checkpoint_metric="macro_f1", log_every=100, exp_dir=Path(tmp),
                                  data=SimpleNamespace(resample_paths=False), checkpoint_smoothing=w)
            model = torch.nn.Linear(1, 1)
            trainer = Trainer(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                              scheduler=None, criterion=torch.nn.CrossEntropyLoss(), cfg=cfg,
                              logger=_Logger(), checkpointer=Checkpointer(Path(tmp), "macro_f1", "max"))
            values = iter(v for v in self.SEQUENCE for _ in ("val", "test"))
            trainer._train_epoch = lambda _d, _e: {"loss": 0.0}  # type: ignore[method-assign]
            trainer._eval_epoch = lambda _d, _m, _e=-1: {"loss": 0.0, "macro_f1": next(values)}  # type: ignore
            trainer.fit(SimpleNamespace(to=lambda _device: SimpleNamespace(target_val_mask=None,
                                                                            target_test_mask=None)))
            return trainer._best_epoch

    def test_w1_is_today_and_w3_is_the_trailing_mean_argmax(self) -> None:
        self.assertEqual(self._select(1), max(range(7), key=lambda e: (self.SEQUENCE[e], -e)))
        self.assertEqual(self._select(1), 1)
        trailing = [sum(self.SEQUENCE[max(0, e - 2): e + 1]) / len(self.SEQUENCE[max(0, e - 2): e + 1])
                    for e in range(7)]
        self.assertEqual(self._select(3), max(range(7), key=lambda e: (trailing[e], -e)))
        self.assertEqual(self._select(3), 5)


class TuningSpaceTests(unittest.TestCase):
    """(e) the space of the re-tuned cells and the fingerprints."""

    def _sweep(self, name: str, **pack):
        from scripts.run_h2snn_wandb_tuning import _build_sweep_for_run, load_raw_config
        from src.configs.config import TrainConfig

        raw = load_raw_config(ROOT / "configs" / "experiments" / f"{name}_h2snn_v3b.yaml")
        if pack:
            raw.setdefault("data", {})["eval_draws"] = pack["eval_draws"]
            raw["checkpoint_smoothing"] = pack["checkpoint_smoothing"]
        cfg = TrainConfig.from_dict(raw)
        sweep = _build_sweep_for_run(args=argparse.Namespace(d_values=None, grid_point=None), base_cfg=cfg,
                                     base_exp_name="fp", monitor="val_macro_f1", run_index=0)
        return cfg, sweep

    def test_acm_pack_space(self) -> None:
        _, sweep = self._sweep("acm", eval_draws=8, checkpoint_smoothing=5)
        space = sweep["parameters"]
        self.assertEqual((space["input_dropout"]["min"], space["input_dropout"]["max"]), (0.0, 0.9))
        self.assertEqual(space["input_dropout"]["distribution"], "uniform")
        self.assertEqual((space["encoder_dropout"]["min"], space["encoder_dropout"]["max"]), (0.0, 0.99))
        self.assertEqual(space["lr"], {"distribution": "log_uniform_values", "min": 1e-4, "max": 3e-2})
        self.assertIn(128, space["d"]["values"])
        self.assertEqual(space["ema_decay"]["values"], [0.0, 0.9, 0.99, 0.999])
        self.assertEqual(space["classifier_dropout"]["max"], 0.9)  # unchanged
        from src.baselines import search_grids
        self.assertEqual((search_grids.DROPOUT["max"], search_grids.LR["max"]), (0.9, 1e-2))  # shared unchanged

    def test_featureless_pack_spaces_have_no_input_dropout(self) -> None:
        for name in ("freebase", "lastfm"):
            with self.subTest(name=name):
                _, sweep = self._sweep(name, eval_draws=8, checkpoint_smoothing=5)
                self.assertNotIn("input_dropout", sweep["parameters"])
                self.assertNotIn("encoder_dropout", sweep["parameters"])

    def test_acm_without_the_keys_has_the_b0f13fb_fingerprints(self) -> None:
        from scripts.run_h2snn_wandb_tuning import base_config_fingerprint
        from src.utils.wandb_campaign import canonical_fingerprint

        cfg, sweep = self._sweep("acm")
        self.assertNotIn("input_dropout", sweep["parameters"])
        self.assertEqual(sweep["parameters"]["lr"]["max"], 1e-2)
        self.assertEqual(base_config_fingerprint(cfg), B0F13FB_ACM_BASE_FINGERPRINT)
        self.assertEqual(canonical_fingerprint(sweep), B0F13FB_ACM_SWEEP_FINGERPRINT)
        # the keys written at their off values: the same fingerprints; the pack: new ones
        cfg_off, sweep_off = self._sweep("acm", eval_draws=1, checkpoint_smoothing=1)
        self.assertEqual(base_config_fingerprint(cfg_off), B0F13FB_ACM_BASE_FINGERPRINT)
        self.assertEqual(canonical_fingerprint(sweep_off), B0F13FB_ACM_SWEEP_FINGERPRINT)
        cfg_on, sweep_on = self._sweep("acm", eval_draws=8, checkpoint_smoothing=5)
        self.assertNotEqual(base_config_fingerprint(cfg_on), B0F13FB_ACM_BASE_FINGERPRINT)
        self.assertNotEqual(canonical_fingerprint(sweep_on), B0F13FB_ACM_SWEEP_FINGERPRINT)

    def test_trial_parameters_reach_the_config(self) -> None:
        from scripts.run_h2snn_wandb_tuning import apply_sweep_overrides

        payload = apply_sweep_overrides({"data": {"dataset_name": "acm"}},
                                        {"input_dropout": 0.4, "ema_decay": 0.99})
        self.assertEqual((payload["model"]["input_dropout"], payload["optim"]["ema_decay"]), (0.4, 0.99))


class OtherTrainersRefuseTests(unittest.TestCase):
    """(f) C4-C6 belong to the full-batch NC trainer."""

    def test_minibatch_and_lp_configs_with_eval_draws_raise(self) -> None:
        from scripts.run_h2snn_wandb_tuning import load_raw_config
        from src.configs.config import TrainConfig
        from src.trainers.graph_classification_trainer import GraphClassificationTrainer
        from src.trainers.large_lp_trainer import LargeLinkTrainer
        from src.trainers.link_prediction_trainer import LinkPredictionTrainer
        from src.trainers.minibatch_trainer import MiniBatchNodeTrainer

        def build(name, trainer, **overrides):
            raw = load_raw_config(ROOT / "configs" / "experiments" / f"{name}_h2snn_v3b.yaml")
            for key, value in overrides.items():
                if key == "eval_draws":
                    raw.setdefault("data", {})[key] = value
                elif key == "ema_decay":
                    raw.setdefault("optim", {})[key] = value
                else:
                    raw[key] = value
            cfg = TrainConfig.from_dict(raw)
            if trainer is MiniBatchNodeTrainer or trainer is LargeLinkTrainer:
                return trainer(None, None, cfg, None, None)
            return trainer(model=None, optimizer=None, scheduler=None, cfg=cfg, logger=None,
                           checkpointer=None)

        cases = (("ogbn-mag", MiniBatchNodeTrainer), ("acm_pp", LinkPredictionTrainer),
                 ("mutag", GraphClassificationTrainer))
        for name, trainer in cases:
            for key, value in (("eval_draws", 2), ("ema_decay", 0.9), ("checkpoint_smoothing", 3)):
                with self.subTest(name=name, key=key), self.assertRaisesRegex(ValueError, "card 23"):
                    build(name, trainer, **{key: value})
        cfg = TrainConfig.from_dict({"data": {"eval_draws": 2}})
        with self.assertRaisesRegex(ValueError, "LargeLinkTrainer"):
            LargeLinkTrainer(None, None, cfg, None, None)
        # off: no refusal from the guard (the stand-in model fails later or not at all)
        from src.configs.config import reject_full_batch_nc_options
        reject_full_batch_nc_options(TrainConfig.from_dict({}), "MiniBatchNodeTrainer")

    def test_invalid_values_raise(self) -> None:
        from src.configs.config import TrainConfig

        for raw in ({"data": {"eval_draws": 0}}, {"data": {"eval_draws": True}}, {"optim": {"ema_decay": 1.0}},
                    {"model": {"input_dropout": 1.0}}, {"checkpoint_smoothing": 0}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                TrainConfig.from_dict(copy.deepcopy(raw))


if __name__ == "__main__":
    unittest.main()
