"""Part M, M9: the training protocol shared by every method (spec v1.15 §6, J4).

AdamW everywhere; no learning-rate scheduler; no gradient clipping; epochs, patience and the
selection metric per task and source protocol (HGB NC 300/30 on validation Macro-F1; LP 40/40
on validation ROC-AUC; H2GB NC 500 without early stopping; GC as HetSheaf: patience 50 on
validation loss, final weights); H2S-NN's scalar group (theta_tau, b: 10x lr, no weight decay)
unchanged."""

from __future__ import annotations

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

import torch

from src.configs.config import PROTOCOL_BUDGETS, TrainConfig, protocol_budget

REPO = Path(__file__).resolve().parents[1]
OPTIMIZER_CALL = re.compile(r"\b(?:torch\.)?optim\.([A-Za-z]+)\(")


#: The one optimizer that is not a model's training optimizer (decision D-P9, author 2026-11-13;
#: spec v1.19 T2): H2GB's MetaPath2Vec pretraining of H2G-former's positional input encodings
#: uses H2GB's own SparseAdam (its sparse embeddings; AdamW rejects sparse gradients). AdamW
#: governs every model's training (spec v1.15 J4). (file, optimizer, enclosing function).
PRETRAINING_EXCEPTION = ("_h2gformer_encoders.py", "SparseAdam", "metapath2vec_embeddings")


def _optimizer_calls(path: Path) -> list[tuple[str, str]]:
    """(optimizer name, enclosing top-level function or '') of every `optim.X(` call."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = []
    for top in tree.body:
        owner = top.name if isinstance(top, (ast.FunctionDef, ast.ClassDef)) else ""
        for node in ast.walk(top):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                value = node.func.value
                if (isinstance(value, ast.Attribute) and value.attr == "optim") or (
                        isinstance(value, ast.Name) and value.id == "optim"):
                    if node.func.attr != "lr_scheduler":
                        calls.append((node.func.attr, owner))
    return calls


class OptimizerTests(unittest.TestCase):
    def test_every_adapter_builds_adamw(self) -> None:
        """Every model optimizer constructed by a registered adapter module or one of its
        helper modules is AdamW; the one named exception is H2GB's MetaPath2Vec pretraining
        (`PRETRAINING_EXCEPTION`, D-P9). Source-level check over the registry (the adapters'
        vendor imports are not exercised here)."""
        from src.baselines.registry import MODELS

        modules = {spec.module for spec in MODELS}
        files = {Path(importlib.util.find_spec(name).origin) for name in modules}
        files |= set((REPO / "src" / "baselines" / "adapters").glob("_*.py"))
        files |= {REPO / "src" / "baselines" / "training.py"}
        found: dict[str, list[tuple[str, str]]] = {}
        for path in sorted(files):
            calls = _optimizer_calls(path)
            # the regex view must agree with the AST view (no call form missed)
            self.assertEqual(sorted(n for n in OPTIMIZER_CALL.findall(path.read_text(
                encoding="utf-8")) if n != "lr_scheduler"), sorted(n for n, _ in calls),
                path.name)
            if calls:
                found[path.name] = calls
        self.assertGreaterEqual(len(found), 10)
        file, name, owner = PRETRAINING_EXCEPTION
        self.assertEqual(found[file].count((name, owner)), 1, "the named exception, once")
        found[file].remove((name, owner))
        offenders = {f: sorted({n for n, _ in calls}) for f, calls in found.items()
                     if any(n != "AdamW" for n, _ in calls)}
        self.assertEqual(offenders, {})

    def test_h2snn_default_optimizer_is_adamw_with_the_scalar_group(self) -> None:
        from src.builders import build_optimizer
        from src.models.h2snn_model import H2SNN, ModelDims
        from tests.test_factorized_maps import _random_pap_data
        from tests.test_v3b_model import _config

        data = _random_pap_data()
        config = _config("full")
        self.assertEqual(config.optim.name, "adamw")
        model = H2SNN(config, ModelDims.from_prepared_data(data))
        optimizer = build_optimizer(model, config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        groups = optimizer.param_groups
        scalar = [g for g in groups if any(p is model.theta_tau for p in g["params"])]
        self.assertEqual(len(scalar), 1)
        self.assertAlmostEqual(scalar[0]["lr"], 10 * config.optim.lr)
        self.assertEqual(scalar[0]["weight_decay"], 0.0)


class SchedulerAndClippingTests(unittest.TestCase):
    def test_no_scheduler_object_is_created(self) -> None:
        from src.builders import build_scheduler

        for dataset in ("acm", "lastfm", "mutag"):
            config = TrainConfig.from_dict({"data": {"dataset_name": dataset}})
            self.assertEqual((config.optim.scheduler, config.optim.grad_clip), ("none", 0.0))
            optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
            self.assertIsNone(build_scheduler(optimizer, config), dataset)

    def test_no_gradient_clipping_by_default(self) -> None:
        """A link-prediction training epoch with the default config never calls clipping."""
        from src.builders import build_optimizer
        from src.data.link_prediction_dataset import build_link_prediction_dataset
        from src.models.h2snn_model import build_link_prediction_model
        from src.trainers.link_prediction_trainer import LinkPredictionTrainer
        from tests.test_task_heads import _write_tiny_lastfm

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tiny_lastfm(root)
            config = TrainConfig.from_dict({"device": "cpu", "data": {
                "dataset_name": "lastfm", "root": str(root), "wisp_walks_per_anchor": 2,
                "wisp_max_length": 2}, "model": {"d": 8, "d_r": 4, "classifier_hidden_dim": 8,
                                                  "face_mlp_hidden_dim": 8,
                                                  "edge_mlp_hidden_dim": 8, "K": 1}})
            data = build_link_prediction_dataset(config)
            model = build_link_prediction_model(config, data)
            trainer = LinkPredictionTrainer(model, build_optimizer(model, config), None, config,
                                            logger=None, checkpointer=None)
            original = torch.nn.utils.clip_grad_norm_

            def refuse(*args, **kwargs):
                raise AssertionError("gradient clipping called")

            torch.nn.utils.clip_grad_norm_ = refuse
            try:
                trainer._train_epoch(data, epoch=0)
            finally:
                torch.nn.utils.clip_grad_norm_ = original


class BudgetTableTests(unittest.TestCase):
    def test_protocol_budgets(self) -> None:
        expected = {
            ("node_classification", "hgb"): (300, 30, "macro_f1", "max"),
            ("node_classification", "h2gb"): (500, None, "macro_f1", "max"),
            ("link_prediction", "hgb"): (40, 40, "auroc", "max"),
            ("graph_classification", "hetsheaf"): (200, 50, "loss", "min"),
        }
        self.assertEqual({k: (b.epochs, b.patience, b.monitor, b.mode)
                          for k, b in PROTOCOL_BUDGETS.items()}, expected)
        self.assertEqual(protocol_budget("node_classification", "acm").source, "hgb")
        self.assertEqual(protocol_budget("node_classification", "ogbn-mag").source, "h2gb")

    def test_baseline_budgets_come_from_the_table(self) -> None:
        from src.baselines.budget import default_budget

        self.assertEqual((default_budget("node_classification").epochs,
                          default_budget("node_classification").patience), (300, 30))
        self.assertEqual((default_budget("link_prediction").epochs,
                          default_budget("link_prediction").patience), (40, 40))
        self.assertEqual((default_budget("graph_classification").epochs,
                          default_budget("graph_classification").patience), (200, 50))
        mag = default_budget("node_classification", "mag-year")
        self.assertEqual((mag.epochs, mag.patience), (500, 500))  # no early stopping

    def test_h2snn_defaults_follow_the_protocol(self) -> None:
        cases = {"acm": (300, 30, "macro_f1"), "freebase_bb": (40, 40, "auroc"),
                 "lastfm": (40, 40, "auroc"), "mutag": (200, 50, "macro_f1")}
        for dataset, want in cases.items():
            data = {"dataset_name": dataset}
            if dataset == "freebase_bb":
                data.update(task_type="link_prediction", target_relation=["book", "rel_0", "book"])
            config = TrainConfig.from_dict({"data": data})
            self.assertEqual((config.epochs, config.early_stopping_patience,
                              config.checkpoint_metric), want, dataset)
        explicit = TrainConfig.from_dict({"data": {"dataset_name": "acm"}, "epochs": 3,
                                          "early_stopping_patience": 2})
        self.assertEqual((explicit.epochs, explicit.early_stopping_patience), (3, 2))

    def test_baseline_gc_stops_on_validation_loss_and_reports_final_weights(self) -> None:
        """Validation losses 0.9, 0.8, 0.85, 0.86 (then 0.1, never reached) with patience 2:
        the loop stops after epoch 3 and reports epoch 3 (final weights), not epoch 1."""
        from src.baselines.training import train_graph_classification

        val_losses = iter([0.9, 0.8, 0.85, 0.86, 0.1])
        model = torch.nn.Linear(1, 1)

        class Batch:
            def __init__(self, tag: str) -> None:
                self.tag, self.y = tag, torch.tensor([0])
                self.batch = torch.zeros(1, dtype=torch.long)

            def to(self, device):
                return self

        def embed_batch(batch: Batch) -> torch.Tensor:
            anchor = 0 * model.weight.sum()
            if batch.tag == "val":
                p = torch.tensor(-next(val_losses)).exp()  # CE of label 0 = the loss value
                return torch.stack([p.log(), (1 - p).log()]).view(1, 2) + anchor
            return torch.zeros(1, 2) + model.weight.sum()

        class Head(torch.nn.Module):
            def forward(self, x, batch):
                return x

        record = train_graph_classification(
            model=model, head=Head(), embed_batch=embed_batch,
            loaders=([Batch("train")], [Batch("val")], [Batch("test")]),
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.0), device=torch.device("cpu"),
            epochs=5, patience=2)
        self.assertEqual(record["final_epoch"], 3.0)
        self.assertAlmostEqual(record["val_loss"], 0.86, places=5)


if __name__ == "__main__":
    unittest.main()
