from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import mock_open, patch


class _Data:
    target_val_mask = object()
    target_test_mask = object()

    def to(self, _device):
        return self


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, *_args, **_kwargs) -> None:
        return None

    def info(self, message, *_args) -> None:
        self.messages.append(str(message))

    def log_summary(self, *_args, **_kwargs) -> None:
        return None


class _Checkpointer:
    def __init__(self) -> None:
        self.decisions = iter((True, False, False, False))
        self.best_calls = 0

    def save(self, **_kwargs) -> None:
        return None

    def save_if_best(self, **_kwargs) -> bool:
        self.best_calls += 1
        return next(self.decisions)


def _config(output_dir: str):
    return SimpleNamespace(
        device="cpu",
        epochs=10,
        early_stopping_patience=2,
        checkpoint_metric="acc",
        log_every=100,
        exp_dir=Path(output_dir),
        # Read by Trainer.training_draw (D8 protocol); False keeps one fixed path draw.
        data=SimpleNamespace(resample_paths=False),
    )


def _metrics() -> dict[str, float]:
    return {
        "loss": 1.0,
        "acc": 0.5,
        "macro_f1": 0.5,
        "micro_f1": 0.5,
        "auroc": 0.5,
        "mrr": 0.5,
    }


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_node_trainer():
    stubs = {
        "src.data.structures": _module("src.data.structures", PreparedH2SData=object),
        "src.models.h2snn_model": _module("src.models.h2snn_model", H2SNNOutput=object),
    }
    sys.modules.pop("src.trainers.trainer", None)
    with patch.dict(sys.modules, stubs):
        return importlib.import_module("src.trainers.trainer").Trainer


def _load_link_trainer():
    torch_geometric = _module("torch_geometric")
    torch_geometric.__path__ = []  # type: ignore[attr-defined]
    torch_geometric_utils = _module(
        "torch_geometric.utils", negative_sampling=lambda **_kwargs: None
    )
    torch_geometric.utils = torch_geometric_utils  # type: ignore[attr-defined]
    stubs = {
        "torch_geometric": torch_geometric,
        "torch_geometric.utils": torch_geometric_utils,
        "src.data.link_pred_structures": _module(
            "src.data.link_pred_structures", PreparedLinkPredData=object
        ),
        "src.models.h2snn_model": _module(
            "src.models.h2snn_model", H2SNN=object, H2SNNOutput=object
        ),
    }
    sys.modules.pop("src.trainers.link_prediction_trainer", None)
    with patch.dict(sys.modules, stubs):
        return importlib.import_module(
            "src.trainers.link_prediction_trainer"
        ).LinkPredictionTrainer


class FullGraphEarlyStoppingTests(unittest.TestCase):
    def test_config_rejects_nonpositive_patience(self) -> None:
        from src.configs.config import TrainConfig

        for invalid in (0, -1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "early_stopping_patience must be positive"
            ):
                TrainConfig.from_dict({"early_stopping_patience": invalid})

    def test_node_trainer_honors_patience(self) -> None:
        import torch

        Trainer = _load_node_trainer()

        model = torch.nn.Linear(1, 1)
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            scheduler=None,
            criterion=torch.nn.CrossEntropyLoss(),
            cfg=_config("."),
            logger=_Logger(),
            checkpointer=_Checkpointer(),
        )
        trainer._train_epoch = lambda _data, _epoch: _metrics()  # type: ignore[method-assign]
        trainer._eval_epoch = (  # type: ignore[method-assign]
            lambda _data, _mask, _epoch=-1: _metrics()
        )
        seen_epochs: list[int] = []
        with patch.object(Path, "open", mock_open()):
            trainer.fit(
                _Data(),
                epoch_callback=lambda epoch, *_metrics_by_split: seen_epochs.append(epoch),
            )

        self.assertEqual(seen_epochs, [0, 1, 2])

    def test_link_trainer_honors_patience(self) -> None:
        import torch

        LinkPredictionTrainer = _load_link_trainer()

        model = torch.nn.Linear(1, 1)
        trainer = LinkPredictionTrainer(
            model=model,  # type: ignore[arg-type]
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            scheduler=None,
            cfg=_config("."),
            logger=_Logger(),
            checkpointer=_Checkpointer(),
        )
        trainer._train_epoch = lambda _data, _epoch: _metrics()  # type: ignore[method-assign]
        trainer._eval_epoch = (  # type: ignore[method-assign]
            lambda _data, _split, _epoch=-1: _metrics()
        )
        # Test metrics come from HGB's two test sets since Part N (N1, `_eval_test_sets`).
        trainer._eval_test_sets = lambda _data: {  # type: ignore[method-assign]
            f"{name}_{metric}": 0.5 for name in ("2hop", "random") for metric in ("auroc", "mrr")
        }
        seen_epochs: list[int] = []
        with patch.object(Path, "open", mock_open()):
            trainer.fit(
                _Data(),  # type: ignore[arg-type]
                epoch_callback=lambda epoch, *_metrics_by_split: seen_epochs.append(epoch),
            )

        self.assertEqual(seen_epochs, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
