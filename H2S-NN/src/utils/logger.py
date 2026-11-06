from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def flatten_metric_record(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for split, metrics in (
        ("train", train_metrics),
        ("val", val_metrics),
        ("test", test_metrics),
    ):
        for key, value in metrics.items():
            flattened[f"{split}/{key}"] = float(value)
    return flattened


class ExperimentLogger:
    def __init__(
        self,
        exp_dir: Path,
        *,
        use_wandb: bool = False,
        wandb_project: str = "h2snn",
        wandb_entity: str = "",
        wandb_mode: str = "online",
        wandb_group: str = "",
        wandb_tags: tuple[str, ...] = (),
        wandb_name: str | None = None,
        wandb_log_artifacts: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.exp_dir = exp_dir
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.exp_dir / "metrics.jsonl"
        self.logger = logging.getLogger(f"h2snn.{self.exp_dir}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._wandb: Any | None = None
        self._wandb_run: Any | None = None
        self._wandb_log_artifacts = wandb_log_artifacts

        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            file_handler = logging.FileHandler(self.exp_dir / "run.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

        if use_wandb:
            self._init_wandb(
                project=wandb_project,
                entity=wandb_entity,
                mode=wandb_mode,
                group=wandb_group,
                tags=wandb_tags,
                name=wandb_name,
                config=config,
            )

    def _init_wandb(
        self,
        *,
        project: str,
        entity: str,
        mode: str,
        group: str,
        tags: tuple[str, ...],
        name: str | None,
        config: dict[str, Any] | None,
    ) -> None:
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError("use_wandb=True requires the 'wandb' package.") from exc

        init_kwargs: dict[str, Any] = {
            "project": project,
            "name": name,
            "mode": mode,
            "config": config or {},
            "dir": str(self.exp_dir),
        }
        if entity:
            init_kwargs["entity"] = entity
        if group:
            init_kwargs["group"] = group
        if tags:
            init_kwargs["tags"] = list(tags)

        self._wandb = wandb
        self._wandb_run = wandb.init(**init_kwargs)

    def log(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
    ) -> None:
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if self._wandb_run is not None:
            self._wandb.log(flatten_metric_record(train_metrics, val_metrics, test_metrics), step=epoch)

        summary = {
            "train_loss": train_metrics.get("loss", 0.0),
            "train_acc": train_metrics.get("acc", 0.0),
            "train_auroc": train_metrics.get("auroc", 0.0),
            "train_macro_f1": train_metrics.get("macro_f1", 0.0),
            "train_micro_f1": train_metrics.get("micro_f1", 0.0),
            "val_loss": val_metrics.get("loss", 0.0),
            "val_acc": val_metrics.get("acc", 0.0),
            "val_auroc": val_metrics.get("auroc", 0.0),
            "val_macro_f1": val_metrics.get("macro_f1", 0.0),
            "val_micro_f1": val_metrics.get("micro_f1", 0.0),
            "test_acc": test_metrics.get("acc", 0.0),
            "test_auroc": test_metrics.get("auroc", 0.0),
            "test_macro_f1": test_metrics.get("macro_f1", 0.0),
            "test_micro_f1": test_metrics.get("micro_f1", 0.0),
        }
        rendered = " | ".join(f"{key}: {value:.4f}" for key, value in summary.items())
        self.logger.info("Epoch %04d | %s", epoch, rendered)

    def log_summary(self, metrics: dict[str, Any], prefix: str = "best") -> None:
        if self._wandb_run is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)):
                self._wandb_run.summary[f"{prefix}/{key}"] = value

    def info(self, message: str, *args: object) -> None:
        self.logger.info(message, *args)

    def _log_wandb_artifacts(self) -> None:
        if self._wandb_run is None or not self._wandb_log_artifacts:
            return
        artifact = self._wandb.Artifact(f"{self.exp_dir.name}-local-files", type="experiment")
        for relative in (
            "config.yaml",
            "metrics.jsonl",
            "run.log",
            "best_test_metrics.json",
            "checkpoints/best.pt",
            "checkpoints/last.pt",
        ):
            path = self.exp_dir / relative
            if path.exists():
                artifact.add_file(str(path), name=relative)
        self._wandb_run.log_artifact(artifact)

    def close(self) -> None:
        if self._wandb_run is not None:
            self._log_wandb_artifacts()
            self._wandb.finish()
            self._wandb_run = None
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
