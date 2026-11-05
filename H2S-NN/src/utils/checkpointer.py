from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.configs.config import AMPSS_PROTOCOL_VERSION


@dataclass(frozen=True, slots=True)
class CheckpointLoadResult:
    metrics: dict[str, float]
    epoch: int
    operator_ramp: float
    metadata: dict[str, Any]


class Checkpointer:
    def __init__(
        self,
        exp_dir: Path,
        metric: str,
        mode: str,
        protocol_version: str = AMPSS_PROTOCOL_VERSION,
        selector_manifest: dict[str, Any] | None = None,
        require_selector_manifest: bool = False,
    ) -> None:
        self.ckpt_dir = exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode = mode
        self.protocol_version = protocol_version
        self.selector_manifest = selector_manifest
        self.require_selector_manifest = require_selector_manifest
        if require_selector_manifest and selector_manifest is None:
            raise ValueError("Auto AMPSS checkpoints require a selector manifest.")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'.")
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def _is_better(self, new_score: float) -> bool:
        if self.mode == "min":
            return new_score < self.best_score
        return new_score > self.best_score

    def _extract_score(self, metrics: dict[str, float]) -> float:
        if self.metric in metrics:
            return metrics[self.metric]
        shortened = self.metric.removeprefix("val_")
        if shortened in metrics:
            return metrics[shortened]
        raise KeyError(f"Metric '{self.metric}' not found in checkpoint payload.")

    @staticmethod
    def _model_operator_schedule(model: nn.Module) -> dict[str, int]:
        model_config = getattr(model, "config", None)
        discovery = getattr(model_config, "discovery", None)
        return {
            "warmup_epochs": int(getattr(discovery, "warmup_epochs", 0)),
            "operator_anneal_epochs": int(
                getattr(discovery, "operator_anneal_epochs", 1)
            ),
        }

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        tag: str,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        schedule = self._model_operator_schedule(model)
        payload = {
            "protocol_version": self.protocol_version,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "metadata": metadata or {},
            "operator_schedule": {
                **schedule,
                "saved_operator_ramp": float(
                    model._operator_ramp(epoch)  # type: ignore[attr-defined]
                    if hasattr(model, "_operator_ramp")
                    else 1.0
                ),
            },
            "selector_manifest": self.selector_manifest,
        }
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(payload, path)
        return path

    def load(self, model: nn.Module, tag: str = "best") -> CheckpointLoadResult:
        path = self.ckpt_dir / f"{tag}.pt"
        payload = torch.load(path, map_location="cpu")
        checkpoint_version = payload.get("protocol_version")
        if checkpoint_version != self.protocol_version:
            raise ValueError(
                "Checkpoint AMPSS protocol is incompatible: "
                f"found {checkpoint_version!r}, expected {self.protocol_version!r}. "
                f"Start a fresh {self.protocol_version} run."
            )
        checkpoint_manifest = payload.get("selector_manifest")
        if self.require_selector_manifest and checkpoint_manifest is None:
            raise ValueError("Checkpoint is missing its required AMPSS selector manifest.")
        if checkpoint_manifest != self.selector_manifest:
            raise ValueError(
                "Checkpoint AMPSS manifest does not match the current ordered schemas "
                "or selector/materialized support hashes."
            )
        schedule = payload.get("operator_schedule", {})
        expected_schedule = self._model_operator_schedule(model)
        saved_schedule = {
            "warmup_epochs": int(schedule.get("warmup_epochs", -1)),
            "operator_anneal_epochs": int(
                schedule.get("operator_anneal_epochs", -1)
            ),
        }
        if saved_schedule != expected_schedule:
            raise ValueError(
                "Checkpoint AMPSS operator schedule does not match the current "
                f"configuration: found {saved_schedule}, expected {expected_schedule}."
            )
        # All semantic compatibility checks happen before model mutation.
        model.load_state_dict(payload["state_dict"])
        return CheckpointLoadResult(
            metrics=payload["metrics"],
            epoch=int(payload.get("epoch", -1)),
            operator_ramp=float(schedule.get("saved_operator_ramp", 1.0)),
            metadata=dict(payload.get("metadata", {})),
        )

    def save_if_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        score = self._extract_score(metrics)
        if self._is_better(score):
            self.best_score = score
            self.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                tag="best",
                metrics=metrics,
                metadata=metadata,
            )
            return True
        return False
