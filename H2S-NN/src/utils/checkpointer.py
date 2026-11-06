from __future__ import annotations

from pathlib import Path
from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class CheckpointLoadResult:
    metrics: dict[str, float]
    epoch: int
    metadata: dict[str, Any]


class Checkpointer:
    def __init__(
        self,
        exp_dir: Path,
        metric: str,
        mode: str,
    ) -> None:
        self.ckpt_dir = exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode = mode
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

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        tag: str,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "metadata": metadata or {},
        }
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(payload, path)
        return path

    def load(self, model: nn.Module, tag: str = "best") -> CheckpointLoadResult:
        """Restore weights; the strict `load_state_dict` rejects checkpoints of other models."""
        path = self.ckpt_dir / f"{tag}.pt"
        payload = torch.load(path, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        return CheckpointLoadResult(
            metrics=payload["metrics"],
            epoch=int(payload.get("epoch", -1)),
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
