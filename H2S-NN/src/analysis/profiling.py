from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn.functional as F

from src.data.structures import PreparedH2SData
from src.models.h2snn_model import H2SNN


def count_parameters_by_module(model: H2SNN) -> dict[str, int]:
    breakdown = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in model.named_children()
    }
    breakdown["total"] = sum(p.numel() for p in model.parameters())
    return breakdown


def _forward_only(model: H2SNN, data: PreparedH2SData) -> None:
    with torch.no_grad():
        model(data)


def _forward_backward(model: H2SNN, data: PreparedH2SData) -> None:
    model.zero_grad(set_to_none=True)
    output = model(data)
    logits = output.logits[data.target_train_mask]
    labels = data.target_y[data.target_train_mask]
    if data.is_multilabel:
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    else:
        loss = F.cross_entropy(logits, labels)
    loss.backward()


def _train_step(model: H2SNN, data: PreparedH2SData, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    output = model(data)
    logits = output.logits[data.target_train_mask]
    labels = data.target_y[data.target_train_mask]
    if data.is_multilabel:
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    else:
        loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()


def _time_block(fn: Callable[[], None], num_iters: int, warmup: int, device: str) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(num_iters):
        start = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    mean = sum(timings) / len(timings)
    variance = sum((t - mean) ** 2 for t in timings) / len(timings)
    return {"mean_seconds": mean, "std_seconds": variance**0.5}


def profile_time_and_memory(
    model: H2SNN,
    data: PreparedH2SData,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_iters: int = 50,
    warmup: int = 10,
) -> dict[str, dict[str, float]]:
    """Full-batch/transductive model - each timed call is one whole-graph pass,
    matching what `Trainer._train_epoch`/`_eval_epoch` do per epoch."""
    results: dict[str, dict[str, float]] = {}

    model.eval()
    results["forward_only"] = _time_block(lambda: _forward_only(model, data), num_iters, warmup, device)

    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results["forward_backward"] = _time_block(lambda: _forward_backward(model, data), num_iters, warmup, device)
    if device == "cuda":
        results["forward_backward"]["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results["train_step"] = _time_block(lambda: _train_step(model, data, optimizer), num_iters, warmup, device)
    if device == "cuda":
        results["train_step"]["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)

    return results
