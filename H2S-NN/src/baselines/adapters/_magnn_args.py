"""MAGNN's run arguments from a parameter set (card 8 C1/C3).

HGB's MAGNN runners (`HGB/NC/benchmark/methods/MAGNN/run_<Dataset>.py`) take their settings from module constants
(`dropout_rate`, `lr`, `weight_decay`, `run_Freebase.py:17-19`) and argparse (`run_Freebase.py:209-227`: feats-type,
hidden-dim, num-heads, attn-vec-dim, rnn-type, epoch, patience, batch-size, samples), and train with
`torch.optim.Adam` (`run_Freebase.py:83`). `run_arguments` maps our parameter keys to those names (the names of
`run_model_DBLP`'s arguments and constants); the adapter reads every setting from it.
"""

from __future__ import annotations

from typing import Any

#: our key -> (MAGNN's name, default when the key is absent: HGB `run_Freebase.py` / MAGNN's argparse defaults)
KEYS: dict[str, tuple[str, Any]] = {
    "feats_type": ("feats_type", 2),
    "hidden_dim": ("hidden_dim", 64),
    "num_heads": ("num_heads", 8),
    "attn_vec_dim": ("attn_vec_dim", 128),
    "rnn_type": ("rnn_type", "RotatE0"),
    "epochs": ("num_epochs", 100),
    "early_stopping": ("patience", 5),
    "batch_size": ("batch_size", 8),
    "samples": ("neighbor_samples", 100),
    "dropout": ("dropout_rate", 0.5),
    "lr": ("lr", 0.005),
    "weight_decay": ("weight_decay", 0.001),
    # published only (`run_Freebase.py:83`: Adam); absent -> the shared protocol's AdamW (tuned cells)
    "optimizer": ("optimizer", None),
}


def run_arguments(values: dict[str, Any], *, tuned_budget: dict[str, int] | None = None) -> dict[str, Any]:
    """MAGNN's arguments (its own names) for `values`; `tuned_budget` (epochs, patience) replaces the source's budget
    for a tuned run (the shared budget, src/baselines/budget.py)."""
    out = {name: values.get(key, default) for key, (name, default) in KEYS.items()}
    if tuned_budget is not None:
        out["num_epochs"] = int(tuned_budget["epochs"])
        out["patience"] = int(tuned_budget["patience"])
    return out


__all__ = ["KEYS", "run_arguments"]
