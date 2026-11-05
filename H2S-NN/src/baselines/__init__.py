"""Baseline reproduction for the H2S-NN comparison tables.

Adapters wrap the baseline repositories cloned under ``vendor/`` and drive them
through the same data bundles, sweep engine, and metric reporting as H2S-NN itself, so a
difference in a results table is a difference in the model rather than in the protocol.

Entry points: ``scripts/run_baseline_tuning.py`` and ``scripts/run_baseline_published.py``.
"""

from src.baselines.protocol import (
    GRAPH_CLASSIFICATION,
    LINK_PREDICTION,
    NODE_CLASSIFICATION,
    TASKS,
    BaselineAdapter,
    BaselineParams,
    UnsupportedCell,
)

__all__ = [
    "GRAPH_CLASSIFICATION",
    "LINK_PREDICTION",
    "NODE_CLASSIFICATION",
    "TASKS",
    "BaselineAdapter",
    "BaselineParams",
    "UnsupportedCell",
]
