"""Status of every baseline cell of the paper tables (card 8 C0, amendment 1; decisions log 2026-11-20).

* `filled`: the cell holds a published number (`docs/published_number_register.md`, transcribed into
  `configs/baselines/paper_filled_cells.tsv`); never run.
* `N/A`: the registry marks the cell unsupported (`registry.unsupported_reason`); never run.
* `probe`: a 1-trial dry run at the smallest grid point on the H100 decides N/A (card 8 C3, C5): MAGNN on the H2GB
  node-classification datasets and on the large LP tasks.
* `published`: a published configuration file exists (`configs/baselines/<model>/<task>/<dataset>.yaml`): 5 seeds at
  that configuration (`scripts/run_baseline_published.py`).
* `tune`: everything else: 100 Bayesian trials on seed 0, then seeds 0-4.
"""

from __future__ import annotations

import csv
from functools import lru_cache

from src.baselines.protocol import LINK_PREDICTION, NODE_CLASSIFICATION
from src.baselines.published import CONFIG_ROOT, published_params_path

FILLED_CELLS_FILE = CONFIG_ROOT / "paper_filled_cells.tsv"

#: Card 8 C3: MAGNN's H2GB NC cells and large LP cells are probed (range rule 1; N/A if the smallest point does not
#: fit an H100), not tuned.
PROBE_CELLS: frozenset[tuple[str, str, str]] = frozenset({
    *((NODE_CLASSIFICATION, d, "magnn") for d in ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem",
                                                  "ieee-cis-g", "h-pokec", "pdns")),
    *((LINK_PREDICTION, d, "magnn") for d in ("ogbn-mag_pp", "h-pokec_uu")),
})

STATUSES = ("filled", "N/A", "probe", "published", "tune")


@lru_cache(maxsize=1)
def filled_cells() -> frozenset[tuple[str, str, str]]:
    """(task, dataset, model) of every cell with a published number."""
    with FILLED_CELLS_FILE.open(encoding="utf-8") as handle:
        return frozenset((row["task"], row["dataset"], row["method"])
                         for row in csv.DictReader(handle, delimiter="\t"))


def cell_status(model: str, task: str, dataset: str) -> str:
    """One of `STATUSES` (filled wins over N/A: a GC cell with HetSheaf's number is filled, not N/A)."""
    from src.baselines.registry import unsupported_reason

    if (task, dataset, model) in filled_cells():
        return "filled"
    if unsupported_reason(model, task, dataset):
        return "N/A"
    if (task, dataset, model) in PROBE_CELLS:
        return "probe"
    if published_params_path(model, task, dataset).exists():
        return "published"
    return "tune"


__all__ = ["FILLED_CELLS_FILE", "PROBE_CELLS", "STATUSES", "cell_status", "filled_cells"]
