"""Card 1c F8: the registry state of every H2GB NC and large-LP baseline cell (runnable or its reason)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "H2S-NN"))
from src.baselines.registry import H2GB_NC, LARGE_LP, iter_cells  # noqa: E402

for cell in iter_cells():
    if cell["dataset"] in H2GB_NC + LARGE_LP:
        print(f'{cell["task"]}|{cell["model"]}|{cell["dataset"]}|{cell["reason"] or "RUNNABLE"}')
