"""Card 10 A3: build the fit-check probe lists of the real phase lists and count the trials each probe runs (the
grid size of `--grid_point max`; 1 for a published or H2S-NN probe). Lists only; nothing is submitted.

    python ../docs/card10_evidence/probe_counts.py <scratch dir>      (from H2S-NN/, in a CPU job)
"""

import collections
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "scripts"))

import run_baseline_tuning as rbt  # noqa: E402
from scripts.slurm import fit_check  # noqa: E402
from scripts.slurm.grid_points import grid_size  # noqa: E402
from scripts.tsubame.lines import parse_line  # noqa: E402
from src.baselines.campaign import Cell  # noqa: E402

scratch = Path(sys.argv[1])
for phase in ("h2snn", "baselines-early", "baselines-1", "baselines-2", "published"):
    listing = scratch / f"{phase}.txt"
    subprocess.run([sys.executable, "scripts/tsubame/make_cell_list.py", "--phase", phase, "--out", str(listing)],
                   check=True, capture_output=True)
    probes = fit_check.build(listing, scratch / f"fit_{phase}")
    trials, by_model = 0, collections.Counter()
    for probe in probes:
        line = parse_line(probe)
        n = 1
        if line.launcher == "baseline":
            sweep = rbt.cell_sweep_config(Cell(model=line.model, task=line.task, dataset=line.dataset),
                                          monitor="val_x", n_trials=1, epochs=1, grid_point="max")
            n = grid_size(sweep)
        trials += n
        by_model[line.model] += n
    print(f"{phase}: probes {len(probes)}, trials {trials}; trials by method {dict(sorted(by_model.items()))}")
