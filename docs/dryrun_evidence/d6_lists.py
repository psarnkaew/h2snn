"""Card 2 D6: the run lists for d6_array.sbatch (text only; reads the registry and the search spaces of the merged
head, no data).   python ../docs/dryrun_evidence/d6_lists.py <evidence dir>        (from H2S-NN/)

d6_list_main.txt   (30G host, 24 GB GPU): D6.5 H2SGNN LP freebase_bb with CUDA_LAUNCH_BLOCKING=1; the D6.3 ladders
                   of the cells still OOM after D2's step-down; the D6.4 central-point runs except H-Pokec.
d6_list_hpokec.txt (60G host, 24 GB GPU): D6.6 H2S-NN H-Pokec at D2's largest point; the D6.4 central-point runs on
                   H-Pokec.
Ladder rule (D6.3, decision 1): from D2's largest point, the axis of D2's step-down one grid step at a time (D2's
first step-down value is skipped: it was run and was still OOM); at its minimum, the next memory axis in the order
listed here, the previous axes kept at their minimum.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.baselines.registry import get_adapter, iter_cells  # noqa: E402

ev = Path(sys.argv[1])
H2GB = ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns")
H2SNN_GRID = {"model.d": [16, 32, 64], "data.wisp_max_length": [2, 3, 4, 5], "data.wisp_walks_per_anchor": [8, 16, 32, 64]}

#: cell -> memory axes in ladder order (first = D2's step-down axis)
LADDER_AXES = {
    "h2snn|link_prediction|freebase_bb": ["data.wisp_walks_per_anchor", "model.d", "data.wisp_max_length"],
    "h2snn|node_classification|freebase": ["data.wisp_walks_per_anchor", "model.d", "data.wisp_max_length"],
    "nsd|node_classification|acm": ["d", "hidden_channels", "layers"],
    "nsd|node_classification|freebase": ["d", "hidden_channels", "layers"],
    "nsd|link_prediction|acm_pa": ["d", "hidden_channels", "layers"],
    "nsd|link_prediction|freebase_bb": ["d", "hidden_channels", "layers"],
    "hinormer|node_classification|acm": ["len_seq", "hidden_dim", "num_heads", "num_layers", "num_gnns"],
    "hinormer|node_classification|dblp": ["len_seq", "hidden_dim", "num_heads", "num_layers", "num_gnns"],
    "hinormer|node_classification|imdb": ["len_seq", "hidden_dim", "num_heads", "num_layers", "num_gnns"],
    "gat|link_prediction|freebase_bb": ["num_heads", "hidden_dim", "num_layers"],
    "pshgcn|link_prediction|freebase_bb": ["K", "hidden", "emb_dim", "embed_dim"],
}


def grid_of(cell: str) -> dict[str, list]:
    model, task, dataset = cell.split("|")
    if model == "h2snn":
        return H2SNN_GRID
    space = get_adapter(model).search_space(task, dataset)
    return {k: list(s["values"]) for k, s in space.items() if "values" in s}


def ladder(cell: str) -> list[str]:
    grids = grid_of(cell)
    fixed: dict[str, object] = {}
    steps = []
    for n, axis in enumerate(LADDER_AXES[cell]):
        values = sorted(grids[axis], reverse=True)  # max first
        lower = values[2:] if n == 0 else values[1:]  # D2 ran the first step of the first axis (still OOM)
        if n == 0:
            fixed[axis] = values[1]
        for v in lower:
            fixed[axis] = v
            steps.append(",".join(f"{k}={x}" for k, x in fixed.items()))
    return steps


main, hpokec = [], []
main.append("CLB h2sgnn|link_prediction|freebase_bb D6.5-CLB --point max")
for cell in LADDER_AXES:
    main.append(f"LADDER {cell} D6.3 " + " ".join(ladder(cell)))
# D6.4: H2S-NN on H2GB NC and the 4 GC cells; baselines runnable on H2GB NC (none ran in D2: their first status) and
# the baseline GC cells OK in D2
gc_ok = {r["cell"] for r in csv.DictReader(open(ev / "cells.csv"))
         if r["attempt"] == "24GB" and r["status"] == "OK" and r["task"] == "graph_classification" and r["model"] != "h2snn"}
central = [f"h2snn|node_classification|{d}" for d in H2GB] + \
          [f"h2snn|graph_classification|{d}" for d in ("mutag", "proteins", "enzymes", "nci1")]
central += [f"{r['model']}|{r['task']}|{r['dataset']}" for r in iter_cells()
            if r["runnable"] and r["task"] == "node_classification" and r["dataset"] in H2GB]
central += sorted(gc_ok)
# H2GB NC baseline cells last: on the merged v3b they stop in bundles._base_config (H3b) before any model code
central.sort(key=lambda c: (not c.startswith("h2snn|") and c.split("|")[2] in H2GB))
for cell in central:
    (hpokec if cell.endswith("|h-pokec") else main).append(f"RUN {cell} D6.4-central --point central")
hpokec.insert(0, "RUN h2snn|node_classification|h-pokec D6.6-60G --point max")
(ev / "d6_list_main.txt").write_text("\n".join(main) + "\n")
(ev / "d6_list_hpokec.txt").write_text("\n".join(hpokec) + "\n")
print(json.dumps({"main": len(main), "hpokec": len(hpokec), "central": len(central)}))
