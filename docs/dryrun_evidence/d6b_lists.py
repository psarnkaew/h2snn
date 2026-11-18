"""Card 2 D6b: run lists for d6_array.sbatch (text only; the registry of the merged head, no data).
    python ../docs/dryrun_evidence/d6b_lists.py <evidence dir>        (from H2S-NN/)

d6b_list_max.txt              step 3: the 10 cells fixed by card 1d H3b (e26d659), at D2's largest point
d6b_list_central_<ds>.txt     step 4 (reduced, author): central point of every runnable baseline on H2GB NC
                              ogbn-mag, pdns, h-pokec (one train epoch + one full validation pass)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.baselines.registry import iter_cells  # noqa: E402

ev = Path(sys.argv[1])
H3B = ["han|node_classification|imdb", "han|link_prediction|imdb_ma", "h2sgnn|link_prediction|imdb_ma",
       "chebnetii|node_classification|ogbn-mag", "chebnetii|node_classification|mag-year",
       "chebnetii|node_classification|oag-cs", "chebnetii|node_classification|oag-eng",
       "chebnetii|node_classification|oag-chem", "sehgnn|node_classification|ieee-cis-g",
       "hinormer|node_classification|freebase"]
(ev / "d6b_list_max.txt").write_text("".join(f"RUN {c} D6b-max --point max\n" for c in H3B))
for ds in ("ogbn-mag", "pdns", "h-pokec"):
    cells = sorted(f"{r['model']}|{r['task']}|{r['dataset']}" for r in iter_cells()
                   if r["runnable"] and r["task"] == "node_classification" and r["dataset"] == ds)
    (ev / f"d6b_list_central_{ds}.txt").write_text("".join(f"RUN {c} D6b-central --point central\n" for c in cells))
    print(ds, len(cells))
