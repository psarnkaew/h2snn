"""Card 7 amendments 9-10 evidence (structure only, nothing trained): eta_v (`mechanism_common.eta`, per-path mean
over M_2) on the test targets of ACM, DBLP, IMDB and Freebase, and the eta groups (G0 = eta_v 0, G1-G3 = tertiles of
eta_v > 0): tertile edges, nodes per group, distinct values, largest tie, test targets excluded (no labelled
neighbour on any M_2 path). Test targets = the HGB store's `test_mask` as `src/data/dataset.py` reads it
(`_first_mask_column`). Writes eta_groups_table.json next to this script; prints the table."""
import json
import sys
from pathlib import Path

import numpy as np

root = Path("/mnt/data/peeradon/h2snn-model/H2S-NN")
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scripts/analysis"))
import mechanism_common as mc  # noqa: E402
from src.data.dataset import _first_mask_column  # noqa: E402

out = {}
for ds in mc.ABLATION_DATASETS:
    data, target, ref = mc.reference_graph(root / "data", ds)
    test = _first_mask_column(data[target].test_mask).numpy()
    values = mc.eta(root / "data", ds)
    e = values["eta"][test]
    paths = values["paths"][test]
    edges = mc.group_edges(e)
    g = mc.group_of(e, edges)
    defined = e[~np.isnan(e)]
    vals, counts = np.unique(np.round(defined, 12), return_counts=True)
    names = [ref.metapath_name(p) for p in ref.h2gb_metapaths(data, target)]
    row = {"M2": names, "test_targets": int(test.sum()), "excluded": int((g == -1).sum()),
           "tertile_edges": [round(x, 4) for x in edges],
           "groups": {n: int((g == i).sum()) for i, n in enumerate(mc.GROUPS)},
           "distinct_values": int(len(vals)), "largest_tie": int(counts.max()) if counts.size else 0,
           "largest_tie_value": float(vals[counts.argmax()]) if counts.size else None,
           "eta_mean": float(defined.mean()) if defined.size else None,
           "paths_counted": {int(k): int((paths == k).sum()) for k in np.unique(paths)}}
    out[ds] = row
    print(f"== {ds}: target {target}, M_2 ({len(names)}) = {names}")
    print(f"  test targets {row['test_targets']}, excluded {row['excluded']}, eta mean {row['eta_mean']:.4f}, "
          f"tertile edges t1/t2 {row['tertile_edges']}, groups {row['groups']}, distinct values "
          f"{row['distinct_values']}, largest tie {row['largest_tie']} at {row['largest_tie_value']}, "
          f"test targets per number of paths counted {row['paths_counted']}", flush=True)
(Path(__file__).resolve().parent / "eta_groups_table.json").write_text(json.dumps(out, indent=1) + "\n")
