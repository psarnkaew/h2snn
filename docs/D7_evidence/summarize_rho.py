"""Print a compact table of inventory_rho_output.json."""
import json
import sys
from pathlib import Path

data = json.loads(Path(__file__).with_name("inventory_rho_output.json").read_text(encoding="utf-8"))
for r in data:
    if "error" in r:
        print(r["dataset"], "ERROR", r["error"])
        continue
    print(f"{r['dataset']}: facts before {r['facts_before']} after {r['facts_after']}; "
          f"self-inverse {r['self_inverse']}; mirror pairs {len(r['mirror_pairs'])}; "
          f"untied {r['untied']}; near {r['near_partners']}")
    for x in r["relations"]:
        print(f"    {x['relation']}: stored {x['stored_edges']} dup {x['duplicate_edges']} "
              f"loops {x['self_loops']} recip {x['reciprocal_pairs_within']} "
              f"rho {x['rho']} unmatched {x['unmatched_edges']}")
    for k in ("num_graphs", "pipeline_vs_pyg_label_mismatch"):
        if k in r:
            print("   ", k, r[k])
    if "pyg_label_analysis" in r:
        p = r["pyg_label_analysis"]
        print(f"    [pyg edge_attr labels] self-inverse {p['self_inverse']} untied {p['untied']} "
              f"facts after {p['facts_after']}")
