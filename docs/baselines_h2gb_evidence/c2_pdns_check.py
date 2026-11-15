"""C2 real-data check (card baselines-h2gb): PDNS through `load_h2gb_dataset("pdns", root, seed)`; the split record per
seed (sizes, class counts, sha256 of each index set); seed 0's counts compared with Part M3b's H2GB-own load
(`docs/partM3b_evidence/b5_pdns_check.json`). Reads `data/NC/PDNS/raw` only; writes the JSON record below.
    python ../docs/baselines_h2gb_evidence/c2_pdns_check.py --root data --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "H2S-NN"))

from src.data.h2gb_nc import load_h2gb_dataset, split_record  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="data")
parser.add_argument("--seeds", type=int, nargs="+", default=[0])
args = parser.parse_args()

b5 = json.loads((HERE.parents[0] / "partM3b_evidence" / "b5_pdns_check.json").read_text())["labels"]
out = {"root": args.root, "seeds": {}}
for seed in args.seeds:
    start = time.time()
    data = load_h2gb_dataset("pdns", args.root, seed=seed)[0]
    record = split_record(data, "domain_node")
    record["nodes"] = {t: int(data[t].num_nodes) for t in data.node_types}
    record["edges"] = int(sum(data[e].edge_index.size(1) for e in data.edge_types))
    record["labelled"] = int((data["domain_node"].y != -1).sum())
    record["load_seconds"] = round(time.time() - start, 1)
    out["seeds"][str(seed)] = record
    print(f"seed {seed}: " + ", ".join(f"{s} {record[s]['size']} {record[s]['class_counts']} {record[s]['sha256'][:12]}"
                                       for s in ("train", "val", "test"))
          + f"; labelled {record['labelled']}, nodes {sum(record['nodes'].values())}, edges {record['edges']}", flush=True)
zero = out["seeds"].get("0")
if zero is not None:
    equal = all(zero[s]["size"] == b5["splits"][s]["size"] and zero[s]["class_counts"] == b5["splits"][s]["class_counts"]
                for s in ("train", "val", "test")) and zero["labelled"] == b5["labelled"]
    out["seed0_equals_b5"] = equal
    print(f"seed 0 sizes and class counts equal b5_pdns_check.json: {equal}")
(HERE / "c2_pdns_split_record.json").write_text(json.dumps(out, indent=1) + "\n")
