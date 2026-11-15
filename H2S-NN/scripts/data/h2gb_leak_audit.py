"""Part M3b B4 data check: venue-leak audit of the OAG loaders (F18 style), and the label-source rows of the
other H2GB NC datasets.

    python scripts/data/h2gb_leak_audit.py --root data --out <file.json> oag-cs [oag-eng oag-chem]

For each key: the inventory and fingerprint probe of `src/data/leak_audit.py` on the graph H2GB trains on
(`load_h2gb_nc`), and the relation list H2S-NN's `build_dataset` and the baseline bundles flatten from it
(`flatten_edges`; the same edges, so the same audit). Reads only; writes only `--out`. Run in a SLURM job.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.h2gb_nc import H2GB_NC_SPECS, load_h2gb_nc  # noqa: E402
from src.data.hetero_flatten import flatten_edges  # noqa: E402
from src.data.leak_audit import CARRIER_SHARE, audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", nargs="+")
    parser.add_argument("--root", default=str(REPO / "data"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out: dict = {"carrier_share": CARRIER_SHARE, "datasets": {}}
    for key in args.keys:
        start = time.perf_counter()
        data = load_h2gb_nc(key, Path(args.root))
        target = H2GB_NC_SPECS[key].target_node_type
        result = audit(data, target, seed=0)
        # what H2S-NN's build_dataset and the bundles flatten (relation names only; same edge_index tensors)
        node_slices, offset = {}, 0
        for node_type in data.node_types:
            node_slices[str(node_type)] = (offset, offset + int(data[node_type].num_nodes))
            offset += int(data[node_type].num_nodes)
        _, _, relation_names, _ = flatten_edges(data, node_slices)
        out["datasets"][key] = {
            "target": target,
            "passed": result["passed"],
            "carriers": result["carriers"],
            "inventory": result["inventory"],
            "h2snn_flattened_relations": list(relation_names),
            "probes": [{"relation": "|".join(p.relation), "hubs_with_2plus": p.hubs, "incidences": p.neighbours,
                        "covered_targets": p.covered_targets, "majority_share": round(p.majority_share, 4),
                        "control_share": round(p.control_share, 4), "carries_label": p.carries_label}
                       for p in result["probes"]],
            "seconds": round(time.perf_counter() - start, 1),
        }
        print(key, "passed" if result["passed"] else f"FAILED {result['carriers']}", flush=True)
    out["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
