"""Card 6 H2c-1: (1) the clique condition on ieee-cis-g's M2 paths (card3 edges per transaction and the other
fields), (2) the real-data check: every NC M2 path already finished with `mde_chunked` (records/) that satisfies the
same condition, closed form vs the recorded value within 1e-5 relative. One dataset per process (H2GB's package
registers its modules once per process). Prints JSON lines; exit 1 on a mismatch.
    python docs/heterophily_evidence/h2c/h2c1_check.py <records dir> <out json> survey|<dataset>   (from H2S-NN/)"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("stats", ROOT / "scripts" / "heterophily_stats.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
het = stats.het

records, out, which = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
RTOL = 1e-5


def load(dataset):
    raw, target, _ = stats.load_nc(dataset, Path(stats.REPO_DATA_ROOT))
    sym, _ = het.symmetrise_facts(raw)
    paths, _ = het.hetero2net_m2(dataset, sym.edge_types, target)
    return sym, target, paths


def survey(sym, target, path):
    (t, m), _ = path
    first = torch.cat([sym[e].edge_index for e in het.step_relations(sym.edge_types, t, m)], dim=1)
    n_mid = int(sym[m].num_nodes)
    pairs = het._unique_pairs(first, n_mid)
    per = torch.bincount(pairs[0], minlength=int(sym[t].num_nodes))
    mid = het.clique_gp_mid(sym, path)
    return {"metapath": het.metapath_name(path), "max_mid_edges_per_target": int(per.max()),
            "targets_with_0": int((per == 0).sum()), "targets_with_1": int((per == 1).sum()),
            "targets_with_2_or_more": int((per >= 2).sum()), "clique_condition": mid is not None}, mid


result = {"which": which, "rows": [], "mismatch": False}
if which == "survey":
    sym, target, paths = load("ieee-cis-g")
    for path in paths:
        row, mid = survey(sym, target, path)
        if mid is not None:
            sizes = torch.bincount(mid[mid >= 0])
            row.update(cliques=int((sizes > 0).sum()), gp_edges_with_self_pairs=int((sizes * sizes).sum()),
                       self_pairs=int((mid >= 0).sum()))
        result["rows"].append(row)
        print(json.dumps(row), flush=True)
else:
    rec = json.loads((records / f"nc__{which}__mde.json").read_text())
    rows = [r for r in rec.get("per_metapath") or [] if str(r.get("code_path", "")).startswith("mde_chunked")]
    if not rows:
        result["note"] = "no meta-path finished with mde_chunked in this record"
        out.write_text(json.dumps(result, indent=1))
        print("no mde_chunked path; mismatch False")
        sys.exit(0)
    sym, target, paths = load(which)
    x = sym[target].x.float()
    by_name = {het.metapath_name(p): p for p in paths}
    for r in rows:
        row, mid = survey(sym, target, by_name[r["metapath"]])
        row.update(dataset=which, record_job=rec.get("job"), recorded=r["value"], recorded_edges=r["edges"])
        if mid is not None:
            value, edges = het.clique_mde(mid, x)
            rel = abs(value - r["value"]) / abs(r["value"])
            row.update(closed_form=value, edges=edges, rel_diff=rel, edges_equal=edges == r["edges"],
                       ok=rel <= RTOL and edges == r["edges"])
            result["mismatch"] |= not row["ok"]
        result["rows"].append(row)
        print(json.dumps(row), flush=True)
out.write_text(json.dumps(result, indent=1))
print("mismatch", result["mismatch"])
sys.exit(1 if result["mismatch"] else 0)
