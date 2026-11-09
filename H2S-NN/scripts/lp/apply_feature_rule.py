"""One feature rule for every link-prediction task (spec v1.10 F17; task card Part J, J6.1).

A node feature is removed in the LP task files when it is (a) computed from the target relation
(the whole feature block of that node type) or (b) an attribute of the other endpoint
(individual columns). Nothing else is removed; nothing is rebuilt (supersedes F12's rebuild,
`rebuild_lp_features.py`). A removed block becomes an empty fourth column, as the HGB release
writes a featureless type, so every model learns an embedding for it.

Removals follow from the data (J5.1, J5 review, `info.dat`):
- acm_pa: author block (OR of its papers' features, J5.1);
- dblp_ap: author block (a function of its paper set, J5.1);
- imdb_ma: actor block (mean of its movies' token columns, J5.1); movie columns 3, 4, 7, 12
  (actor_1/actor_3/cast_total/actor_2_facebook_likes: attributes of the movie's actors);
- freebase_bb: nothing (no features); recorded so the loader guard (F20) covers it.

The script always starts from the release's node.dat (md5 checked against `split_meta.json`),
so it is idempotent and is run after `build_derived_lp.py`. It writes `<task>/node.dat`,
`<task>/feature_rule.json` (removals, widths, node.dat md5: read by the loader guard) and
updates `split_meta.json`'s output md5 of node.dat. The release is never written.

Run from H2S-NN/: python scripts/lp/apply_feature_rule.py [--tasks acm_pa dblp_ap imdb_ma freebase_bb]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: F17 removals per derived task: node types whose whole block goes, and per type the columns.
FEATURE_RULE: dict[str, dict] = {
    "acm_pa": {"remove_types": [1], "remove_columns": {}},
    "dblp_ap": {"remove_types": [0], "remove_columns": {}},
    "imdb_ma": {"remove_types": [2], "remove_columns": {0: [3, 4, 7, 12]}},
    "freebase_bb": {"remove_types": [], "remove_columns": {}},
}
#: The release's names of the removed columns (checked against info.dat when present).
COLUMN_NAMES = {("imdb_ma", 0): {3: "actor_1_facebook_likes", 4: "actor_3_facebook_likes",
                                 7: "cast_total_facebook_likes", 12: "actor_2_facebook_likes"}}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def type_names(info: dict) -> dict[int, str]:
    raw = info.get("node.dat", {}).get("node type", {})
    return {int(k): str(v) for k, v in raw.items()}


def apply_rule(task: str, task_dir: Path, rule: dict | None = None) -> dict:
    """Write the rule-applied node.dat and feature_rule.json of one task; return the record."""
    rule = FEATURE_RULE[task] if rule is None else rule
    meta_path = task_dir / "split_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    release = Path(meta["source"])
    if md5(release / "node.dat") != meta["source_md5"]["node.dat"]:
        raise ValueError(f"{release / 'node.dat'} differs from the release recorded at build time")
    try:
        info = json.loads((release / "info.dat").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        info = {}  # Freebase's info.dat is plain text; it has no features to name
    names = type_names(info)
    remove_types = {int(t) for t in rule["remove_types"]}
    remove_cols = {int(t): sorted(int(c) for c in cols)
                   for t, cols in rule["remove_columns"].items()}
    attributes = info.get("node.dat", {}).get("Attributes", {})
    for (name, t), cols in COLUMN_NAMES.items():
        if name == task and attributes:
            for c, col_name in cols.items():
                if attributes.get(str(c)) != col_name:
                    raise ValueError(f"{task}: info.dat attribute {c} is "
                                     f"{attributes.get(str(c))!r}, not {col_name!r}")
    out, widths = [], {}
    for line in (release / "node.dat").read_bytes().splitlines(keepends=True):
        body = line.rstrip(b"\r\n")
        ending = line[len(body):]
        parts = body.split(b"\t")
        t = int(parts[2])
        if t in remove_types:
            parts = parts[:3] + [b""]
        elif t in remove_cols and len(parts) == 4 and parts[3]:
            values = parts[3].split(b",")
            drop = set(remove_cols[t])
            if max(drop) >= len(values):
                raise ValueError(f"{task}: type {t} has {len(values)} columns")
            parts[3] = b",".join(v for j, v in enumerate(values) if j not in drop)
        width = len(parts[3].split(b",")) if len(parts) == 4 and parts[3] else 0
        if widths.setdefault(t, width) != width:
            raise ValueError(f"{task}: type {t} has rows of widths {widths[t]} and {width}")
        out.append(b"\t".join(parts) + ending)
    (task_dir / "node.dat").write_bytes(b"".join(out))
    record = {
        "task": task, "rule": "spec v1.10 F17 (remove, never rebuild)",
        "release": str(release), "release_node_dat_md5": md5(release / "node.dat"),
        "removed_types": {str(t): names.get(t, f"type_{t}") for t in sorted(remove_types)},
        "removed_columns": {str(t): {str(c): COLUMN_NAMES.get((task, t), {}).get(c, "")
                                     for c in cols} for t, cols in remove_cols.items()},
        "widths": {str(t): w for t, w in sorted(widths.items())},
        "node_dat_md5": md5(task_dir / "node.dat"),
    }
    (task_dir / "feature_rule.json").write_text(json.dumps(record, indent=1) + "\n",
                                                encoding="utf-8")
    stale = task_dir / "feature_rebuild.json"  # J5's rebuild record, superseded by F17
    if stale.exists():
        stale.unlink()
    meta["output_md5"]["node.dat"] = record["node_dat_md5"]
    meta.pop("node_dat_features", None)
    meta["node_dat_feature_rule"] = ("spec v1.10 F17 applied by scripts/lp/apply_feature_rule.py "
                                     "(J6.1); see feature_rule.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lp-root", type=Path, default=PROJECT_ROOT / "data" / "lp_derived")
    parser.add_argument("--tasks", nargs="+", default=list(FEATURE_RULE),
                        choices=list(FEATURE_RULE))
    args = parser.parse_args()
    for task in args.tasks:
        print(json.dumps(apply_rule(task, args.lp_root / task)), flush=True)


if __name__ == "__main__":
    main()
