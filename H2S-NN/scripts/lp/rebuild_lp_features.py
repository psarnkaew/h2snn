"""Rebuild or drop link-prediction endpoint features that encode the target relation
(spec v1.9 F12; task card Part J, J5.3). LP task files only: `data/lp_derived/<task>/node.dat`.

SUPERSEDED by spec v1.10 F17 (J6.1, `apply_feature_rule.py`: remove, never rebuild). Kept for
the J5 record; not part of the pipeline. Running it would make the loader refuse the task (F20).

Input: the J5.1 identification (`lp_feature_rules.py --json-out`). Per affected type:
- an exact rule exists -> its features are the rule applied to the node's **training** facts of
  the target relation (link.dat minus link.dat.valid; test facts are not in link.dat); a node with
  no training fact gets the zero vector, as the rule gives;
- no exact rule -> the type's features are dropped (empty fourth column, as the release writes a
  featureless type), so every model learns an embedding for it.
Where several exact rules exist, the first in the fixed order V1, V1p, V2:*, V3*; or, max, sum,
mean is used, and the others are applied to the training facts too and compared (recorded).

The release is read, never written; the task's other rows stay byte-identical. Idempotent: it
always starts from the release's node.dat, so it is run after `build_derived_lp.py` (which copies
the release's node.dat) and again after any rebuild of the split. It records what was done in
`<task>/feature_rebuild.json` and updates `split_meta.json`'s output md5 of node.dat.

Run from H2S-NN/: python scripts/lp/rebuild_lp_features.py --rules <j51 json> --tasks acm_pa ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_feature_rules as fr  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ORDER = ("V1", "V1p", "V2", "V3")
AGG_ORDER = ("or", "max", "sum", "mean")


def rule_order(name: str) -> tuple[int, str, int]:
    source, agg = name.split("/")
    family = next(i for i, p in enumerate(SOURCE_ORDER) if source.startswith(p))
    return family, source, AGG_ORDER.index(agg)


def value_format(texts: list[str]):
    """The release's number format for a type: integers ("1") or Python repr ("1.0")."""
    integer = all("." not in t and "e" not in t.lower() for t in texts)
    if integer:
        return lambda v: str(int(v)) if float(v).is_integer() else repr(float(v)), "integer"
    return lambda v: repr(float(v)), "repr"


def training_neighbourhood(spec: fr.TaskSpec, rel_types, x: fr.NodeType,
                           y: fr.NodeType) -> tuple[sp.csr_matrix, int]:
    links = fr.read_links(spec.folder / "link.dat")
    valid = fr.read_links(spec.folder / "link.dat.valid").get(spec.target, np.zeros((0, 2), int))
    held = {tuple(p) for p in valid.tolist()}
    pairs = np.array([p for p in links[spec.target].tolist() if tuple(p) not in held],
                     dtype=np.int64).reshape(-1, 2)
    return fr.neighbourhood(pairs, rel_types[spec.target], x, y), len({tuple(p) for p in
                                                                       pairs.tolist()})


def rebuild_task(spec: fr.TaskSpec, identification: dict) -> dict:
    types = fr.read_nodes(spec.release)
    links = fr.read_links(spec.release / "link.dat")
    rel_types = fr.relation_types(spec.release / "info.dat")
    t_start, t_end = rel_types[spec.target]
    release_lines = (spec.release / "node.dat").read_bytes().splitlines(keepends=True)
    task_md5_before = fr.md5(spec.folder / "node.dat")
    new_rows: dict[int, bytes] = {}  # raw id -> rewritten line
    record: dict = {"task": spec.name, "release": str(spec.release),
                    "release_node_dat_md5": fr.md5(spec.release / "node.dat"),
                    "task_node_dat_md5_before": task_md5_before, "types": {}}
    for type_key, entry in identification["types"].items():
        type_id = int(type_key)
        if not entry.get("affected"):
            record["types"][type_key] = {"action": "unchanged", "status": entry["status"]}
            continue
        x, other = types[type_id], types[int(entry["other_type"])]
        n_train, facts = training_neighbourhood(spec, rel_types, x, other)
        texts = x.feature_text
        if not entry["exact_rules"]:
            for raw, name in zip(x.ids, x.names):
                new_rows[int(raw)] = f"{raw}\t{name}\t{type_id}\t".encode("utf-8")
            record["types"][type_key] = {"action": "dropped", "reason": "no exact rule",
                                         "nodes": len(x.ids)}
            continue
        names = sorted(entry["exact_rules"], key=rule_order)
        sources = fr.vocabulary_sources(types, links, rel_types, type_id, other.type_id,
                                        spec.target,
                                        spec.keyword_type if other.type_id == 0 else None)
        n_full = fr.neighbourhood(links[spec.target], rel_types[spec.target], x, other)
        def on_training(name: str) -> tuple[np.ndarray, np.ndarray]:
            source, agg = name.split("/")
            g = sources[source]
            identity = source in ("V1", "V1p") and g.shape[1] == x.x.shape[1]
            mapping, amb, matched, holds, _ = fr.evaluate_rule(
                x.x, fr.aggregate(n_full, g, agg), g, identity)
            if not (holds.all() and amb == 0 and matched == len(mapping)):
                raise ValueError(f"{spec.name} type {type_id}: {name} is not exact on the release")
            cols = np.flatnonzero(mapping >= 0)
            agg_train = fr.aggregate(n_train, g, agg).tocsc()[:, mapping[cols]].tocsr()
            dense = np.zeros((x.x.shape[0], len(mapping)))
            dense[:, cols] = agg_train.toarray()
            return dense, mapping

        chosen = names[0]
        dense, mapping = on_training(chosen)
        agree = {n: bool(np.array_equal(on_training(n)[0], dense)) for n in names[1:]}
        fmt, fmt_name = value_format(texts)
        # the release's own rows must print back to their text with this format
        reprinted = sum(",".join(fmt(v) for v in x.x[i].toarray().ravel()) == texts[i]
                        for i in range(len(texts)))
        if reprinted != len(texts):
            raise ValueError(f"{spec.name} type {type_id}: the {fmt_name} format reproduces "
                             f"{reprinted}/{len(texts)} release rows")
        deg_train = np.asarray(n_train.sum(axis=1)).ravel()
        changed = 0
        for i, (raw, name) in enumerate(zip(x.ids, x.names)):
            text = ",".join(fmt(v) for v in dense[i])
            changed += text != texts[i]
            new_rows[int(raw)] = f"{raw}\t{name}\t{type_id}\t{text}".encode("utf-8")
        record["types"][type_key] = {
            "action": "rebuilt", "rule": chosen, "other_exact_rules_agree_on_training": agree,
            "mapping": mapping.tolist(), "format": fmt_name, "nodes": len(x.ids),
            "training_facts": facts, "nodes_without_training_fact": int((deg_train == 0).sum()),
            "nodes_changed": int(changed),
        }
    out_lines = []
    for line in release_lines:
        raw = int(line.split(b"\t", 1)[0])
        if raw in new_rows:
            ending = b"\r\n" if line.endswith(b"\r\n") else b"\n" if line.endswith(b"\n") else b""
            out_lines.append(new_rows[raw] + ending)
        else:
            out_lines.append(line)
    (spec.folder / "node.dat").write_bytes(b"".join(out_lines))
    record["task_node_dat_md5_after"] = fr.md5(spec.folder / "node.dat")
    record["rows_rewritten"] = len(new_rows)
    record["rows_byte_identical_to_release"] = sum(
        a == b for a, b in zip(release_lines, out_lines))
    (spec.folder / "feature_rebuild.json").write_text(json.dumps(record, indent=1) + "\n",
                                                      encoding="utf-8")
    meta_path = spec.folder / "split_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["output_md5"]["node.dat"] = record["task_node_dat_md5_after"]
    meta["node_dat_features"] = "rebuilt or dropped by scripts/lp/rebuild_lp_features.py " \
                                "(J5.3, spec v1.9 F12); see feature_rebuild.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["acm_pa", "dblp_ap", "imdb_ma"])
    parser.add_argument("--lp-root", type=Path, default=PROJECT_ROOT / "data" / "lp_derived")
    args = parser.parse_args()
    identification = {r["task"]: r for r in json.loads(args.rules.read_text(encoding="utf-8"))}
    for name in args.tasks:
        rec = rebuild_task(fr.task_spec(name, args.lp_root), identification[name])
        summary = {k: {kk: vv for kk, vv in v.items() if kk != "mapping"}
                   for k, v in rec["types"].items()}
        print(json.dumps({"task": name, "md5_before": rec["task_node_dat_md5_before"],
                          "md5_after": rec["task_node_dat_md5_after"],
                          "rows_rewritten": rec["rows_rewritten"],
                          "rows_byte_identical_to_release": rec["rows_byte_identical_to_release"],
                          "types": summary}), flush=True)


if __name__ == "__main__":
    main()
