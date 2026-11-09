"""J5.3 gate 1 (bitwise): for every node of a rebuilt type, the stored feature equals the rule
applied to its training neighbours, computed by the independent node-by-node path of
`lp_feature_leak.rule_on_training` (parsed float64 values, and the loader's float32 tensor).
A dropped type has no features in the loader. Every other node.dat row is byte-identical to the
release, and the node-classification releases are unchanged (md5 against the file written
before the rebuild). Read only. Run from H2S-NN/: python ../docs/lp_evidence/j53_bitwise_check.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("scripts/lp").resolve()))
sys.path.insert(0, str(Path(".").resolve()))
import lp_feature_leak as fl  # noqa: E402
import lp_feature_rules as fr  # noqa: E402
from src.data.link_prediction_dataset import (  # noqa: E402
    _LOCAL_HGB_LINK_DATASETS, _load_local_hgb_nodes)

ok = True
for task in ("acm_pa", "dblp_ap", "imdb_ma"):
    spec = fr.task_spec(task, Path("data/lp_derived"))
    record = json.loads((spec.folder / "feature_rebuild.json").read_text(encoding="utf-8"))
    names = _LOCAL_HGB_LINK_DATASETS[task]["node_type_names"]
    loaded = _load_local_hgb_nodes(spec.folder, names).data
    stored = fr.read_nodes(spec.folder)
    for type_key, info in record["types"].items():
        type_id = int(type_key)
        name = names[type_id]
        if info["action"] == "rebuilt":
            source, agg = info["rule"].split("/")
            g, mapping, other, _, rel_types, _, _ = fl.chosen_rule(spec, type_id, source, agg)
            same_map = mapping.tolist() == info["mapping"]
            train = fl.training_pairs(spec.folder, spec.target)
            expected = fl.rule_on_training(train, stored[type_id].ids, stored[other].ids,
                                           rel_types[spec.target][0] == type_id, g, mapping, agg)
            got = stored[type_id].x.toarray()
            nodes_equal = int((got == expected).all(axis=1).sum())
            tensor = loaded[name].x.numpy()
            tensor_equal = bool(np.array_equal(tensor, expected.astype(np.float32)))
            zero_without_fact = bool(np.all(expected[np.abs(expected).sum(axis=1) == 0] == got[
                np.abs(expected).sum(axis=1) == 0]))
            passed = same_map and nodes_equal == len(got) and tensor_equal and zero_without_fact
            ok &= passed
            print(f"{task} {name}: rebuilt by {info['rule']}; mapping as recorded {same_map}; "
                  f"nodes bitwise equal {nodes_equal}/{len(got)}; loader float32 tensor equal "
                  f"{tensor_equal}; {'PASS' if passed else 'FAIL'}")
        elif info["action"] == "dropped":
            passed = stored[type_id].x is None and "x" not in loaded[name]
            ok &= passed
            print(f"{task} {name}: dropped; no stored feature and no loader x: "
                  f"{'PASS' if passed else 'FAIL'}")
    release = (spec.release / "node.dat").read_bytes().splitlines(keepends=True)
    task_rows = (spec.folder / "node.dat").read_bytes().splitlines(keepends=True)
    rewritten = {int(t) for t, i in record["types"].items() if i["action"] != "unchanged"}
    other_rows = [(a, b) for a, b in zip(release, task_rows)
                  if int(a.split(b"\t")[2]) not in rewritten]
    same_other = len(release) == len(task_rows) and all(a == b for a, b in other_rows)
    ok &= same_other
    print(f"{task}: rows {len(task_rows)}; rows of unchanged types byte-identical to the release: "
          f"{same_other} ({len(other_rows)} rows)")

before = {}
for line in Path("../docs/lp_evidence/j53_nc_release_md5_before.txt").read_text().splitlines():
    digest, path = line.split(maxsplit=1)
    before[path.lstrip("*")] = digest
changed = [p for p, d in before.items() if hashlib.md5(Path(p).read_bytes()).hexdigest() != d]
ok &= not changed
print(f"node-classification releases (data/acm, dblp, imdb, freebase): {len(before)} files, "
      f"changed {len(changed)}")
print("ALL PASS" if ok else "SOME FAIL")
