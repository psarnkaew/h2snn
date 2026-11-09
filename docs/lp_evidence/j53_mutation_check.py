"""J5.3 mutations of the rebuild (`scripts/lp/rebuild_lp_features.py`); file restored
byte-identical (md5). Toy: the bitwise tests of `tests/test_lp_feature_rebuild.py` must fail.
Real: the mutated rebuild runs on a scratch copy of data/lp_derived/acm_pa (the real task files
are not touched), and the features-only scorer must beat Ctrl-G there.
Run from H2S-NN/: python ../docs/lp_evidence/j53_mutation_check.py <scratch dir>"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

P = Path("scripts/lp/rebuild_lp_features.py")
T = "tests.test_lp_feature_rebuild.FeatureRebuildTests."
MUTATIONS = {
    "rebuild from all release facts (test included)": (
        "            agg_train = fr.aggregate(n_train, g, agg)",
        "            agg_train = fr.aggregate(n_full, g, agg)",
        [T + "test_or_rule_is_rebuilt_from_training_facts_only",
         T + "test_mean_rule_over_mapped_tokens_is_rebuilt"], True),
    "validation facts kept in the training neighbourhood": (
        "if tuple(p) not in held],",
        "if True],",
        [T + "test_or_rule_is_rebuilt_from_training_facts_only",
         T + "test_mean_rule_over_mapped_tokens_is_rebuilt"], False),
}
scratch = Path(sys.argv[1])
original = P.read_bytes()
caught = 0
for name, (old, new, tests, real) in MUTATIONS.items():
    text = original.decode("utf-8").replace("\r\n", "\n")
    assert text.count(old) == 1, name
    P.write_bytes(text.replace(old, new).encode("utf-8"))
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", *tests], capture_output=True,
                             text=True)
        toy = out.returncode != 0
        real_msg = ""
        if real:
            root = scratch / "lp_derived_mut"
            if root.exists():
                shutil.rmtree(root)
            shutil.copytree("data/lp_derived/acm_pa", root / "acm_pa")
            subprocess.run([sys.executable, str(P), "--rules",
                            "../docs/lp_evidence/j51_feature_rules.json", "--tasks", "acm_pa",
                            "--lp-root", str(root)], check=True, capture_output=True)
            res_path = scratch / "mut_leak.json"
            subprocess.run([sys.executable, "scripts/lp/lp_feature_leak.py", "--rules",
                            "../docs/lp_evidence/j52_rules.json", "--tasks", "acm_pa",
                            "--lp-root", str(root), "--json-out", str(res_path)], check=True,
                           capture_output=True)
            res = json.loads(res_path.read_text())[0]["splits"]
            diffs = {s: round(r["diff_mrr"], 4) for s, r in res.items()}
            beat = all(d > 0.005 for d in diffs.values())
            real_msg = f"; real acm_pa copy: scorer - Ctrl-G MRR {diffs} -> " \
                       f"{'beats the control (CAUGHT)' if beat else 'NOT CAUGHT'}"
            toy = toy and beat
    finally:
        P.write_bytes(original)
    assert hashlib.md5(P.read_bytes()).digest() == hashlib.md5(original).digest()
    caught += toy
    print(f"{name}: toy bitwise tests {'FAIL (CAUGHT)' if out.returncode else 'pass (NOT CAUGHT)'}"
          f"{real_msg}", flush=True)
print(f"caught {caught}/{len(MUTATIONS)}; rebuild script byte-identical after restore; real "
      f"task files untouched (mutations ran on a scratch copy)")
