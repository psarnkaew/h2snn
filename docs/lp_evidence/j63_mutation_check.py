"""J6.3 mutations of the loader guard (spec v1.10 F20); src/data/link_prediction_dataset.py is
restored byte-identical (md5). Run from H2S-NN/: python ../docs/lp_evidence/j63_mutation_check.py"""
import hashlib
import subprocess
import sys
from pathlib import Path

P = Path("src/data/link_prediction_dataset.py")
T = "tests.test_lp_feature_rule.FeatureRuleAndGuardTests."
MUTATIONS = {
    "guard removed": ("    if spec.get(\"feature_rule\"):\n        _assert_feature_rule(dataset_dir)\n",
                      "    pass\n", [T + "test_rerun_of_the_split_builder_is_refused"]),
    "md5 check removed": ("    if hashlib.md5(node_bytes).hexdigest() != record[\"node_dat_md5\"]:\n",
                          "    if False:\n", [T + "test_hand_edited_value_is_refused"]),
    "width check removed": ("    if any(widths.get(t, 0) != 0 for t in removed) or widths != recorded:\n",
                            "    if False:\n",
                            [T + "test_removed_type_with_features_is_refused_even_with_a_matching_md5"]),
}
original = P.read_bytes()
caught = 0
for name, (old, new, tests) in MUTATIONS.items():
    text = original.decode("utf-8").replace("\r\n", "\n")
    assert text.count(old) == 1, name
    P.write_bytes(text.replace(old, new).encode("utf-8"))
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", *tests], capture_output=True, text=True)
    finally:
        P.write_bytes(original)
    assert hashlib.md5(P.read_bytes()).digest() == hashlib.md5(original).digest()
    ok = out.returncode != 0
    caught += ok
    print(f"{name}: {'CAUGHT' if ok else 'NOT CAUGHT'} ({out.stderr.strip().splitlines()[-1]})", flush=True)
print(f"caught {caught}/{len(MUTATIONS)}; loader byte-identical after restore")
