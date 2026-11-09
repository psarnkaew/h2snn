"""J5.1 mutations: each breaks one part of the rule search; the known-answer tests must fail.
Files restored byte-identical (md5); the mutations exist only while the test runs.
Run from H2S-NN/: python ../docs/lp_evidence/j51_mutation_check.py"""
import hashlib
import subprocess
import sys
from pathlib import Path

T = "tests.test_lp_feature_rules.FeatureRuleTests."
P = Path("scripts/lp/lp_feature_rules.py")
MUTATIONS = {
    "OR computed as SUM": (
        "        out.data = (out.data > 0).astype(np.float64)\n",
        "        out.data = out.data.astype(np.float64)\n",
        T + "test_or_of_neighbour_features_is_found_exactly"),
    "MEAN divides by degree twice": (
        "        total.data = total.data / deg[rows]\n",
        "        total.data = total.data / deg[rows] / deg[rows]\n",
        T + "test_mean_of_keyword_token_counts_is_found"),
    "column matching disabled (identity only)": (
        "        cands = by_key.get(x_keys[j], [])\n",
        "        cands = [j] if j < agg.shape[1] else []\n",
        T + "test_permuted_vocabulary_is_matched_by_column"),
    "every candidate called exact": (
        "        return (self.nodes_holding == self.nodes and self.ambiguous == 0 and self.nodes > 0\n"
        "                and self.columns_matched == len(self.mapping))\n",
        "        return True\n",
        T + "test_own_features_have_no_rule"),
}
caught = 0
for name, (old, new, test) in MUTATIONS.items():
    original = P.read_bytes()
    text = original.decode("utf-8")
    assert text.replace("\r\n", "\n").count(old) == 1, name
    P.write_bytes(text.replace("\r\n", "\n").replace(old, new).encode("utf-8"))
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", test], capture_output=True, text=True)
    finally:
        P.write_bytes(original)
    assert hashlib.md5(P.read_bytes()).digest() == hashlib.md5(original).digest()
    ok = out.returncode != 0
    caught += ok
    print(f"{name}: {'CAUGHT' if ok else 'NOT CAUGHT'} ({out.stderr.strip().splitlines()[-1]})",
          flush=True)
print(f"caught {caught}/{len(MUTATIONS)}; file byte-identical after restore")
