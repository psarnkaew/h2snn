"""J4.3 mutations: each must make its targeted test fail; files restored byte-identical (md5).
Run from H2S-NN/: python ../docs/lp_evidence/j43_mutation_check.py"""
import hashlib
import subprocess
import sys
from pathlib import Path

T = "tests.test_lp_derived.FreebasePairSplitTests."
MUTATIONS = {
    # The card's mutation: split by directed fact (each stored row its own unit).
    "split by directed fact instead of unordered pair": (
        Path("scripts/lp/build_derived_lp.py"),
        "    return (min(pair), max(pair))",
        "    return pair",
        T + "test_reciprocal_leak_no_held_out_pair_in_link_dat_either_orientation"),
    "self-loops kept among the training positives": (
        Path("src/data/link_prediction_dataset.py"),
        '        "exclude_self_loop_positives": True,',
        '        "exclude_self_loop_positives": False,',
        T + "test_loader_leak_checks_pass_and_positives_exclude_self_loops"),
}
caught = 0
for name, (path, old, new, test) in MUTATIONS.items():
    original = path.read_bytes()
    text = original.decode("utf-8")
    assert text.count(old) == 1, name
    path.write_bytes(text.replace(old, new).encode("utf-8"))
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", test], capture_output=True, text=True)
    finally:
        path.write_bytes(original)
    assert hashlib.md5(path.read_bytes()).digest() == hashlib.md5(original).digest()
    ok = out.returncode != 0
    caught += ok
    print(f"{name}: {'CAUGHT' if ok else 'NOT CAUGHT'} ({out.stderr.strip().splitlines()[-1]})",
          flush=True)
print(f"caught {caught}/{len(MUTATIONS)}; files byte-identical after restore")
