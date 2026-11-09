"""J4.2 mutations of the loader: each must make its targeted test fail; the file is restored
(byte-identical, md5 asserted) after each run.
Run from H2S-NN/: python ../docs/lp_evidence/j42_mutation_check.py"""
import hashlib
import subprocess
import sys
from pathlib import Path

T = "tests.test_lp_valid_files."
LOADER = Path("src/data/link_prediction_dataset.py")
MUTATIONS = {
    "(a) link.dat.valid ignored (the carve always runs)": (
        "    file_valid = _read_local_hgb_link_file(valid_path, nodes) if valid_path.exists() else None",
        "    file_valid = None",
        T + "DerivedTaskValidFileTests.test_validation_is_the_file_and_the_split_is_81_9_10"),
    "(b) reverse-only guard removed": (
        "    _assert_reverse_partner_is_tied(rho, all_relations, target_relations)",
        "    pass",
        T + "DerivedTaskValidFileTests.test_reverse_only_fixture_raises"),
    "(c) file facts not in the carve's sorted order": (
        "            valid_edge_index = _pairs_to_edge_index(sorted(facts))",
        "            valid_edge_index = _pairs_to_edge_index(sorted(facts, reverse=True))",
        T + "FileWrittenFromTheCarveTests.test_written_file_reproduces_the_loaders_carve"),
}
original = LOADER.read_bytes()
text = original.decode("utf-8")
caught = 0
for name, (old, new, test) in MUTATIONS.items():
    assert text.count(old) == 1, name
    LOADER.write_bytes(text.replace(old, new).encode("utf-8"))
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", test], capture_output=True, text=True)
    finally:
        LOADER.write_bytes(original)
    ok = out.returncode != 0
    caught += ok
    print(f"{name}: {'CAUGHT' if ok else 'NOT CAUGHT'} ({out.stderr.strip().splitlines()[-1]})",
          flush=True)
assert hashlib.md5(LOADER.read_bytes()).digest() == hashlib.md5(original).digest()
print(f"caught {caught}/{len(MUTATIONS)}; loader byte-identical after restore")
