"""Review response: removing the reversed-pair check must make the two finding-3 tests fail;
the loader is restored byte-identical (md5). Run from H2S-NN/."""
import hashlib
import subprocess
import sys
from pathlib import Path

LOADER = Path("src/data/link_prediction_dataset.py")
OLD = "        _assert_no_reversed_held_out_pairs(train_edges, valid_edges, test_edges, target_relations)"
T = "tests.test_lp_derived.FreebasePairSplitTests."
original = LOADER.read_bytes()
text = original.decode("utf-8")
assert text.count(OLD) == 1
LOADER.write_bytes(text.replace(OLD, "        pass").encode("utf-8"))
try:
    out = subprocess.run([sys.executable, "-m", "unittest",
                          T + "test_loader_raises_on_a_reversed_test_pair_in_link_dat",
                          T + "test_loader_raises_on_a_one_orientation_validation_file"],
                         capture_output=True, text=True)
finally:
    LOADER.write_bytes(original)
assert hashlib.md5(LOADER.read_bytes()).digest() == hashlib.md5(original).digest()
print(f"reversed-pair check removed: {'CAUGHT' if out.returncode else 'NOT CAUGHT'} "
      f"({out.stderr.strip().splitlines()[-1]}); loader byte-identical after restore")
