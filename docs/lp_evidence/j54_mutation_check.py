"""J5.4 mutations: the loader ignores link.dat.valid (the carve applies to every relation), or the
writer emits every relation's rows; the target-only tests must fail. Files restored
byte-identical (md5); the real data files are not written by these tests.
Run from H2S-NN/: python ../docs/lp_evidence/j54_mutation_check.py"""
import hashlib
import subprocess
import sys
from pathlib import Path

T = "tests.test_lp_valid_files."
MUTATIONS = {
    "loader ignores link.dat.valid": (
        Path("src/data/link_prediction_dataset.py"),
        "    file_valid = _read_local_hgb_link_file(valid_path, nodes) if valid_path.exists() else None\n",
        "    file_valid = None\n",
        [T + "RealValidFileTests.test_pubmed_file_is_target_only",
         T + "RealValidFileTests.test_lastfm_file_is_target_only",
         T + "TargetOnlyFileTests"]),
    "writer emits every relation": (
        Path("scripts/lp/write_valid_files.py"),
        "            if r.split(\"\\t\")[2] in targets]\n",
        "            if True]\n",
        [T + "TargetOnlyFileTests"]),
}
caught = 0
for name, (path, old, new, tests) in MUTATIONS.items():
    original = path.read_bytes()
    text = original.decode("utf-8").replace("\r\n", "\n")
    assert text.count(old) == 1, name
    path.write_bytes(text.replace(old, new).encode("utf-8"))
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", *tests], capture_output=True,
                             text=True)
    finally:
        path.write_bytes(original)
    assert hashlib.md5(path.read_bytes()).digest() == hashlib.md5(original).digest()
    ok = out.returncode != 0
    caught += ok
    print(f"{name}: {'CAUGHT' if ok else 'NOT CAUGHT'} ({out.stderr.strip().splitlines()[-1]})",
          flush=True)
print(f"caught {caught}/{len(MUTATIONS)}; files byte-identical after restore")
