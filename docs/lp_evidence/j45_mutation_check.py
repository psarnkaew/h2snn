"""J4.5 mutations: the forbidden "oracle top-k" made the default must make the IMDB protocol tests
fail; files restored byte-identical (md5). The mutations exist only while the test runs.
Run from H2S-NN/: python ../docs/lp_evidence/j45_mutation_check.py"""
import hashlib
import subprocess
import sys
from pathlib import Path

T = "tests.test_imdb_protocol."
MUTATIONS = {
    "H2S-NN trainer metric: default protocol oracle_topk": (
        Path("src/utils/misc.py"),
        '    protocol: str = "threshold",',  # unique in misc.py (the function's default)
        '    protocol: str = "oracle_topk",',
        T + "TrainerProtocolTests.test_f1_uses_logit_above_zero_on_labelled_rows_only"),
    "baseline metric: default protocol oracle_topk": (
        Path("src/baselines/metrics.py"),
        'DEFAULT_MULTILABEL_PROTOCOL = "threshold"',
        'DEFAULT_MULTILABEL_PROTOCOL = "oracle_topk"',
        T + "BaselineProtocolTests.test_classification_metrics_default_is_threshold"),
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
