"""J0.3 mutations: each must make its targeted test fail; the file is restored after each run.
Run from H2S-NN/: python ../docs/heterophily_evidence/j03_mutation_check.py"""
import subprocess
import sys
from pathlib import Path

T = "tests.test_heterophily_metrics."
SCRIPT = "scripts/heterophily/compute_heterophily.py"
MUTATIONS = {
    # The card's mutation: H_adj without the class-imbalance correction (uniform class shares).
    "H_adj: class-imbalance correction dropped": (SCRIPT,
        "        expected = float((shares ** 2).sum())",
        "        expected = 1.0 / shares.numel()",
        T + "ToyGraphTests.test_adjusted_homophily_applies_the_class_imbalance_correction"),
    "G_P: self-loops dropped (H2GB keeps them)": (SCRIPT,
        "        yield torch.stack([row + start, col])",
        "        keep = (row + start) != col\n"
        "        yield torch.stack([row[keep] + start, col[keep]])",
        T + "ToyGraphTests.test_own_counts_equal_hand_values"),
    "random fill: not the reference's RNG state": (SCRIPT,
        "    label = y.clone()\n    torch.manual_seed(seed)",
        "    label = y.clone()\n    torch.manual_seed(seed + 1)",
        T + "RandomGraphTests.test_random_fill_matches_the_reference_and_labels_are_restored"),
    # Added in J0.5: the original wrapper, which let calcHomophily fill the caller's tensor.
    "wrapper: caller's label tensor filled in place": (SCRIPT,
        "    y_original = data[target].y\n    data[target].y = y_original.clone()",
        "    y_original = data[target].y.clone()",
        T + "RandomGraphTests.test_random_fill_matches_the_reference_and_labels_are_restored"),
}
caught = 0
for name, (path, old, new, test) in MUTATIONS.items():
    file = Path(path)
    original = file.read_bytes().decode("utf-8")
    assert original.count(old) == 1, name
    file.write_bytes(original.replace(old, new).encode("utf-8"))
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", test], capture_output=True, text=True)
    finally:
        file.write_bytes(original.encode("utf-8"))
    ok = out.returncode != 0
    caught += ok
    print(f"{name}: {'CAUGHT' if ok else 'NOT CAUGHT'} ({out.stderr.strip().splitlines()[-1]})",
          flush=True)
clean = subprocess.run([sys.executable, "-m", "unittest", T[:-1]], capture_output=True, text=True)
print(f"caught {caught}/{len(MUTATIONS)}; clean after restore: {clean.stderr.strip().splitlines()[-1]}")
