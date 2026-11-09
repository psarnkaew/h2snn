"""J4.1 mutations: each must make its targeted test fail; the file is restored (byte-identical)
after each run. The four J0.3 mutations are carried over (the wrapper mutation updated to the J4.1
`labels` argument), and the three J4.1 mutations of the task card are added.
Run from H2S-NN/: python ../docs/heterophily_evidence/j41_mutation_check.py"""
import hashlib
import subprocess
import sys
from pathlib import Path

T = "tests.test_heterophily_metrics."
SCRIPT = "scripts/heterophily/compute_heterophily.py"
MUTATIONS = {
    # J0.3 (carried over)
    "H_adj: class-imbalance correction dropped": (
        "        expected = float((shares ** 2).sum())",
        "        expected = 1.0 / shares.numel()",
        T + "ToyGraphTests.test_adjusted_homophily_applies_the_class_imbalance_correction"),
    "G_P: self-loops dropped (H2GB keeps them)": (
        "        yield torch.stack([row + start, col])",
        "        keep = (row + start) != col\n"
        "        yield torch.stack([row[keep] + start, col[keep]])",
        T + "ToyGraphTests.test_own_counts_equal_hand_values"),
    "random fill: not the reference's RNG state": (
        "    label = y.clone()\n    torch.manual_seed(seed)",
        "    label = y.clone()\n    torch.manual_seed(seed + 1)",
        T + "RandomGraphTests.test_random_fill_matches_the_reference_and_labels_are_restored"),
    "wrapper: caller's label tensor filled in place": (
        "    data[target].y = (y_original if labels is None else labels).clone()",
        "    data[target].y = y_original if labels is None else labels",
        T + "RandomGraphTests.test_random_fill_matches_the_reference_and_labels_are_restored"),
    # J4.1 (task card): skip the fill = labeled-only as the primary value (both paths)
    "skip the fill, single label: labeled-only as primary": (
        "        mlh_h2gb_heterophily_mean=1.0 - sum(h_edges) / len(h_edges),",
        "        mlh_h2gb_heterophily_mean=1.0 - sum(r.labeled_only_h_edge for r in rows) / len(rows),",
        T + "FillConventionTests.test_primary_is_the_filled_reference_value"),
    "skip the fill, per genre: labeled-only as primary": (
        '        metapath=name, num_edges=by_path[name]["pairs"][0], h_edge=_mean(by_path[name]["h_edge"]),',
        '        metapath=name, num_edges=by_path[name]["pairs"][0], h_edge=_mean(by_path[name]["lo_edge"]),',
        T + "MultilabelPerGenreTests.test_unlabeled_movies_are_filled_per_genre_and_labeled_only_is_a_diagnostic"),
    # J4.1 (task card): exact match instead of per genre
    "exact match instead of per genre": (
        "        mlh_h2gb_heterophily_mean=1.0 - _mean([row.h_edge for row in rows]),",
        "        mlh_h2gb_heterophily_mean=1.0 - _mean(list(exact.values())),",
        T + "MultilabelPerGenreTests.test_primary_values_are_per_genre_means_not_exact_match"),
}
file = Path(SCRIPT)
original_bytes = file.read_bytes()
original = original_bytes.decode("utf-8")
caught = 0
for name, (old, new, test) in MUTATIONS.items():
    assert original.count(old) == 1, name
    file.write_bytes(original.replace(old, new).encode("utf-8"))
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", test], capture_output=True, text=True)
    finally:
        file.write_bytes(original_bytes)
    ok = out.returncode != 0
    caught += ok
    print(f"{name}: {'CAUGHT' if ok else 'NOT CAUGHT'} ({out.stderr.strip().splitlines()[-1]})",
          flush=True)
assert hashlib.md5(file.read_bytes()).hexdigest() == hashlib.md5(original_bytes).hexdigest()
clean = subprocess.run([sys.executable, "-m", "unittest", T[:-1]], capture_output=True, text=True)
print(f"caught {caught}/{len(MUTATIONS)}; file byte-identical after restore; clean run: "
      f"{clean.stderr.strip().splitlines()[-1]}")
