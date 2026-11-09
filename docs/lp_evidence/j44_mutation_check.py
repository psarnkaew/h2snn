"""J4.4 mutations of the HeaRT generator: each must make its targeted test fail; the file is
restored byte-identical (md5) after each run.
Run from H2S-NN/: python ../docs/lp_evidence/j44_mutation_check.py"""
import hashlib
import subprocess
import sys
from pathlib import Path

GEN = Path("scripts/lp/build_heart_negatives.py")
T = "tests.test_heart_negatives."
MUTATIONS = {
    # The card's mutation: heuristics computed on the full graph (validation and test facts in).
    "heuristic graph on the full graph (train + validation + test facts)": (
        "        train_by_relation=split.train_edge_index_by_relation,",
        "        train_by_relation={r: torch.cat([e, split.test_edge_index_by_relation.get(r, e[:, :0])], 1)"
        " for r, e in split.link_edge_index_by_relation.items()},",
        [T + "SameTypeWithFeaturesTests.test_heuristics_read_training_facts_only",
         T + "SameTypeByPairTests.test_heuristics_read_training_facts_only"]),
    "L3 on every target fact (validation and test included)": (
        "        train_pos=pos(split.train_edge_index_by_relation[target]),",
        "        train_pos=pos(torch.cat([split.link_edge_index_by_relation[target],"
        " split.test_edge_index_by_relation[target]], 1)),",
        [T + "CrossTypeWithFeaturesTests.test_heuristics_read_training_facts_only"]),
    "mask only training positives (HeaRT's own filter)": (
        "    all_pos = np.concatenate([data.train_pos, data.valid_pos, data.test_pos])",
        "    all_pos = data.train_pos",
        [T + "CrossTypeWithFeaturesTests.test_no_negative_is_a_positive_of_any_split_either_orientation",
         T + "SameTypeWithFeaturesTests.test_no_negative_is_a_positive_of_any_split_either_orientation",
         T + "SameTypeByPairTests.test_no_negative_is_a_positive_of_any_split_either_orientation"]),
    "random fill with replacement from all zero-score candidates (HeaRT's default)": (
        "    return np.concatenate([head, rng.choice(pool, size=need, replace=False)]), need, int(pool.size)",
        "    return np.concatenate([head, rng.choice(np.nonzero(zero)[0], size=need)]), need, int(pool.size)",
        [T + "PieceTests.test_select_top_k_and_fill_without_replacement",
         T + "PieceTests.test_fill_draws_without_replacement"]),
}
original = GEN.read_bytes()
text = original.decode("utf-8")
caught = 0
for name, (old, new, tests) in MUTATIONS.items():
    assert text.count(old) == 1, name
    GEN.write_bytes(text.replace(old, new).encode("utf-8"))
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", *tests], capture_output=True,
                             text=True)
    finally:
        GEN.write_bytes(original)
    assert hashlib.md5(GEN.read_bytes()).digest() == hashlib.md5(original).digest()
    ok = out.returncode != 0
    caught += ok
    print(f"{name}: {'CAUGHT' if ok else 'NOT CAUGHT'} ({out.stderr.strip().splitlines()[-1]})",
          flush=True)
print(f"caught {caught}/{len(MUTATIONS)}; generator byte-identical after restore")
