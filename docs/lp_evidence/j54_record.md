# J5.4 Validation files, target relation only (spec v1.9 F13)
2026-11-10.

## Change
`scripts/lp/write_valid_files.py` gains `target_rows` and `--mode target` (now the default): the target relation's rows of the loader's carve, byte for byte. `--mode carve` keeps the J4.2 file (every relation) reachable.
- An existing file is overwritten only if it is one of these two files.
- The loader needs no change: a relation absent from the file keeps all its facts (J4.2 (a)).

**Command (VERIFIED-RUN):** `python scripts/lp/write_valid_files.py --mode target`. Output: `j54_write_valid_files_output.txt`.

| Dataset | Target | Rows in the file | md5 before (J4.2, every relation) | md5 after |
|---|---|---|---|---|
| PubMed | relation 2 | 1,942 | `154fd2c5…` | `3d2b5114…` |
| LastFM | relation 0 | 7,301 | `24cee86e…` | `8ffd3a68…` |

**The target rows are identical to before.** The script's carve has the same md5 as the previous file (`carve_md5` = `previous_md5`), and the new file is that carve's target rows.

## Gate: loader check (`j54_loader_check.py`, before/after snapshot → `j54_loader_check_output.txt`: ALL PASS)
- **Target positives identical tensors** (train / validation / test):
  - PubMed: 32,167 / 1,942 / 8,528.
  - LastFM: 66,966 / 7,301 / 18,567.
- The 1:1 validation and test negatives are also identical. This was reported, not gated.
- **Every non-target relation is complete:** training = all of its `link.dat` rows, validation 0.
  - PubMed relations 0, 1, 3–9 regain 954, 1,729, 2,054, 3,442, 4,330, 297, 225, 340 and 38 facts.
  - LastFM relations 1 and 2 regain 2,174 and 1,721.
- **Message graph:**
  - PubMed: 169,036 → 179,784 edges. The 10,748 added edges are fewer than the 13,409 restored rows, because the tied fact layer (D7) stores each self-inverse relation once per unordered pair: PubMed relations 0, 2, 5 and 9. The per-relation edge counts sum exactly to 179,784 (`j54_edge_count_diag.py` → `j54_edge_count_diag_output.txt`).
  - LastFM: 100,128 → 102,936 edges; LastFM relation 1 is self-inverse.

## Tests and mutations
`tests/test_lp_valid_files.py`: 11 tests, OK (`j54_affected_tests.txt`).
- The two J4.2 real-data tests "file equals the carve" encoded the superseded F9 behaviour. They are replaced by "file is target-only": the target split equals the carve's, and the others are complete.
  - The new tests read the split through the loader's own path (`_load_local_hgb_link_data`), so a loader that ignored the file would fail them.
- New toy test: the target-only file keeps the carve's target bytes and split and leaves the rest complete.

Mutation check (`j54_mutation_check.py` → `j54_mutation_output.txt`): **2/2 caught**.
- The loader ignores the file: 3 tests fail.
- The writer emits every relation: the toy test fails.
- The files are restored byte-identical, and `src/` is clean after the run.

## Consequence
PubMed's and LastFM's message graphs changed, so their HeaRT negatives are regenerated in J5.5 (CN, PPR and L3 read the message graph).
