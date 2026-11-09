# J4.2 Loader fixes (spec v1.8 F9): record (2026-11-09)
Pre-registration: `j4_preregistration.md` §J4.2. The only `src/` file changed is `src/data/link_prediction_dataset.py`.

## Change (VERIFIED-CODE `src/data/link_prediction_dataset.py`)
- **`_split_validation`** is now the one place where the split by fact happens (the old loop moved here unchanged).
  - **Without `link.dat.valid`:** the old carve, same function and seeds.
  - **With the file (a):** each relation listed takes exactly the file's facts as validation, in the carve's sorted order. Its ρ-partner takes the same split flipped, and a self-inverse relation is listed in both orientations and split over unordered pairs. A relation absent from the file keeps all its facts in training.
  - **The file is checked, raising on:** a fact not in `link.dat`; a self-inverse relation listed in one orientation only; a relation and its inverse listed with different facts; a relation absent from `link.dat`.
- **`_assert_reverse_partner_is_tied` (b):** raises when a relation r′ ≠ target has the target's endpoint types reversed while ρ(target) is `None`.
  - The guard passes on every task on disk (pubmed, lastfm, acm_pa, dblp_ap, imdb_ma); YouTube and Amazon are not on disk.
  - freebase_bb has no other book→book relation, so the guard does not apply to it.
- **(c)** `scripts/lp/write_valid_files.py` runs the carve (preset `link_val_ratio` 0.10, `acm_split_seed` 42) and writes every relation's validation rows. It never overwrites a different existing file.

| File (gitignored data) | Rows | md5 |
|---|---|---|
| `data/PubMed/link.dat.valid` | 15,351 (target rel_2: 1,942) | `154fd2c5c88432369f20fa62f5c0d41d` |
| `data/LastFM/link.dat.valid` | 11,196 (target rel_0: 7,301) | `24cee86e122aaa99cbc99d560004ce21` |

## Evidence (VERIFIED-RUN)
- **`tests/test_lp_valid_files.py` (new) → 10 OK**:
  - a PubMed-like synthetic release (self-inverse target with loops): file == carve, tensor for tensor, through `build_link_prediction_dataset`;
  - a self-inverse relation listed in one orientation raises;
  - a synthetic derived task: validation = file, split 162/18/20, validation facts absent from the message graph in both orientations, non-target relations whole, a missing fact raises, the reverse-only fixture raises;
  - real: the PubMed and LastFM files equal the carve, and the derived tasks evaluate 81/9/10.
- **Real loads** (`j42_loader_check.py` → `j42_loader_check_output.txt`):
  - PubMed and LastFM: every field (positives, negatives, known positives, message graph) is **identical** with and without the file;
  - acm_pa 8,059/895/995, dblp_ap 15,913/1,768/1,964, imdb_ma 11,971/1,330/1,478 = **81.00/9.00/10.00 %**, every leak check passing, 0 validation facts in the message graph in either orientation.
- **Mutations** (`j42_mutation_check.py` → `j42_mutation_output.txt`): **3/3 caught**, loader restored byte-identical:
  - (a) file ignored;
  - (b) guard removed;
  - file facts in a different order (would change the negatives).
- **Affected modules:** `test_lp_valid_files`, `test_lp_derived`, `test_paths_and_facts`, `test_data`, `test_task_heads` → **114 OK** (`j42_affected_tests.txt`).
  - The existing test "a test fact back as its target row is caught" still passes. The guard now catches it first: its message names a leak.

## Consequences to record
- **J1.3 Finding 1 is closed.** The evaluated split of the derived tasks is J1.2's 81/9/10, instead of about 84/6/10.
- **Asymmetry (for the author, §9).** PubMed and LastFM keep the carve of their **non-target** relations, since their files list every relation's carve ("no numeric change"). So about 10 % of each non-target relation stays out of their message graph, as before J4.
  - The derived tasks' files list only the target relation, so all their non-target facts stay in the message graph. For ACM, the message graph has 272,046 edges = 273,936 facts − 895 − 995.
  - Before J4, every task lost about 9–10 % of every non-target relation. That was not recorded in J1.3.
- **J1.3 Finding 2 is closed:** a reverse-only row now raises before the split.
