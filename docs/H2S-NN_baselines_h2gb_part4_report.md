# Baselines on the H2GB datasets, part 4 (card 1d) — report (slim mode)

Card `docs/H2S-NN_baselines_h2gb_part4_card.md`; branch `baselines-h2gb-4` from `v3b` `72d5768c94e690e0090776d2ebdde05c5a23ac9e`,
then `git merge --no-ff 29250017dfcf0a32e716461a6787450f493bf161` (clean). Jobs: allocation 16741 on g2n01 (`srun --overlap`),
env `h2snn-b` (nothing installed or removed); evidence `docs/baselines_h2gb_evidence/p4_*`. Before each test or mutation the
job printed the sha256 of the edited files and it matched the login node's. No training run on real data, no metric. No
`src/` change outside `src/baselines/`, so no golden/mb0 compare.

**Verdict.** H2 (Hetero²Net HGB validation route) and H3 (Hetero²Net MAG ×3) are done. **H1 stopped before any change**:
D1-b's VENDORS.tsv clone row needs `/vendor/sparse_tools/` in `H2S-NN/.gitignore`, because an existing test requires every
clone folder to be ignored, and that file is outside card 1d's list. A one-line amendment unblocks it; PSHGCN on
ogbn-mag/mag-year stays pending until then.

## Units
| Unit | Commit | Jobs (16741 step) | State |
|---|---|---|---|
| H0 preconditions | `afa4f5e` | 178 (zip sha256) | done (`p4_h0_preconditions.md`) |
| H1 `sparse_tools` vendor, PSHGCN MAG label part (D1-b) | `9b435f1` | — | **stopped** (`p4_h1_stop.md`) |
| H2 Hetero²Net HGB validation route (D2) | `ad5cd88` | 179 tests, 180 mutation | done |
| H3 Hetero²Net ogbn-mag, mag-year, ogbn-mag P–P | `405f9cc` | 181-183 | done |
| H4 suite and report | this commit | suite (below) | see H4 |

## H0 — what was checked
- **The author's file.** `mag_metapath2vec_emb.zip`, 920,582,404 bytes, sha256
  `00e34ce933d7bdc12bb7d35e6a23c1057e79053f0c679f1dc2e3295be3185dbf`. Card 1c recorded none (the file was absent then), so
  this is the first record; the loader writes it beside the zip at first use and refuses a different file.
- **`sparse_tools` (`0596f00`).** `__init__.py:24` is the only route to `utils.py` (and to `sortednp`).
  `spspmm_diag_sym_ABA` = diag((D_row⁻¹A)(D_col⁻¹Aᵀ)) and `spspmm_diag_sym_AAA` = diag((D⁻¹A)²) (kernels
  `spspmm_diag_cpu.cpp:20-85`), both float32 only (`:8`).
- **PSHGCN's ogbn-mag label part.** It calls only those two functions (`processing.py:134-150`); stage-0 labels are the
  training rows only (`main.py:96-101`); self-effect removal at `main.py:110-115`.

## H1 — stop
`tests/test_vendor_manifest.py:77-81` requires every `kind=clone` row's folder to be git-ignored, and
`H2S-NN/.gitignore:90-106` lists each by name. `vendor/sparse_tools/` is not covered. The amendment needed: add
`/vendor/sparse_tools/` to `H2S-NN/.gitignore` (one line, beside the other clone rows). Nothing of H1 was written; the
build, the stub import, the label part and its four tests are prepared from H0 (`p4_h1_stop.md`). The diagonal test will be
exact on a fixture whose degrees are powers of two, because the kernels compute in float32.

## H2 — Hetero²Net HGB validation route (D2)
`src/baselines/adapters/hetero2net.py`: `evaluation_logits` takes validation logits from `train_data` (validation and test
labels hidden in `y_emb`) and test logits from `eval_data` (test hidden), as `Hetero2Net/main.py:215-219`. Before this,
validation read `eval_data`, so each validation target saw its own label (card 1c finding 2). **This changes this baseline's
HGB NC validation outputs, and with them model selection** (stated; no tuning of it has started). Test outputs are
unchanged in kind (same copy as before).

## H3 — Hetero²Net on MAG
The MAG cells read PyG's MetaPath2Vec embeddings from the author's zip (`_hetero2net_h2gb.read_mag_embeddings`: read inside
the zip, never extracted). `LargeLPData` now carries the data root, so the ogbn-mag P–P path finds the same file. The
Hetero²Net LP lane notebook was regenerated with the unchanged generator (the cell claim changed); node and login sha256
agree (`6b3fb7fe20effa52`).

## Cell delta (registry)
| Cell | Before (card 1c) | Now |
|---|---|---|
| Hetero²Net ogbn-mag, mag-year (NC) | waiting for the zip | runnable |
| Hetero²Net ogbn-mag P–P (LP) | waiting for the zip | runnable |
| PSHGCN ogbn-mag, mag-year (NC) | stop item (F2) | still pending: H1 stopped |
| Hetero²Net HGB NC (ACM/DBLP/IMDB/Freebase) | runnable, validation route leaked | runnable, fixed (D2) |

Runnable cells: 225 → 228 (NC 122 → 124 incl. HGB; LP 67 → 68). On the H2GB NC datasets 79 of 96 baseline cells are now
runnable, and 14 of 14 large-LP cells with an LP path.

## Tests and mutations (slim)
| Check | Step | Result |
|---|---|---|
| Hetero²Net HGB route: validation outputs unchanged when validation/test labels change; test outputs unchanged when test labels change | 179 | OK |
| Mutation: validation from the test-hiding copy (the code before D2) | 180 | caught ("validation/test labels reach validation targets") |
| Hetero²Net MAG path with a fake zip (inputs replaced, sha256 recorded, a different file refused, MAG metapath list) + one step with metapath-drop | 183 | OK |
| H1 tests (diagonal, label route + mutation, sys.modules no-leak, one step) | — | not run (H1 stopped) |

## H4 — full suite
`scripts/slurm/run_tests.sbatch` at `405f9cc` on g2n01 (job 16741; 2080 Ti visible): **Ran 705 tests, OK** in 2,166 s (`p4_h4_suite_g2n01_16741.txt`) = card 1c's 703 + 2 new (Hetero²Net HGB route, Hetero²Net MAG path). `test_baselines.py` has only its cell counts updated (runnable 225 → 228).

## Deviations
- H3 regenerated the Hetero²Net LP lane notebook in its own commit (the card puts regeneration in H4), so that H3's commit
  passes the notebook-comparison test.
- The route test of H2 uses a synthetic HGB-shaped heterogeneous graph (paper targets, author, field) and the adapter's
  own label builder; it does not load an HGB release (tests never read real data).

## Decisions for the author / design session
| # | Item | Options |
|---|---|---|
| 1 | H1 unblock | allow one line `/vendor/sparse_tools/` in `H2S-NN/.gitignore` (recommended: it is what every other clone row has), then H1 runs as written; or make `sparse_tools` a different kind of row (the card fixes `clone`) |

The report commit's own SHA cannot be written inside it; it is given in the final message.

branch=baselines-h2gb-4 sha=<see final message> card=docs/H2S-NN_baselines_h2gb_part4_card.md base=72d5768c94e690e0090776d2ebdde05c5a23ac9e
