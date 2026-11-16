# Task card — Baselines on H2GB, part 4: PSHGCN label part on MAG, Hetero²Net HGB validation route, Hetero²Net MAG (card 1d)

**Base:** `v3b` = `72d5768c94e690e0090776d2ebdde05c5a23ac9e` (card 1c approved). New branch `baselines-h2gb-4` from that hash; then `git merge --no-ff <design-docs SHA in the prompt>`. Worktree `/mnt/data/peeradon/h2snn` (S1). Env `h2snn-b`; jobs only via SLURM (`srun --jobid <alloc> --overlap`, or `sbatch` if the allocation is gone).

**Mode:** Slim preparation (`docs/working_preferences.md`). Correctness checks only; no training run on real data, no metric reported.

**Author decisions (2026-11-16, review of card 1c, `docs/experiment_settings_decisions.md` row "Card 1c review"):**
- **D1-b (PSHGCN MAG label part, F2 stop item):** `sparse_tools` becomes a vendor (`VENDORS.tsv` row, `https://github.com/Yangxc13/sparse_tools`, commit `0596f00`, kind `clone`), its CPU extensions built **inside the vendor folder** by a script under `H2S-NN/scripts/setup/` (no `pip install` into `h2snn-b`, no lock line; pin count stays as on `v3b`/card 3). The adapter imports it with a stub `sortednp` module: verified by the design session that `sortednp` is used only by `sparse_tools/utils.py:292, 331` (`SparseAdjList`), which PSHGCN's ogbn-mag code never calls (it calls `spspmm_diag_sym_ABA/AAA` only, `PSHGCN/ogbn-mag/processing.py:137-179`); the import reaches `utils` only through `sparse_tools/__init__.py`'s last line.
- **D2 (Hetero²Net HGB validation route, card 1c finding 2):** fix. Validation is evaluated on the copy whose `y_emb` hides validation **and** test labels (their `train_data`, `Hetero2Net/main.py:139-143, 217-219`); test stays on the copy that hides test labels only (their `test_data`). Changes that baseline's HGB NC outputs (stated; no tuning of this baseline has started, so nothing is re-run).

## Units (in order; a unit that stops is reported and the next independent unit starts)

### H0 — Preconditions (no code)
Fetch; `origin/v3b` = `72d5768…`; branch + merge. Confirm `<data root>/NC/mag/hetero2net/mag_metapath2vec_emb.zip` exists (the author placed it, 920,582,404 bytes, 2026-11-16) and record its sha256. Read and write file:line for: `sparse_tools/__init__.py` (library loading, the `utils` import), `spspmm_diag.py` (definition of each diagonal: which product's diagonal it returns), PSHGCN ogbn-mag's label part (`processing.py`, `main.py`: which labels enter, how the diagonals are used), the HGB Hetero²Net adapter's validation call (`src/baselines/adapters/hetero2net.py` ≈ 231-300 on `v3b`).

### H1 — `sparse_tools` as a vendor and PSHGCN MAG label part (D1-b, G-2)
- `VENDORS.tsv` row; `clone_vendors.sh` clones it as any other clone row. Build script `H2S-NN/scripts/setup/build_sparse_tools.sh`: builds the two CPU extensions in place (`python setup.py build_ext --inplace` or equivalent, with `h2snn-b`'s torch, no build isolation), prints commit, torch version, the `.so` paths and `pip freeze` before/after (must be identical).
- Adapter import (`src/baselines/adapters/_pshgcn_mag.py` or a helper beside it): vendor path pushed only inside the import; a stub `sortednp` placed in `sys.modules` only for the import of `sparse_tools` and removed afterwards; the stub raises `RuntimeError("sortednp is not available: SparseAdjList is not used by PSHGCN")` on any attribute access. No import at registry/package import time.
- Wire the label part of their ogbn-mag code in full (G-2) for ogbn-mag and mag-year; flip the 2 PSHGCN MAG cells to runnable.
- **Tests (slim):** (1) each diagonal used by PSHGCN equals the dense formula its source defines, on a small random sparse fixture (float64, exact or ≤ 1e-12); (2) label route test: no validation/test label reaches any input of the label part (change held-out labels → inputs unchanged), with **one mutation** (validation labels fed into the label part) caught; (3) no leak of modules: after importing the registry and another adapter, `sortednp` and `sparse_tools` are not in `sys.modules`; (4) one step finite and parameters move.

### H2 — Hetero²Net HGB validation route (D2)
Validation from the copy hiding validation + test labels; test unchanged. **Tests:** route test on an HGB fixture (validation outputs unchanged when validation/test labels change; test outputs unchanged when test labels change), **one mutation** (validation on the test-hiding copy, the old code) caught. Reuse F4's route-test helper where it fits.

### H3 — Hetero²Net on ogbn-mag, mag-year, ogbn-mag P–P (G-5, waiting cells of card 1c)
Use the zip as their code does (read, do not unzip into git; sha256 checked at first use as card 1c set up). Flip the 3 cells. **Test:** one step finite and moves on a MAG-typed synthetic graph with the embedding path exercised (a small fake embedding file in the fixture; the real file is never read by a test).

### H3b — Baseline blockers found by the dry run (added 2026-11-16; card 2 D2, `dry-run` worktree `docs/dryrun_evidence/cells.csv`, base older than `2ef9cac`)
First check each on `baselines-h2gb-4` (some may already be fixed by cards 1b/1c); fix the rest following the source's code, each with one synthetic test that reproduces the error (dry-run rule). No real-data run here; card 2 re-runs these cells after the merge.
| Cell(s) | Error (first file:line) |
|---|---|
| HAN NC imdb | `hgb_han.py:153` (`fit_task`): too many values to unpack (expected 11) |
| HINormer NC freebase | `training.py:81` → DGL `graphconv.py:408`: 0-in-degree nodes (follow HINormer's own handling of self-loops) |
| HAN LP imdb_ma, H2SGNN LP imdb_ma | `_target_block.py:44`: target types actor, movie not contiguous in the flattened layout |
| H2SGNN LP freebase_bb | CUDA illegal memory access (`torch/cuda/__init__.py:892`); find the kernel with `CUDA_LAUNCH_BLOCKING=1` on a synthetic graph; stop item if it does not reproduce off real data |
| ChebNetII NC ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem; SeHGNN NC ieee-cis-g | `bundles.py:375` (`_base_config`) → `config.py:1023`: `data.batch_size` required in mini-batch training |

### H4 — Suite and report
Notebooks regenerated with the unchanged generator if cell claims change (`--check` after). Full suite on one node. No `src/` change outside `src/baselines/` → no golden/mb0. Report `docs/H2S-NN_baselines_h2gb_part4_report.md`: units table, cell table delta, tests/mutations, deviations, decisions; last line `branch=… sha=… card=… base=…`.

## Files
`src/baselines/**`, baseline tests, `H2S-NN/vendor/VENDORS.tsv` (one row), `H2S-NN/.gitignore` (one line `/vendor/sparse_tools/` beside the other clone rows; amendment 2026-11-16 after H1 stop, `tests/test_vendor_manifest.py:77-81`), `H2S-NN/scripts/setup/build_sparse_tools.sh`, `notebooks/baselines/` (regenerated only), evidence `docs/baselines_h2gb_evidence/p4_*`, the report. Nothing else.

## Rules and stop conditions
- Never install into or remove from `h2snn-b` (4 agents share it). `pip freeze` before/after any build: identical, or stop.
- Vendor code is never edited; the stub and paths live in our adapter.
- Stop and report if: the build needs anything not in `h2snn-b`; PSHGCN's ogbn-mag label part calls anything from `sparse_tools.utils`; a diagonal does not match its formula; the zip's sha256 differs from what card 1c recorded (if it recorded one).
- One commit per unit; push `baselines-h2gb-4` after each unit; never push another branch; after the report line, no further push.
- **Continuation (2026-11-16):** H1 and H3b run on a new branch `baselines-h2gb-5` from `7df909df4487ca8deed03aace71ccce34d5eeb69` (H0, H2, H3 reviewed there); report `docs/H2S-NN_baselines_h2gb_part5_report.md`.
