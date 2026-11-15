# Report — Part M3a-fix (F0–F4)

**Branch** `part-m3a-fix`. It was created from `v3b` at `310e5c7b139ead942205cd8c0f8012f5e3e616d3`, then merged with `design-docs` at `9f12a04cf0d3496ece98950b83a264a3857e4910`. Both hashes were checked after `git fetch`, and the merge had no conflict.
**Card:** `docs/H2S-NN_partM3a_fix_card.md`. **Evidence:** `docs/partM3a_fix_evidence/`, with each job script next to its output.

- **Commits:** one per unit, staged by explicit path, each pushed. No other branch was touched.
- **Files never edited:** the spec, cards, decisions log and working preferences.
- **Files never touched:** the reference files, the vendor clones and the author's untracked files.

**Where it ran.** Every computation ran as a step of the author's allocation **job 16741** (g2n01, RTX 2080 Ti, i9-10900K, 6 CPUs), using `srun --jobid=16741 --overlap`. Each job script exports E and `TORCH_EXTENSIONS_DIR`; F4 ran `scripts/slurm/run_tests.sbatch` itself. Mutations were applied and restored inside the unit's own job, with no other job of mine running at the time, and each restore was checked with `cmp`.

**Gate after every unit** (`docs/partM3a_fix_evidence/gate.sh`, E exported): the golden tests (`v2_golden_cluster.pt`) must pass, and `mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt` must report `differing: none`.

| After | Step | Golden | mb0 compare |
|---|---|---|---|
| F1 | 16741 step 2 | 5/5 OK | `differing: none`, exit 0 |
| F2 | 16741 step 3 | 5/5 OK | `differing: none`, exit 0 |
| F3 | 16741 step 4 | 5/5 OK | `differing: none`, exit 0 |
| F4 | 16741 step 6 | 5/5 OK | `differing: none`, exit 0 |

## Verdict

- **Done: F0–F4.** No stop condition was hit:
  - the gate never differed;
  - no test of kept code needed deleting;
  - no non-test caller builds `concat_mlp` (the only builder is `h2snn_model.py:670`, from `config.data.decoder_type`, which the config now refuses).
- **`concat_mlp` is refused by `DataConfig` validation and removed from `LinkPredictionReadout`** (spec v1.22 W2). The dot and DistMult readouts are bit-identical to `310e5c7`: parameters, logits and gradients, checked directly.
- The three Colab notebooks no longer have the decoder switch or any `concat_mlp`. The documentation text is corrected.
- **Full suite (F4, git `8033a87`): Ran 711 tests — FAILED (failures=1).** The only failure is `test_notebooks_are_committed_clean`. That test reads the baseline notebooks, which this card does not touch. 711 = 706 (M3a-A4) + 4 (F1) + 1 (F2).

## F0 — Preconditions (merge commit `c9b1f3e`; the list was committed with F1)

`git fetch` showed `origin/v3b` = `310e5c7…` and `origin/design-docs` = `9f12a04…`. I then ran `git checkout -b part-m3a-fix 310e5c7…` and `git merge --no-ff 9f12a04…`, with no conflict.

**The F0 list** (`docs/partM3a_fix_evidence/f0_hits.md`) comes from the card's grep: `concat_mlp|include_decoder_type|INCLUDE_DECODER_TYPE` over `H2S-NN` (`*.py`, `*.ipynb`, `*.md`, `*.html`, `*.yaml`, vendor excluded). VERIFIED-RUN on login01.

| Hits (file:line) | Unit |
|---|---|
| `src/configs/config.py:672` (comment), `:958` (validation), `:206-208` (comment "model code kept") | F1 |
| `src/models/link_readout.py:21, 48, 76` (docstring, `__init__`, `forward`) | F1 |
| `src/models/h2snn_model.py:15` | F1 |
| `scripts/run_h2snn_runs.py:39`, `scripts/train_h2snn.py:24`, `scripts/run_h2snn_wandb_tuning.py:76-77` (comments "model code stays") | F1 |
| `tests/test_task_heads.py:149` | F1 (stated test change) |
| `notebooks/h2snn_link_prediction_colab.ipynb:119, 288, 331-332`; `h2snn_graph_classification_colab.ipynb:296, 339-340`; `h2snn_node_classification_colab.ipynb:1` (two places in the one-line JSON) | F2 |
| `CLAUDE.md:884, 962`; `AGENTS.md:164` (the same text) | F3 |
| `docs/h2snn_current_model_documentation.html:1553, 1590` | F3 |
| `src/configs/config.py:109-112, 209, 217` (M3a refusal); `scripts/run_h2snn_wandb_tuning.py:311, 318, 336`; `tests/test_partM3a_decoder.py`; `tests/test_search_spaces.py:47, 56-57`; `tests/test_wandb_and_runs.py:484, 528, 749` | kept: these refuse, test the refusal, or state the removal |

The other five `h2snn_*` notebooks have no hit. F2's test now covers all of them.

## F1 — `concat_mlp` refused by config, removed from the model (commit `eff3e51`)

**Code changes.**
- **`src/configs/config.py`.** In the LP branch of `DataConfig` validation, `concat_mlp` now raises `ValueError(REMOVED_DECODER_MESSAGE.format(where="data.decoder_type"))`, i.e. "data.decoder_type: concat_mlp is removed from tuning and runs (spec v1.22 W2) …". Any other value outside {`dot`, `distmult`} still raises "Unsupported data.decoder_type.". The `decoder_type` comment now reads `"dot" | "distmult"`. The M3a refusal in `reject_never_tuned_settings` stays; only its comment was updated.
- **`src/models/link_readout.py`.** The `concat_mlp` branch is removed from `__init__` and `forward`, and the docstring is rewritten. An unknown decoder still raises.
  - `hidden_dim` and `dropout` stay, now unused, because the caller `src/models/h2snn_model.py:674-675` passes them (`classifier_hidden_dim`, `classifier_dropout`). The docstring says so. This is the card's "may stay if removing them would change a caller".
- **Comments and docstrings corrected:** `src/models/h2snn_model.py:15`, `scripts/run_h2snn_runs.py:39`, `scripts/train_h2snn.py:24`, `scripts/run_h2snn_wandb_tuning.py:76-77`.

**Tests.** New `tests/test_partM3afix_concat_mlp.py` (4 tests):
- **config refusal:** `TrainConfig.from_dict` on pubmed with `concat_mlp` raises, with a message matching `data.decoder_type: concat_mlp is removed .*W2`; dot and distmult still resolve.
- **YAML and the run scripts:** a YAML goes through the loader. For `train_h2snn.py --config` and `run_h2snn_runs.py --config`, the test runs `parse_args` with the patched argv, then the script's own `load_config`. These are the first two steps of `main()`. `main()` itself is not called, so nothing can train under a mutation.
- **readout:** `LinkPredictionReadout(decoder_type="concat_mlp")` raises `Unsupported decoder_type 'concat_mlp'`; dot and distmult have no `mlp`.
- **tuning refusal:** the M3a refusal still holds.

**Stated test change:** `tests/test_task_heads.py:149`, see the table below.

**VERIFIED-RUN** (16741 step 2, `f1_tests.txt`).
- `tests.test_partM3afix_concat_mlp`, `test_task_heads`, `test_partM3a_decoder`, `test_search_spaces`, `test_v2_model`, `test_partMB_link` → **Ran 75, OK**.
- **Direct check that dot/DistMult are unchanged** (`f1_readout_equal.py`). It loads `link_readout.py` as of `310e5c7` via `git show` next to the current one and seeds both the same. For dot and distmult, with 1 and 2 relations, "parameters, logits, gradients equal to 310e5c7: True", for 4 out of 4 cases.
- **Mutation** (config accepts `concat_mlp` again): `test_config_refuses_concat_mlp` **FAILED** with `AssertionError: ValueError not raised`. Restored identical, then OK.

## F2 — Notebooks (commit `3d04136`)

**Notebook edits.**
- In `h2snn_link_prediction_colab.ipynb`, `h2snn_graph_classification_colab.ipynb` and `h2snn_node_classification_colab.ipynb`, the line `INCLUDE_DECODER_TYPE_IN_TUNING = False` is removed, and so is the two-line `if INCLUDE_DECODER_TYPE_IN_TUNING: command.append('--include_decoder_type')`.
- In the LP notebook, the decoder comment now reads `# one of: 'dot', 'distmult'`.
- Edits are line edits of cell sources only, and the result was checked by a parsed comparison with `HEAD`. Per notebook:
  - the cell count, every cell's keys other than `source`, and the top-level metadata are identical;
  - the removed and added source lines are exactly those listed above;
  - outputs and execution counts are unchanged;
  - the NC notebook stays one line.

**Test.** New `tests/test_wandb_and_runs.py::WandbAndRunReportingTests::test_h2snn_notebooks_have_no_decoder_switch_or_concat_mlp`. Every `notebooks/h2snn_*.ipynb` (8 files) must contain none of `--include_decoder_type`, `INCLUDE_DECODER_TYPE` or `concat_mlp`.

**VERIFIED-RUN** (16741 step 3, `f2_tests.txt`).
- The new test and the two existing notebook tests: Ran 3, OK.
- The whole `tests.test_wandb_and_runs` module: Ran 13, OK.
- **Mutation** (the switch line put back in the GC notebook): **FAILED** with `AssertionError: 'INCLUDE_DECODER_TYPE' unexpectedly found in …`. Restored identical, then OK.

## F3 — Documentation text (commit `8033a87`)

These lines only, text only:
- **`H2S-NN/CLAUDE.md` l.884:** `data.decoder_type` (`dot` | `distmult`; spec v1.22 W2, `concat_mlp` removed).
- **`H2S-NN/CLAUDE.md` l.962:** the LP head row now reads `dot` / `distmult` (spec v1.22 W2; `concat_mlp` removed).
- **`H2S-NN/AGENTS.md` l.164:** the same text in the key-files line.
- **`H2S-NN/docs/h2snn_current_model_documentation.html` l.1553:** a historical changelog entry. A note says both flags were later removed, `--include_decoder_type` by W2, and that the decoder is `dot` | `distmult`.
- **The same file, l.1590:** "`dot` stays relation-agnostic (the LP decoder is `dot` | `distmult`; `concat_mlp` was removed by spec v1.22 W2)".

After F3, every remaining hit of the F0 grep either refuses or tests `concat_mlp` / the flag, or states the removal.

Gate: 16741 step 4 (`f3_gate.txt`). It was taken before the commit, so it shows `git=3d04136` with the F3 edits in the working tree.

## F4 — Full suite (this commit)

VERIFIED-RUN: `srun --jobid=16741 --overlap bash scripts/slurm/run_tests.sbatch` ran as step 5 on g2n01 (RTX 2080 Ti, AVX2 with E, 6 threads) at git `8033a87`. Output: `f4_full_suite.txt`.

**`Ran 711 tests in 1084.712s — FAILED (failures=1)`**: `test_baselines.BaselineNotebookTests.test_notebooks_are_committed_clean` (`AssertionError: 1 is not None`, the baseline notebooks). There are 0 errors and 0 skips. The golden tests, the float64 scope test and all new tests are green.

Gate (step 6, `f4_gate.txt`): golden 5/5 OK, `compared 139 tensors; differing: none`, exit 0.

**Deviation:** as in M3a, this was a step of the author's allocation with 6 CPUs, not 8. The bit-exact checks force 1 thread themselves.

## Stated test changes (old, new, reason)

| Test | Old | New | Reason |
|---|---|---|---|
| `tests/test_task_heads.py::…test_each_decoder_trains_one_step_and_reaches_the_encoder` (l.149) | loop over (`dot`, `distmult`, `concat_mlp`): each decoder trains one step | loop over (`dot`, `distmult`); a comment points to the refusal test | Card F1: the `concat_mlp` subcase is the one subcase the card names for removal. The model no longer builds `concat_mlp` (spec v1.22 W2), and its refusal is tested in `tests/test_partM3afix_concat_mlp.py` |

No other existing assertion changed. F2 added a new test; the other notebook tests are unchanged.

## Decisions for the author

| # | Decision | Options | Recommendation | Reason |
|---|---|---|---|---|
| 1 | `LinkPredictionReadout` keeps two unused constructor arguments (`hidden_dim`, `dropout`) | remove them and the two keyword arguments in `h2snn_model.py:674-675` (no RNG or parameter change) / keep | remove in the legacy-removal card | They no longer do anything; keeping them follows the card's "may stay if removing them would change a caller" |
| 2 | `classifier_dropout` is still passed to the LP readout and appears in LP configs, where it has no effect (M3a-A0) | leave as is (the tuner never puts it in the LP space) / refuse or ignore it for LP explicitly | leave | The tuner already excludes it on LP; a config value is harmless |

## Out-of-scope findings (listed, not fixed)

1. `tests/test_wandb_and_runs.py:484, 528` build `SimpleNamespace(..., include_decoder_type=False, ...)` for two tuning-state tests. The attribute is read by no code since M3a; it is harmless, and the card names no change there.
2. `test_notebooks_are_committed_clean` still fails, as before: the baseline notebooks have an `execution_count`. It is outside this card.
3. `scripts/run_graph_classification_repeated_cv_wandb_tuning.py:59` still has the dead `--decoder_type` without choices (M3a out-of-scope 3). It accepts any string but is read nowhere.
4. Still assigned to M3b, unchanged: the graphbolt `GBWarning`, the dirty `vendor/H2GB` clone, `run_use_dataset.sbatch`, and the H2GB-data items.

branch=part-m3a-fix sha=<see final message> card=docs/H2S-NN_partM3a_fix_card.md base=310e5c7b139ead942205cd8c0f8012f5e3e616d3
