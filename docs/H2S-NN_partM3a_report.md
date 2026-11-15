# Report — Part M3a (A0, A1, A2, A3, A3b, A4)

**Branch** `part-m3a`. Created from `v3b` at `a68cbcc84042406705ca45c3db42684ef52daff1` and merged with `design-docs` at `19782cfbdfc317b20869bec7c88510959e51b459`. Both hashes were checked after `git fetch`; the merge had no conflict.
**Card** `docs/H2S-NN_partM3a_card.md`.
**Evidence** `docs/partM3a_evidence/`. Every job script sits there next to its output. Each output starts with `git rev-parse HEAD`, the hostname and the job id/step.
**Rules followed:** one commit per unit, staged by explicit path and pushed after each unit; no other branch touched. Spec, cards, decisions log, handoffs, working preferences and every reference file are untouched. Environment `h2snn-b`.

**Where it ran.** All computation ran as steps of the author's allocation **job 16721** (g2n01, RTX 2080 Ti, i9-10900K, 6 CPUs; `srun --jobid=16721 --overlap`).
- Each unit's job script exports E (`ATEN_CPU_CAPABILITY=avx2`, `MKL_CBWR=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`) and `TORCH_EXTENSIONS_DIR`, as `scripts/slurm/run_tests.sbatch` does.
- A4 ran `scripts/slurm/run_tests.sbatch` itself.
- Mutations were applied and restored inside the unit's own job. No other job of mine was running at the time, and each restore was checked with `cmp` (and `git diff` for `src/`).

**Gate after every unit** (`docs/partM3a_evidence/gate.sh`, with E): the golden tests (`NodeClassificationGoldenTests`, `SymmetricIsThePreviousModelTests` against `v2_golden_cluster.pt`) must pass, and `python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt` must report `differing: none` with exit 0.

| After | Step | Golden | mb0 compare |
|---|---|---|---|
| A1 | 16721 step 18 | 5/5 OK | `differing: none`, exit 0 |
| A2 | 16721 step 19 | 5/5 OK | `differing: none`, exit 0 |
| A3 | 16721 step 20 | 5/5 OK | `differing: none`, exit 0 |
| A3b | 16721 step 21 | 5/5 OK | `differing: none`, exit 0 |
| A4 | 16721 step 23 | 5/5 OK | `differing: none`, exit 0 |

## Verdict

- **Done: A0–A4.** No stop condition occurred:
  - full batch is bit-identical after every unit;
  - A2's equality holds on every field;
  - no method change;
  - no baseline LP decoder outside {dot, DistMult}.
- **A4 full suite** (`run_tests.sbatch`, git `0d08927`): **Ran 706 tests — FAILED (failures=1)**. The only failure is `test_notebooks_are_committed_clean`, as in CS8-5. 706 = 697 (CS8-5) + 6 (A1) + 3 (A2). All new tests are green.
- No real data, no metric, no comparison between methods. `src/` changes:
  - `src/configs/config.py`: the never-tuned guard;
  - `src/data/scopes.py`: the CSR index, with output identical to the former builder;
  - `src/baselines/runtime/dependencies.py`: messages only.

  The method (energy, maps, `g`, solver, WISP's `p`, seed formulas) is unchanged.

## A0 — Preconditions (merge commit `b58900c`; findings committed with A1)

**Branch.** `git fetch`, then:
- `origin/v3b` = `a68cbcc…`;
- `origin/design-docs` = `19782cf…`;
- `git checkout -b part-m3a a68cbcc…`;
- `git merge --no-ff 19782cf…`: no conflict.

**Findings.** Full list with file:line in `docs/partM3a_evidence/a0_findings.md` (VERIFIED-CODE).

**Where the decoder is chosen today.**
- Config:
  - `src/configs/config.py:652` (`decoder_type`, default `dot`);
  - validation at `:938-939` accepts {dot, distmult, concat_mlp}.
- Model:
  - `src/models/h2snn_model.py:669-677` builds `LinkPredictionReadout`;
  - `src/models/link_readout.py:42-57` (construction) and `:68-79` (scoring).
- Tuner `scripts/run_h2snn_wandb_tuning.py`:
  - `:93` the conditional parameter name;
  - `:132-133` `--decoder_type` (with concat_mlp) and `--include_decoder_type`;
  - `:270-275` trial validation;
  - `:297-345` `build_sweep_config`: the decoder entered the space only with `--include_decoder_type`, as {dot, distmult, concat_mlp};
  - `:379-380` a trial's value is written to the config.
- Run scripts: `scripts/run_h2snn_runs.py:38` and `scripts/train_h2snn.py:23` (choices included concat_mlp).
- GC tuning script: `scripts/run_graph_classification_repeated_cv_wandb_tuning.py:59-60`. Its `--decoder_type` and `--include_decoder_type` are read nowhere else (dead flags).

**Dropout of the LP head.** `classifier_dropout` is consumed only by the NC head (`h2snn_model.py:688`) and by the LP readout's argument (`:675`). `LinkPredictionReadout` uses that argument only inside the `concat_mlp` MLP (`link_readout.py:48-54`). The dot/DistMult branches and the endpoint projection have no dropout. So with dot/DistMult, `classifier_dropout` has **no effect**, and the tuner's rule (`:330-337`, LP classifier dropout only for `concat_mlp`) was consistent with this. After W2 that branch is gone.

**Baselines' LP decoder values.** Every baseline LP space has {dot, distmult}, and so does the shared head (`src/baselines/heads.py:58-64`):

| Baselines | Location |
|---|---|
| gcn, gat | `hgb_gnn.py:129` |
| han | `hgb_han.py:124` |
| nsd | `nsd.py:226` |
| csnn | `csnn.py:177` |
| pshgcn | `pshgcn.py:156` |
| h2gformer | `h2gformer.py:219` |
| hetero2net | `hetero2net.py:178` |
| h2sgnn | `h2sgnn.py:171` |

No baseline differs (the report-only stop case did not occur).

**Scopes and transport.**
- `src/data/scopes.py:52-63` was O(E) per target.
- The trainers call `build_scope_batch(graph, targets, cells)` (`minibatch_trainer.py:119`, `minibatch_lp.py:76`).
- Three tests patch it with a 3-argument spy, so that signature is kept.
- `build_cell_group` (`sheaf_energy.py:230-241`) uses `PathTransportOperator` (`map_operators.py:472-509`). That operator validates orientation first, then picks near/far maps from the hop sign.

## A1 — LP decoder in {dot, DistMult} (commit `481cbdd`)

**Tuner** (`scripts/run_h2snn_wandb_tuning.py`):
- new `SPACE_DECODER_TYPE = {"values": ["dot", "distmult"]}`, commented as the decoders of the shared LP head;
- `build_sweep_config` puts `decoder_type: {values: [dot, distmult]}` into **every LP sweep** and never into NC/GC;
- `--include_decoder_type` is removed, and so is the `concat_mlp` branch that added `classifier_dropout`;
- a base-config decoder outside the two raises a named error;
- `--decoder_type` choices are {dot, distmult};
- trial validation accepts only the two.

**`--include_decoder_type`** is also removed from `scripts/run_graph_classification_repeated_cv_wandb_tuning.py`.

**Refusal** (`src/configs/config.py`): `reject_never_tuned_settings` refuses `data.decoder_type: concat_mlp` in a base config. It also refuses any `decoder_type` search-space entry or trial value containing `concat_mlp` (`{values: [...]}`, `{value: ...}` or a scalar). The error is the named `REMOVED_DECODER_MESSAGE`: "concat_mlp is removed from tuning and runs (spec v1.22 W2) …".

**Run scripts:** `scripts/run_h2snn_runs.py` and `scripts/train_h2snn.py` now have `--decoder_type` choices {dot, distmult}.

**Unchanged:** the model code of `concat_mlp` (`link_readout.py`) and the config validation (`config.py:938`).

**Tests:**
- New `tests/test_partM3a_decoder.py` (6 tests):
  - every LP dataset of the campaign, for base decoders None, dot and distmult, gets exactly `{values: [dot, distmult]}`, and the rest of the space equals the pre-W2 dot space (no `classifier_dropout`);
  - NC/GC sweeps contain no `decoder_type`;
  - the tuned values equal the `decoder_type` values of **every runnable baseline LP cell**, read via `iter_cells()` / `get_adapter().search_space()`;
  - `concat_mlp` is refused by the sweep builder, the guard, a loaded YAML, three search-space/trial forms, `validate_trial_params`, and the command lines of the tuner and both run scripts (`invalid choice: 'concat_mlp'`);
  - `--include_decoder_type` is unrecognised by both tuning scripts.
- Stated test changes: listed under "Assertions changed by W2".

VERIFIED-RUN, 16721 step 18 (`a1_tests.txt`): `tests.test_partM3a_decoder tests.test_search_spaces tests.test_wandb_and_runs tests.test_cooperative tests.test_task_heads tests.test_paths_and_facts` → **Ran 109, OK**.

**Mutations** (run against their target test only):

| Mutation | Result |
|---|---|
| The tuner omits `decoder_type` on LP (`if False:`) | `test_lp_sweeps_tune_the_decoder` **FAILED (failures=24)**: `AssertionError: 'decoder_type' not found in {...}`, one failure per LP dataset × base decoder |
| `concat_mlp` accepted by the tuner (added to `SPACE_DECODER_TYPE`) | `test_concat_mlp_is_refused_by_the_tuner` **FAILED**: `AssertionError: ValueError not raised` |

Both were restored identical.

**Deviation.** In the first run (step 17), mutation 1 surfaced as a `KeyError` (the test popped the key) rather than an assertion. I added `assertIn("decoder_type", params)` before the pop and reran everything (step 18). The evidence file holds the second run.

## A2 — Scope builder with a CSR incidence index (commit `f714270`)

**`src/data/scopes.py`:**
- `IncidenceIndex` (`rowptr`, `fact_ids`; `facts_of(v)`) holds each node's fact ids in ascending order, with a self-loop fact listed once.
- `build_incidence_index(edge_index, num_nodes)` builds it in O(E log E).
- `incidence_index(graph)` builds the index once per graph and caches it on the graph's `edge_index` tensor. A cache hit needs the same live tensor (weak reference), the same `_version` (an in-place change rebuilds) and the same `num_nodes`. A training graph and each evaluation graph therefore get their own index (N6).
- `scope_members` and `build_scope_batch` read it. Their positional signatures are unchanged, with an optional `index` argument, so the trainers and the 3-argument spies are untouched.
- The rest of the builder is unchanged.

**Tests:**
- `tests/scope_reference.py` is the `v3b` `a68cbcc` builder, verbatim. Only the names are prefixed (`reference_`); checked with `diff`.
- New `tests/test_partM3a_scopes.py` (3 tests):
  - **24 random synthetic graphs.** Papers are the targets, authors are featureless, plus fields. They include multi-edges, self-loops (at least one on a target), cites pairs stored in both orientations, and isolated nodes. On each graph: every target's `ScopeMembers` with its own cells, and 3 random target sets (size 1..all) with their mini-batch WISP cells. Every field of `ScopeMembers` and `ScopeBatch` — including the batch `PreparedH2SData`, its merged `FaceGroup`s and `ScopeIndex` — is compared recursively with dtype, shape and `torch.equal`.
  - **The index** equals the per-node `((src==v)|(dst==v)).nonzero()` on every node of every graph.
  - **N6.** Two graphs each get their own index, and the index is built once. An evaluation-graph batch equals the reference on that graph, and an in-place change to the facts invalidates the cached index.

VERIFIED-RUN, 16721 step 19 (`a2_tests.txt`): `tests.test_partM3a_scopes` plus every Part MB / MB-fix module (`test_partMB_draws, _link, _model, _scopes, _trainer`, `test_partMBfix_flag, _functoriality, _order`) → **Ran 38, OK**. The Part MB/MB-fix tests are unchanged.

**Mutation** (index built without self-loop facts): `test_equal_to_the_reference_builder` **FAILED** with `AssertionError: Tuples differ: (torch.int64, (13,)) != (torch.int64, (14,))`. Restored identical, then OK.

**Cost** (informational, not a gate). Command `python ../docs/partM3a_evidence/a2_timing.py`, CPU, 6 threads, one synthetic graph with 100,000 nodes and 10⁶ facts, no cells:

| Measurement | Value |
|---|---|
| Index build (once per graph) | 54.4 ms |
| Reference, per target (50 targets) | **0.764 ms** |
| CSR, per target (200 targets) | **0.069 ms** (≈ 11×) |
| Facts equal on 10 targets | True |

## A3 — Mutations that reach the T1 residual on scopes (commit `9f3b638`)

Run against `tests/test_partMBfix_functoriality.py` only. Each mutation was applied to `src/models/map_operators.py` (`PathTransportOperator.__init__`, after `validate_face_orientation` at l.472 has run on the true signs) and restored in the job. `git diff` is clean afterwards; neither `src/` nor the test is changed.

VERIFIED-RUN, 16721 step 20 (`a3_mutations.txt`). Unmutated: OK.

| Mutation | Result | Residual | Worst incidence (scope, cell, hop) | Incidences |
|---|---|---|---|---|
| (i) the transport ignores the hop sign (`reversed_hop = None`: every hop forward) | **FAIL on the residual assertion** (`test_partMBfix_functoriality.py:99`) | **8.081** (T1_TOL 1e-5) | (1, 4, 1) | 48 |
| (ii) the near/far maps of one reversed hop swapped (the first reversed hop met keeps the forward maps) | **FAIL on the residual assertion** (l.99) | **7.703** | (0, 1, 0) | 48 |

Neither mutation is caught earlier: the test's own `validate_face_orientation` (l.55-57) and the operator's check both read the unchanged `face_edge_sign`.

## A3b — Stale DGL install messages (commit `0d08927`)

**`src/baselines/runtime/dependencies.py`** (messages only; the logic of the checks is unchanged):
- the `_dgl_cuda_defect` docstring (l.42-46);
- the defect message (l.62-63);
- the Linux `dgl_cuda_report` message (l.280-283).

All three now say: on Linux, DGL 2.4.0+cu124 is the conda package installed by `scripts/setup/install_env_b.sh` (URL and sha256 in `requirements/linux-lock-b.txt`), and pip cannot install it. `dgl-2.5.0+cu124`, DGL's wheel index and `--force-reinstall` are gone.

**`H2S-NN/CLAUDE.md`**, "Dependencies": the paragraph on the DGL pin now gives the same route. `AGENTS.md` has no such text (grep), so it is unchanged.

**Tests.** `tests/test_baselines.py` (stated test change):
- `test_cpu_only_dgl_on_linux_counts_as_not_installed` asserted `"--force-reinstall"`. It now asserts `scripts/setup/install_env_b.sh` and the absence of `--force-reinstall`.
- `test_report_explains_itself_on_every_platform` additionally asserts no `--force-reinstall`, and the script name on Linux.

VERIFIED-RUN, 16721 step 21 (g2n01, RTX 2080 Ti; `a3b_tests.txt`):
- `DglCudaGateTests` + `RequirementGroupTests` → **Ran 12, OK**.
- The real `dgl_cuda_report()` in `h2snn-b` → `(True, 'DGL has CUDA support')`.
- With a CPU-only DGL patched in on Linux, the new messages are printed.

## A4 — Full suite (this commit)

VERIFIED-RUN: `srun --jobid=16721 --overlap bash scripts/slurm/run_tests.sbatch` (step 22; g2n01, RTX 2080 Ti, i9-10900K, AVX2 with E, 6 threads; git `0d08927`; `a4_full_suite.txt`).

**`Ran 706 tests in 1078.160s — FAILED (failures=1)`**: the only failure is `test_baselines.BaselineNotebookTests.test_notebooks_are_committed_clean` (`AssertionError: 1 is not None`), as in CS8-5. There are 0 errors and 0 skips. The two golden tests and the float64 scope test are green.

Gate (step 23, `a4_gate.txt`): golden 5/5 OK; `mb0_reference.py compare mb0_reference_cluster.pt` → `compared 139 tensors; differing: none`, exit 0.

**Deviation.** The card allows a step in the author's allocation. The step had the allocation's 6 CPUs (the script's `#SBATCH --cpus-per-task=8` does not apply to a step). The bit-exact checks force 1 thread themselves.

## Assertions changed by W2 (old, new, reason)

| Test | Old | New | Reason |
|---|---|---|---|
| `tests/test_search_spaces.py::SweepConfigTests::test_link_prediction_spaces` | LP space with `decoder_type="dot"`/`"distmult"` == `COMMON` (+ `encoder_dropout` with features), with no `decoder_type`. `decoder_type="concat_mlp"` gave `COMMON + encoder_dropout + classifier_dropout`. `include_decoder_type=True` gave `decoder_type: {values: [dot, distmult, concat_mlp]}` | LP space == `COMMON` (+ `encoder_dropout`) + `decoder_type: {values: [dot, distmult]}`, without any flag. `decoder_type="concat_mlp"` raises `ValueError` (concat_mlp) | W2: the decoder is always tuned in {dot, DistMult}; `concat_mlp`, and with it LP's `classifier_dropout`, is removed; `--include_decoder_type` no longer exists |
| `tests/test_wandb_and_runs.py` (sweep-builder test, former `decoder_ablation`) | `build_sweep_config(..., decoder_type="concat_mlp", include_decoder_type=True)` holds `decoder_type` and no `classifier_dropout` | `build_sweep_config(..., decoder_type="dot")` holds `decoder_type == {values: [dot, distmult]}` and no `classifier_dropout` | as above (the flag and `concat_mlp` are gone) |
| `tests/test_search_spaces.py` module docstring | "LP decoder_type only on request (open)" | "LP decoder_type in {dot, distmult} on every LP task (spec v1.22 W2 …)" | as above |

Not a W2 change, but stated: `tests/test_baselines.py` `DglCudaGateTests` (A3b) — `"--force-reinstall"` was replaced by `"scripts/setup/install_env_b.sh"` plus a not-in check.

## Decisions for the author

| # | Decision | Options | Recommendation | Reason |
|---|---|---|---|---|
| 1 | The Colab notebooks (`h2snn_{link_prediction,graph_classification,node_classification}_colab.ipynb`) still append `--include_decoder_type` when `INCLUDE_DECODER_TYPE_IN_TUNING = True` (default False). The LP notebook still mentions `concat_mlp` | edit them (generator or notebook) in a later card / leave, since the cluster is the only run place | edit in M3b or a notebook card | With the default False nothing breaks; with True the tuner now exits on an unrecognised argument. Notebooks were outside A1's listed files |
| 2 | `concat_mlp` is still accepted by `src/configs/config.py:938` (config validation). A YAML with `decoder_type: concat_mlp` can still be run by `train_h2snn.py`/`run_h2snn_runs.py --config` (only the CLI `choices` refuse it) | also refuse it in `DataConfig` validation / keep, since the model code stays by decision | decide with the W2 spec wording | The card asked for refusal by the tuner and the run scripts' `choices`; the config path was not named. Refusing in config would also make the `concat_mlp` model code unreachable from configs |
| 3 | The tuner keeps `--decoder_type {dot, distmult}`. It sets the base config's decoder, which every trial then overrides | keep / remove | keep | Harmless; it only validates the base config |

## Out-of-scope findings (listed, not fixed)

1. **The `concat_mlp` model code** in `src/models/link_readout.py:48-54` and `:76-77` stays. Tuning and the run scripts cannot reach it (A1); a config file still can (Decision 2).
2. `src/baselines/runtime/_dgl_compat.py:37` (docstring) still names "`dgl 2.5.0+cu124` for torch 2.6" as the healthy Linux install. It is not in A3b's authorised list; stale wording only.
3. `scripts/run_graph_classification_repeated_cv_wandb_tuning.py:59` keeps a dead `--decoder_type` (no choices, read nowhere). Only `--include_decoder_type` was removed, as the card says.
4. Still open from CS7/CS8 and assigned to M3b: the graphbolt `GBWarning`/`PYTORCH_CUDA_ALLOC_CONF`, the dirty `vendor/H2GB` clone, `run_use_dataset.sbatch`.
5. **M3b items (card "Why a split", decisions log)**, none started here:
   - the 8 H2GB NC loaders, the OAG identity check, the venue-leak audit and PDNS's temporal loader;
   - the ogbn-mag P–P and H-Pokec U–U tasks with the large-LP epoch loop, including the check that each has one relation per node pair (`remove_held_out_facts` removes every fact of a held-out pair);
   - baselines' batch size and epoch through the H2GB loaders;
   - Hetero²Net MetaPath2Vec on MAG vs H2GB (M-path 3, 4).
6. A2's index is cached by tensor identity in a module-level dict. Entries are dropped when the `edge_index` tensor is freed, so memory stays bounded by the live graphs.

branch=part-m3a sha=<see final message> card=docs/H2S-NN_partM3a_card.md base=a68cbcc84042406705ca45c3db42684ef52daff1
