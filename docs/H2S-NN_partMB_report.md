# Part MB report — mini-batch training and evaluation of H2S-NN (scopes)

**Card:** `docs/H2S-NN_partMB_card.md` (MB0–MB6). **Design authority:** spec v1.21 (Lemma R, L5, R7, §6), `docs/minibatch_design_v3B.md`, decisions MB0–MB6 (2026-11-12) and N1–N6 (2026-11-13). **Branch:** `part-mb` from `part-pfix` at `203fbbf89b13a69be73b1f1e32d29a4e566b8852` (`git rev-parse HEAD` checked). **Evidence:** `docs/partMB_evidence/`.
**Rule kept throughout:** correctness only (unit tests, oracle tests on tiny synthetic graphs, mutations, the full suite once). No H2GB data exist here; every test is synthetic. No metric reported, no comparison.

---

## MB0 — Preconditions (commit `6e4488d`)

**Code gaps of `minibatch_design_v3B.md` §10, re-checked on `203fbbf` (VERIFIED-CODE). All seven are open:**

| # | gap | where (on `203fbbf`) | closed by |
|---|---|---|---|
| 1 | no scope builder | `src/data/scopes.py` does not exist | MB1 |
| 2 | featureless rows read by position | `src/models/node_encoder.py:101-104` (`mask.nonzero() - type_offsets`) | MB1 |
| 3 | the model never passes `scope_rows` | `src/models/h2snn_model.py:1341-1342` (`anchored_solve(...)` without `scope_rows`); the solver supports it (`anchored_solve.py:129-186, 233-241`) | MB3 |
| 4a | no chunked draw | `src/data/wisp_sampler.py` `sample_wisp_walks`: all walkers at once | MB2 |
| 4b | merge without the anchor in the key | `wisp_sampler.merge_length_group` key = (signature, facts) | MB2 |
| 5 | trainer: one step per epoch, full batch | `src/trainers/trainer.py:209-233` | MB4 |
| 6 | warm start checks the shape only | `h2snn_model.py:1336-1338` | MB3 |
| 7 | baselines on H2GB (batch range, epoch) | adapters | **out of scope (N1, M3 card)** |

**Layout of the solved system (VERIFIED-CODE, needed by `scope_rows`):**
- rows = nodes, then kept fact states, then cells;
- kept facts = `cells_per_fact ≥ 2`, in ascending fact id (`sheaf_energy.py:478`);
- cells are numbered across groups by `cell_offsets` (`sheaf_energy.py:57-110`).

**Reference outputs before any change (VERIFIED-RUN):** `docs/partMB_evidence/mb0_reference.py` saves `mb0_reference.pt`, with 139 tensors:
- full-batch WISP draws, merged into groups: `test_wisp` fixture, `ℓ_max` 2/3/4, with and without an adaptive proposal;
- the full-batch draw of a synthetic ACM-shaped graph with one featureless type;
- the full-batch anchored model on that graph: two consecutive evaluations (the second warm-started), one training step's loss and every parameter gradient, and `h₀`.

`compare` on a second run → `compared 139 tensors; differing: none`. Each later unit is compared against this file.

## MB1 — Scope builder and global node ids (commit `20522b3`)

**Change (VERIFIED-CODE).**
- New `src/data/scopes.py`:
  - `scope_members(graph, t, cells)` is the target, its facts (self-loops included) and their endpoints, and its cells' nodes and facts. Nothing else is added.
  - `build_scope_batch(graph, targets, cells)` returns a `ScopeBatch` holding:
    - the batch graph: a `PreparedH2SData`, the disjoint union of the scopes, with rows copied per scope and `target_indices` = the targets' rows;
    - `global_node_id` and `global_fact_id`;
    - `node_scope`, `fact_scope`, and `cell_scope` per group;
    - `target_rows`.
  - The graph is an argument (N6). Cells keep their fact orientation, and signature tables are merged per length.
- `PreparedH2SData.global_node_id` (optional; `None` = the current behaviour).
- `TypeSpecificNodeEncoder.forward(…, global_node_id)` reads a featureless row by global id, which closes gap 2. `H2SNN.encode_nodes`/`encode`/`forward` thread it through.

**Tests (VERIFIED-RUN).** `tests/test_partMB_scopes.py` (5) on the hand §2-style graph. `tests.test_partMB_scopes tests.test_task_heads tests.test_v3b_model` → **Ran 12, OK** (`mb1_tests_output.txt`):
- P1's scope equals the hand lists: nodes 0,1,2,3,6,7,8,10; nine facts; cells P1–A6–P1′ and P1–F10–P3–A7;
- lower set; the induced fact `cites(1,3)` (both endpoints in the scope) is absent; every batch fact is a copy of its graph fact;
- `validate_face_orientation` holds on the scope cells;
- author A6, featureless and copied into two scopes, gets `h₀` equal to the whole graph's row;
- a scope built on a second graph uses that graph's facts.

**Mutations (VERIFIED-RUN, `mb1_mutations.txt`).**
- Local position instead of the global id → the `h₀` test **errors** (negative embedding row).
- Induced facts added → **3 failures**, including the no-induced test.

**Full-batch bit-identity (VERIFIED-RUN, `mb1_bit_identity.txt`):** `compared 139 tensors; differing: none`.

## MB2 — Chunked draw and per-anchor merge (commit `6ff21a2`)

**Change (VERIFIED-CODE, `src/data/wisp_sampler.py`).** Additions only. The full-batch functions keep their code path; `merge_length_group` calls the shared `_merge_cells` with no anchor.
- `WISP_CHUNK_ANCHORS = 4096` (N2).
- **`chunk_seed(seed, k) = seed · 2²⁰ + k`**, the one stated formula. It is distinct for every k < 2²⁰ (4.3e9 anchors) and raises beyond.
- `sample_wisp_walks_chunked` runs `sample_wisp_walks` per chunk with its own generator.
- `merge_wisp_cells_per_anchor`: key = (anchor, signature, facts), `m_f` over that anchor's walks only.
- `AnchorCells.of(anchor)` gives a scope's cells; `build_wisp_anchor_cells` makes the mini-batch draw on the whole given graph.
- This closes gaps 4a and 4b.

**Tests (VERIFIED-RUN).** `tests/test_partMB_draws.py` (5) on the `test_wisp` fixture. With `tests.test_task_heads tests.test_wisp` → **Ran 58, OK** (`mb2_tests_output.txt`):
- the same seed and chunk size give an identical draw;
- every walk's `log p` equals the exact enumerated p (12 places), and the walk frequencies of 1e5 walks from anchor 0 are each within 4σ of enumeration;
- chunk seeds are distinct (1,000 chunks), and anchor 0 planted in two chunks does not repeat its walks;
- one instance drawn from anchors 0 and 1 gives one cell (m = 0.75) in the full-batch merge and two cells (0.5 and 0.25) in the per-anchor merge;
- each anchor's mass is (its accepted walks)/r.

**Mutations (VERIFIED-RUN, `mb2_mutations.txt`).**
- Anchor dropped from the key → **2 failures** (the per-anchor tests).
- Chunk seed ignoring the index → the distinct-chunk test **fails**.

**Full-batch bit-identity (`mb2_bit_identity.txt`):** 139 tensors, none differing; `test_task_heads` and `test_wisp` pass.

## MB3 — The model on a batch of scopes (commit `b205ef0`)

**Change (VERIFIED-CODE).**
- `ScopeIndex` (scope of every node, fact and cell, plus the number of scopes), carried as an optional `PreparedH2SData.scope_index` and set by `build_scope_batch`.
- `H2SNN._encode_anchored` builds `solved_system_scope_rows`, the scope of every row of the solved system: nodes, then kept fact states (facts on ≥ 2 cells, ascending id), then cells in group order. It passes them to `anchored_solve(scope_rows=…, num_scopes=…)`, which closes gap 3.
  - Per-scope inner products, stopping and `T_max` were already in `pcg` (`anchored_solve.py:139-200`).
  - Context, degrees, normalisation and elimination are per scope because no fact crosses scopes. Checked in (e), not assumed.
- Warm start:
  - **off in mini-batch** (no use, no storage);
  - in full batch, used only when the evaluation draw is the same one (`_draw_key`/`_same_draw`: edge index plus every group's nodes, facts, signs and `m_f`; identity, else `torch.equal`), not only the same shape. This closes gap 6.
- The log-weight range was already checked per forward (`h2snn_model.py`, `_encode_anchored`), so in mini-batch it is per batch. Unchanged.
- Full batch is unchanged: `scope_index = None` gives `scope_rows = None`.

**Tests (VERIFIED-RUN).** `tests/test_partMB_model.py` (6), synthetic ACM-shaped graph, mini-batch draws merged per anchor. With `tests.test_partMB_scopes tests.test_partMB_draws tests.test_task_heads tests.test_v3b_model tests.test_anchored_solve` → **Ran 25, OK** (`mb3_tests_output.txt`):
- **(a)** float64, tolerance 1e-10: every target's `x*` from a batch of 5 scopes of different sizes equals the scope solved alone within 1e-8 relative.
- **(b)** float32, tol 1e-4: every scope's own relative residual ≤ 1e-4, computed by the test from the captured system with its own row scopes.
- **(c)** logits with batches of 3 and of 7, and in a shuffled order, agree within 1e-8 relative (float64, tol 1e-10).
- **(d)** the whole graph as one scope (global ids `arange`, every scope id 0) reproduces the full-batch logits within 1e-8.
- **(e)** the relational context `[h₀, c]` and the per-row Jacobi scalars `1/(1+τ d_v)` of scope 1's nodes inside a batch equal those of scope 1 alone (≤ 1e-12 relative).
- **(f)** spy on `y0`: two mini-batch evaluations of the same batch both start from the D-B7 guess (`None`). In full batch the second evaluation warm-starts, and an evaluation with another draw of the same shapes does not.

**Mutations (VERIFIED-RUN, `mb3_mutations.txt`).**
- `scope_rows` = None (one global inner product) → (b) **FAILED**: `scope 2: relative residual 5.92e-01`.
- Warm start re-enabled in mini-batch → (f) **FAILED**.

**Deviation in my own test design (stated).** My first two versions of (b) did not catch the global-inner-product mutation:
- (i) scaling one scope's input up or down. CG's iterates are one polynomial in the operator for every block, so a block's *relative* residual does not depend on its scale.
- (ii) scaling the right-hand side of all but the hardest scope by 1e6. The global CG takes a different iterate path, and with 6 vs 7 iterations per scope no scope stayed above tolerance.

The final test builds the case a global criterion cannot handle:
- the other scopes' right-hand sides are ×1e6 and start at their (scaled) per-scope solution;
- the hardest scope starts at the D-B7 guess.

One global norm is then satisfied at once, while per-scope stopping still iterates the hardest scope to its own tolerance. Only the right-hand side and the start are changed, through a spy on `solve_anchored`; the operator and `g` are untouched. A first attempt that scaled `h₀` itself also changed `g` and triggered the log-weight stop condition, so the scaling was moved to the solve. No tolerance was relaxed.

**Full-batch bit-identity (`mb3_bit_identity.txt`):** 139 tensors, none differing, including the warm-started second evaluation.

## MB4 — Mini-batch trainer for node classification (commit `01ee220`)

**Change (VERIFIED-CODE).**
- `src/configs/config.py`:
  - **Constants:** `TRAINING_MODES`, `MINIBATCH_BATCH_SIZES = (32, 256, 1024)`, `MINIBATCH_EPOCH_TARGETS = 8192`, `EVAL_BATCH_TARGETS = 1024`, `MINIBATCH_ANALYSIS_DATASETS = (acm, dblp)`, `LARGE_LP_DATASETS`; `default_training_mode(task, dataset)` and `default_eval_period(dataset)` (H2GB 4; OAG 2; 1 for the ACM/DBLP flag).
  - **`DataConfig` fields** `training_mode`, `batch_size`, `epoch_targets`, `eval_period`, resolved per dataset:
    - the mode is the dataset's; any other mode raises, except `minibatch` on ACM/DBLP (N4);
    - in mini-batch, `batch_size` is required, `epoch_targets` defaults to 8,192 and must be a multiple of B, and `eval_period` defaults per dataset;
    - in full batch, the three mini-batch fields raise.
  - **Tuner** (`reject_never_tuned_settings`): it refuses the flag outside the H2GB datasets, an `epoch_targets` value, and `training_mode`/`epoch_targets`/`eval_period` as search keys. It allows `batch_size` only on the H2GB NC datasets.
- New `src/trainers/minibatch_trainer.py`:
  - `TrainingOrder`: a seeded permutation per pass, `pass_seed = base·100,003 + pass`, continued across epochs, crossing pass boundaries.
  - `steps_per_epoch = epoch_targets / B`; one step per batch; loss = the batch mean.
  - Training draw per batch with **`training_draw_seed = (base·100,003 + epoch)·65,536 + batch`**, drawn on the batch's distinct targets (chunked, per-anchor merge) and cut into scopes on the training graph.
  - Evaluation every `eval_period` epochs and after the last: one evaluation draw per run (base seed) for all validation and test targets, whole splits in batches of 1,024, metric over the concatenation.
  - Checkpoint at the best validation evaluation; stop conditions raise and end the run (`logged_stop_conditions`); `fit(graph, eval_graph=None)` cuts evaluation scopes from the evaluation graph (N6).
- `build_dataset` builds no whole-graph draw in mini-batch. `build_experiment_components` returns the mini-batch trainer when `training_mode == minibatch`; the full-batch path is unchanged. This closes gap 5.

**Tests (VERIFIED-RUN).** `tests/test_partMB_trainer.py` (8), synthetic ACM-shaped graph with the analysis flag. With `test_partMB_model`, `test_search_spaces`, `test_shared_protocol`, `test_partO_literature_row`, `test_partO_reload`, `test_wandb_and_runs` and `test_task_heads` → **Ran 63, OK** (`mb4_tests_output.txt`):
- **Steps.** With `epoch_targets = 1024`, an epoch takes 32/4/1 steps for B = 32/256/1024, and no whole-graph draw is built.
- **Training order.** 5 passes of 18 targets: each pass is a permutation, the order is deterministic per seed, differs per pass and differs for another seed.
- **Draw seeds.** 12,800 (epoch, batch) seeds are distinct, and the trainer uses exactly `training_draw_seed(3, e, b)`.
- **Whole split.** The evaluated ids equal the split's targets (count and order), and the metric equals `compute_classification_metrics` on the concatenated predictions.
- **Batch-size independence.** Float64, tol 1e-10: evaluation logits with batches of 1,024 and of 3 agree within 1e-8 relative.
- **Stop condition.** A `LogWeightRangeStopError` raised in the second batch ends `fit` with that exception.
- **Mode and tuner.** The H2GB datasets resolve to mini-batch, the HGB ones to full batch; the flag is refused on IMDB, and `batch_size` in full batch. The tuner refusals hold, and `batch_size` is allowed on ogbn-mag.
- **N6.** The evaluation scopes are cut from the evaluation graph passed in (a second synthetic graph).

**Mutations (VERIFIED-RUN, `mb4_mutations.txt`).**
- Epoch = 4 steps regardless of B → **FAILED** for B = 32 and B = 1024. With 4 steps, B = 256 coincides with the correct count.
- Evaluation on a random half of the split → **FAILED**: the whole-split test, and the batch-size test (a different subset each call).

**Full-batch bit-identity (`mb4_bit_identity.txt`):** 139 tensors, none differing.

## MB5 — Large link prediction (commit `c45875b`)

**Change (VERIFIED-CODE).** New `src/trainers/minibatch_lp.py`:
- `remove_held_out_facts` removes every fact whose pair is a held-out pair, in both orientations, before any draw.
- `LinkScopes`:
  - `cells`: one draw with anchors = the step's (or the evaluation's) endpoints only.
  - `representations`: each distinct endpoint solved once, on its own scope, giving `[x*, h₀]`.
  - `score`: the model's `link_readout` on `(x*_u, x*_v)`, as in full batch.
  - `train_step`: positives and 1:1 negatives, BCE.
  - `evaluate`: one draw, every endpoint's scope once, every link scored, endpoints in batches of 1,024.
- `LP_POSITIVES_PER_STEP = 8192`. Epochs and patience (40/40) come from the existing HGB LP budget.
- **Not done (stated):** the epoch loop over a registered large-LP dataset. The two tasks have no data or loader (registry "data pending"), so `LinkScopes` is the per-step and per-evaluation machinery that loop will call. The loop belongs with the loaders (cluster / M3 card).

**Tests (VERIFIED-RUN).** `tests/test_partMB_link.py` (4), synthetic same-type graph (24 users, some facts stored in both orientations). With `tests.test_full_graph_early_stopping tests.test_task_heads` → **Ran 12, OK** (`mb5_tests_output.txt`):
- Float64, tol 1e-10: link scores from a batch of endpoint scopes equal those from each endpoint's scope solved alone within 1e-8 relative.
- Node 0 in three links is solved once: 4 distinct endpoints, 4 representation rows; the spy sees each endpoint once per solve.
- With 4 two-way and 4 one-way held-out pairs removed, no scope of any of the 24 endpoints holds a held-out pair in either orientation.
- Evaluation returns one finite score per link.

**Mutation (VERIFIED-RUN, `mb5_mutation.txt`).** Held-out facts removed in one orientation only → the leak test **FAILED**.

**Full-batch bit-identity (`mb5_bit_identity.txt`):** 139 tensors, none differing.

## MB6 — Full suite and report

**Full suite, run once (VERIFIED-RUN).** `python -m unittest discover -s tests` from `H2S-NN/` on `c45875b` → **Ran 667 tests in 2,010 s; FAILED (failures=2, errors=2, skipped=7)** (`mb6_full_suite_output.txt`; INFO lines of the synthetic fixture trainings removed).
- **Failures.** Exactly the four known `test_baselines` failures: `VendorIsolationTests.test_all_declared_vendor_clones_are_present`, `VendorIsolationTests.test_colliding_vendor_module_names_do_not_leak`, `ReproducibilityTests.test_vendor_model_init_is_reproducible_under_seed_everything`, `BaselineNotebookTests.test_notebooks_are_committed_clean`.
- **Skips.** The 7 are the known environment skips.
- **Full-batch golden tests** (`test_task_heads`) pass. The MB0 reference outputs were compared after every unit: 139 tensors, none differing.
- **Test count.** 639 (Part P-fix) + 5 (MB1) + 5 (MB2) + 6 (MB3) + 8 (MB4) + 4 (MB5) = **667**.
- **No existing test assertion was changed by this card.**
- No stop condition was met. Full batch never changed; MB3 (a) held at 1e-8 with tolerance 1e-10; no change to the energy, maps, `g`, the solver or WISP's `p`.

## Decisions for the author

| ID | Question | What I did | Alternative |
|---|---|---|---|
| D-MB1 | Seed formulas (the card fixes "seed from …" without a formula) | chunk: `seed·2²⁰ + k`; training-order pass: `base·100,003 + pass`; training draw: `(base·100,003 + epoch)·65,536 + batch`. Each is distinct within its stated range and raises outside it | another stated formula (only reproducibility depends on it) |
| D-MB2 | A training target repeated within one batch (a batch crossing a pass boundary) | The draw is made once per distinct target, and each occurrence gets its own scope copy with those cells, so the target counts twice in that step's mean loss | De-duplicate within a batch, which would make the batch smaller than B |
| D-MB3 | Large-LP epoch loop | Not written: no data or loader. `LinkScopes` provides the step and the evaluation | Write the loop with the M3 card's loaders |
| D-MB4 | Early stopping in mini-batch | `bad_epochs` counts epochs between evaluations (`+= eval_period`) against `early_stopping_patience`. On H2GB the budget has no early stopping (patience = epochs) | Count evaluations |

## Out-of-scope findings (not fixed)

1. **Baseline adapters on H2GB** (gap 7): batch range and epoch are not expressible in the adapters yet (N1, M3 card).
2. **Freebase:** full batch, as fixed; whether it fits at the largest configuration is a cluster measurement.
3. **New `DataConfig` fields.** `training_mode`, `batch_size`, `epoch_targets` and `eval_period` now appear in every saved `config.yaml` (`full_batch`, 0, 0, 0 in full batch). The W&B resume fingerprint (`canonical_fingerprint`) of an existing tuning state therefore changes. No campaign has started, so nothing is lost, but an old state would be treated as incompatible.
4. **CUDA.** On GPU, `index_add` is not deterministic (design §4.3 iii), so (a)/(c) hold only within the solver tolerance there. All tests here ran on CPU.
