# H2S-NN v3-B — mini-batch design (reopened 2026-11-11; decisions of 2026-11-12)

**Status:** design document, author decisions of 2026-11-12 (chat, 05:55–07:44 JST). Not yet in the spec: the spec edits are drafted separately (`docs/spec_v1.17_draft_edits.md`, approved 2026-11-12 08:41; to be applied after v1.16, which another session drafted for the Part M decisions).
**Replaces for v3-B:** `docs/minibatch_design_decisions.md` (M0–M4, written for v2's Chebyshev filter) and `docs/minibatch_v3B_addendum.md`. Both stay as history. Every theoretical statement below was re-derived for the v3-B anchored solve; nothing is carried over from the v2 reasoning (receptive field of `K` applications, λ_max ≤ 2, filter gain, Props. 2–7, Lemma N do not apply to v3-B).
**Tags:** VERIFIED-CODE (file:line), VERIFIED-RUN (command and numbers), DERIVED, UNVERIFIED, PENDING (author).

---

## 1. Decisions
| # | Decision (author, 2026-11-12) | Reason |
|---|---|---|
| MB0 | **Local scope per target (option C).** Each target is predicted by the same model, with the same parameters, on its own scope: the target, its facts, the cells of the walks drawn from it. Degrees, relational context and normalisation are the scope's. The predictor is the sheaf of each scope, not the sheaf of the whole graph (§4) | Cheapest; prediction independent of batching; theory holds on each scope. Options A (whole-graph solve every step) and B (block solve with cached boundary values) were considered (§4.4) and not adopted |
| MB1 | **Training mode follows the source protocol:** HGB NC (ACM, DBLP, IMDB, Freebase), the six small LP tasks and GC are full batch / per graph; the eight H2GB NC datasets and the two large LP tasks are mini-batch. Freebase uses mini-batch only if full batch does not fit at the largest configuration (`d` 64, `ℓ_max` 5, `r` 64), stated in the paper. Mode is never a tuning value and never depends on a trial | H2GB trains every dataset with a sampler (§7); a per-trial mode would make the mode a hidden hyperparameter |
| MB2 | **Batch size is tuned on the eight H2GB NC datasets (option b; changes spec J4 "batch size not tuned"), range `{32, 256, 1024}`, the same for every method; on these datasets 1 epoch = 8,192 training targets** (confirmed 07:58). HGB NC stays full graph; LP keeps 8,192 positive links per step (HGB); GC keeps 1 (HetSheaf) | Author. The range is exactly the effective batches of H2GB's configs (32 × accumulation 1, 8, 32); the epoch keeps H2GB's data per epoch whatever the batch (§6) |
| MB3 | **Evaluation in mini-batches, metric over the whole validation/test split**, for every method. H2GB's own code evaluates a random subset (§7.2); this is documented so the author can revisit it | Standard practice (e.g. OGB evaluates all test nodes); the Macro-F1 rule of spec v1.14 is defined on the whole split |
| MB4 | **Large LP: the standard way.** Each endpoint node gets its own scope, `x*` is computed in batches, the decoder scores the pair `(x*_u, x*_v)` as in full batch. A node appearing in several links of a step is computed once (implementation detail) | Same model as NC; as PyG `LinkNeighborLoader` does |
| MB5 | **No memory design.** TSUBAME is available; a job that does not fit is fixed when it happens | Author |
| MB6 | (a) Evaluation warm start is **off** in mini-batch (solves start from the D-B7 initial guess); full batch unchanged. (b) The log-weight range stop condition is checked **per forward**, i.e. per batch in mini-batch (a batch holds tens of thousands of cells, e.g. 1,024 targets × 64 walks ≈ 65,000, so its range is close to the whole-graph range); no accumulation. (c) Spec Lemma R wording corrected (§4.3) | (a) §4.3 finding; (b) author asked for simplicity, and the per-batch check is close enough for a guard against weight blow-up; (c) §4.3 |

## 2. Example (ogbn-mag, target paper P1)
P1 has 20 facts (authors A1–A3, fields F1–F5, 12 citations). With `r = 4`, `ℓ_max = 3`, the walks from P1 give cells such as P1–A1–P9 and P1–F2–P20–A5. **Scope of P1** = P1, its 20 facts and their endpoints, and every node and fact of P1's cells (at most `r·ℓ_max = 12` cell facts). The unchanged model (encoder, context, maps, `g`, solve, head) runs on this small graph; the head reads `x*` of P1. A batch places several scopes side by side as one disjoint graph and solves them together. A scope has at most `deg(t) + r·ℓ_max` facts, whatever the size of the graph.

## 3. Scope and what is computed on it
1. Scope of target `t`: `t`; every fact of `t` (self-loop facts included) and its other endpoint; the cells of the walks drawn from `t`, with every node and fact on them. Nothing else (no induced facts between scope nodes).
2. It is a **lower set of the incidence poset** (spec 1.13): every face brings its facts and nodes, every fact its two nodes. Hence the §3 construction on it is a sheaf on a poset.
3. On the scope: `h₀` (featureless types read the embedding row of the **global** node id), `c` over the scope's facts, maps, `a_e`, `β_f = w_f m_f`, degrees `D_v, D_e, D_f`, elimination (`E_shared` = facts of the scope on ≥ 2 cells of the scope), `L̂`, the anchored solve, the head.
4. Cells are merged **per anchor**: `m_f` is summed over the walks of `t` only.
5. Batch = disjoint union of scopes; PCG inner products and stopping per scope (`scope_rows`).

## 4. Theory under v3-B (re-derived 2026-11-12)
### 4.1 What holds on a scope (DERIVED)
A scope is a graph with its own facts and simple-path cells, so every result of spec §4 that is stated for a graph applies to the scope's sheaf: Thm 1 (forced, functorial cell maps), Thm 3 (kernel = global sections **of the scope**), Thm 4 (existence of class-separating maps), Thm 5 (unique solution for all parameters), Thm 6 (consistent encodings **of the scope** are fixed points), Prop. 7/7′ (elimination), Thm 8, Prop. 10 (`cond(I+τŜ) ≤ 1+τ` per scope), Prop. 11, Thm 9/Prop. S per anchor (`E[L̂_{P,t}^{unnorm}] = Σ_ω p(ω|t) L_{f(ω)}`).
### 4.2 What does not hold
The scope's sheaf is **not** the restriction of the whole-graph sheaf: in P1's scope, author A1 sees only P1 and P9, so its context `c`, its degree and hence its maps differ from the whole graph. The mini-batch predictor therefore differs from the full-batch predictor; the difference is measured (C5, ACM and DBLP), not bounded. This is limitation L5.
### 4.3 Per-scope PCG: equal within the solver tolerance, not bit-identical (VERIFIED-RUN, correction)
The addendum and spec Lemma R said a batch is "bit-identical" to one scope at a time. Run of the repository's `pcg` (`src/models/anchored_solve.py`) on four SPD blocks (sizes 50/80/33/120, cond ≈ 73, 8 columns, tol 1e-4):
- batched iterations 38/40/32/42 = alone 38/40/32/42; solutions differ by 1e-6–7e-5 (relative), **in float64 as well**;
- cause: CG amplifies round-off (difference 4e-16 after 1 iteration, 2e-10 after 25, 7.6e-5 after 34, back to 1.6e-11 after 60);
- with one global inner product instead, each scope's error against the exact solution is 4e-5–1e-4 (the global criterion does not bound each scope).
Consequences: (i) Lemma R reads "equal to one scope at a time within the solver tolerance"; (ii) per-scope PCG is kept because it bounds **each target's** residual; (iii) on GPU `index_add` is not deterministic, which adds run-to-run differences of the same order; (iv) the evaluation warm start of `h2snn_model.py:1330` checks only the shape, so in mini-batch it could start one batch from another batch's `Y*` (same row count): off in mini-batch (MB6a). Scripts: session scratch `pcg_check.py`, `pcg_check3.py` (to be copied into the card's evidence folder).
### 4.4 Whole-graph options considered (not adopted)
- **Locality of the whole-graph solve (DERIVED, VERIFIED-RUN):** with `W = I − Ŝ` (`0 ≼ W ≼ I` by Prop. 10) and `ρ = τ/(1+τ)`, `(I+τŜ)⁻¹ = (1/(1+τ)) Σ_k ρ^k W^k`, so `|[(I+τŜ)⁻¹]_{tu}| ≤ ρ^{d(t,u)}` with `d` the distance in the graph of `Ŝ` (nodes adjacent when they share a fact or a cell). Checked: max ratio 0.991 ≤ 1 over 200 random graphs × τ ∈ {0.5, 1, 5, 50}. Any cut of the graph is an approximation whose size depends on the learned τ (τ = 1: ρ = 0.5; τ = 50: ρ = 0.98). The bound concerns the whole-graph geometry; it does not measure the scope approximation (whose `Ŝ` differs).
- **A. Whole-graph solve every step, loss on the batch:** exact whole-graph sheaf; no memory saving per step (all maps, cells and the full solve every step; ogbn-mag ≈ 70M state rows at `d` 64, `r` 64, estimate).
- **B. Block solve with cached boundary values (GNNAutoScale-type; Fey et al. 2021, recalled, not re-read):** `M` is SPD (Thm 5), so block Gauss–Seidel on the one convex energy converges to the whole-graph solution for fixed parameters (checked: relative error 0.87 / 0.003 / 4e-15 after 1 / 50 / 300 sweeps on a dense SPD test). Evaluation could be the exact whole-graph sheaf; training uses stale boundary values, degrees and context, and a truncated gradient.
- Author decision 2026-11-12: option C.

## 5. Walk draws
- Walks are drawn on the **whole graph** by the unchanged WISP sampler (exact action and unvisited-neighbour counts, so `p(ω|t)` is unchanged); the scope is cut afterwards.
- **Evaluation draw:** one draw from `p` with the run's base seed for the validation and test targets, fixed for the run (depends on `ℓ_max`, `r`, seed); `t`'s evaluation walks do not depend on the batching.
- **Training draw:** fresh draw for the batch's targets, seed from (base seed, epoch, batch index); `proposal: uniform`; prepared on the CPU in a worker.
- **Engineering (VERIFIED-CODE):** `sample_wisp_walks` (`src/data/wisp_sampler.py`) allocates `[W, A]` tensors for all walkers at once (H-Pokec at `r` 64: ≈ 100M walkers × 93 actions × 8 B ≈ 77 GB per tensor, estimate), so the draw is made in anchor chunks of a fixed size with a per-chunk seed from (seed, chunk index); results then depend only on (seed, chunk size, anchor order).

## 6. Batches, epochs and evaluation per task
| Task | Mode | Optimizer batch | Epoch | Budget | Evaluation |
|---|---|---|---|---|---|
| HGB NC (4) | full batch | whole graph (not tuned) | 1 step | ≤ 300, patience 30 | whole split, every epoch |
| H2GB NC (8) | mini-batch | **tuned** `{32, 256, 1024}`, same range every method | 8,192 training targets (`8192/B` steps) | 500 epochs, no early stopping, best validation epoch | whole split, every `eval_period` epochs of H2GB's config (4; OAG 2) |
| LP small (6) | full graph solve per step | 8,192 positive links | pass over training positives | 40, patience 40 | whole split |
| LP large (2) | mini-batch (MB4) | 8,192 positive links | pass over training positives | 40, patience 40 | whole split |
| GC (4) | per graph | 1 graph | pass over training graphs | as HetSheaf | as HetSheaf |

**H2GB NC (decided 2026-11-12, MB2).**
- Batch range `{32, 256, 1024}`: exactly the effective batches of H2GB's own configs (32 targets × accumulation 1, 8 or 32; §7.1).
- **1 epoch = 8,192 training targets** (H2GB: 256 iterations × 32 targets), i.e. `8192 / B` optimizer steps; 500 epochs = 4,096,000 target visits as in H2GB, whatever `B`. (Correction of my chat answer of 2026-11-12: "1 epoch = 256 steps" would make the data seen per epoch grow with `B`, up to 64× H2GB's at `B = 2048`.)
- Loss: mean over the `B` targets of a step (H2GB sums the micro-batch means; AdamW is nearly invariant to that scale).
- This epoch definition applies **only** to the eight H2GB datasets; HGB, LP and GC keep their own (table above).

## 7. What H2GB's code does (VERIFIED-CODE, `H2S-NN/vendor/H2GB`, read 2026-11-12; H2GB paper not checked on these points)
### 7.1 Training
- Every H2GB NC dataset uses a sampler (`sampler: hgt` in the RGCN configs of ogbn-mag, mag-year, oag-cs/eng/chem, IEEE-CIS, Pokec, PDNS), `batch_size: 32` seed targets, `train.iter_per_epoch: 256`: an "epoch" is 256 sampled batches (`LoaderWrapper`, `H2GB/sampler/custom_sampler.py:20-50`), not a pass over the training set (ogbn-mag: 500 epochs ≈ 4.1M target visits ≈ 6.5 passes over 629,571 training papers).
- Gradient accumulation `optim.batch_accumulation` (`H2GB/train/custom_train.py:154-159`, the micro-batch losses are summed, not averaged): RGCN configs ogbn-mag 32, mag-year 32, IEEE-CIS 32, PDNS 32, Pokec 8, OAG 1. Over all 224 configs that set it: 32 (113), 8 (34), 1 (77).
- AdamW, cosine schedule with 5 warm-up epochs, gradient clipping (differences from our shared protocol, already stated in spec 1.15).
### 7.2 Evaluation (the point the author asked to document)
- Validation and test loaders are built like the training loader, with the training batch size 32 (`H2GB/graphgym/loader.py:382-395`) and `shuffle=True` (`H2GB/main.py:100` calls `create_loader` with its default `shuffle=True`, `loader.py:352`).
- Each evaluation stops after `val.iter_per_epoch` batches (`LoaderWrapper`; the test loader also reads `cfg.val.iter_per_epoch`, `custom_sampler.py:339`), then re-creates the iterator. With `shuffle=True` the next evaluation therefore takes a **new random subset**; the code comment "Make sure we are always using the same set of data for evaluation" (`custom_sampler.py:42`) does not hold under shuffling. Neighbour sampling is also random at evaluation. The metric is computed on the concatenation of the predictions of the batches seen in that evaluation and the store is reset after each write (`H2GB/logger.py:142-146` `update_stats` appends, `write_epoch` `:163-213` concatenates and resets; the same pattern in `H2GB/graphgym/logger.py:279-363`).
- Size of the evaluated subset = min(split size, `val.iter_per_epoch` × 32): RGCN configs ogbn-mag 512 × 32 = 16,384 (of 41,939 OGB test papers, about 39 %; split sizes from OGB's official split, not re-counted here), mag-year 8,192, IEEE-CIS 8,192, PDNS 8,192, Pokec 16,384, OAG 4,096. `val.iter_per_epoch` also differs between models of one dataset (ogbn-mag: ACM-GCN 256, RGCN and most others 512).
- Evaluation every `eval_period` epochs (4; OAG 2; `custom_train.py:272-283`).
- **Our rule (MB3):** every method is evaluated on the whole validation and test split, in batches, with the fixed evaluation draw for H2S-NN. Published H2GB numbers are therefore not computed on the same node set (they are not reused anyway, spec 1.5). If the author later prefers H2GB's subset, the change is: per evaluation, a random subset of size min(split, `val.iter_per_epoch` × 32) per dataset, the same for every method.

## 8. Large LP (MB4)
- Each endpoint `u` of a positive or negative link of the step is a target with its own scope (anchors of J1 = the endpoint types; only the endpoints of the step's links are computed).
- Score = decoder(`x*_u`, `x*_v`) as in full batch; BCE with 1:1 training negatives; 8,192 positives per step, 40 epochs, patience 40.
- Training positives stay in the message graph (as full batch and HGB); held-out facts are removed in both orientations before any draw.
- HGB 2-hop test negatives and validation negatives are generated once on the cluster (preprocessing).
- Evaluation: all test and validation links; node scopes computed once per evaluation and shared by the links.

## 9. Stop conditions and solver
- Tolerance schedule 1e-3 for the first 10 epochs, then 1e-4 (epoch as defined per task, §6); `T_max = 100` stops (per scope).
- `τ > 50` stops (τ is global). Log-weight range > 30 stops, checked per forward (per batch) (MB6b).
- Evaluation warm start off in mini-batch (MB6a).

## 10. Code gaps (VERIFIED-CODE on `part-m`, 2026-11-12)
1. No scope builder (cut, relabel, per-anchor merge, `scope_rows`).
2. `TypeSpecificNodeEncoder` reads embedding rows by position in the input (`src/models/node_encoder.py:101-104`): must receive global ids.
3. The model never passes `scope_rows` to the solver (only `anchored_solve.py` supports it).
4. Sampler: chunked draw (§5); per-anchor merge needs the anchor in the merge key (`merge_wisp_cells` merges across anchors).
5. Trainer: one optimizer step per epoch, full batch (`src/trainers/trainer.py`); needs the mini-batch loop, batch-wise evaluation, epoch = 8,192 targets on H2GB.
6. Warm start guard (`h2snn_model.py:1330`).
7. Baselines: the shared batch range and epoch definition apply to every method on H2GB (adapters).

## 11. Tests for the card (tolerances fixed when the card is written)
| Property | Test |
|---|---|
| Scope is a lower set | every cell's facts and every fact's nodes present; functoriality on scope cells |
| Batch = scopes alone | float64, tol 1e-10: batched vs one scope at a time, max relative difference ≤ 1e-8; production tol 1e-4: per-scope residual ≤ tol |
| Prediction independent of batching | evaluation with two batch sizes and shuffled order: equal within the solver tolerance |
| Whole-graph harness | a scope built from all facts and all anchors' walks merged as in full batch reproduces full-batch logits within tolerance (plumbing check) |
| Per-anchor unbiasedness | Monte-Carlo mean of `L̂^{unnorm}_{P,t}` vs exact enumeration on a small graph |
| Global ids | mutation: local id → targeted test fails |
| Evaluation draw fixed | `t`'s walks identical across batchings and epochs |
| Chunked sampler | draw identical for a fixed chunk size; `p(ω|t)` unchanged |

## 12. Spec edits (`docs/spec_v1.17_draft_edits.md`)
R7 and L5 wording (scope = lower set; same model, scope geometry); Lemma R ("within the solver tolerance"; Thms 1–9 listed on the scope sheaf; not the restriction of the whole-graph sheaf); §6 training mode per dataset (MB1), batch size tuned on mini-batch tasks with the shared range and the H2GB epoch definition (MB2, changes J4), evaluation on whole splits (MB3), large LP (MB4), stop condition per forward and warm start (MB6); solver settings ("per-scope inner products and stopping in mini-batch", unchanged).

## 13. Open
- Whether Freebase NC fits full batch at the largest configuration (cluster measurement).
- Baseline adapters on H2GB: their own samplers stay (part of each method); the shared batch range and epoch definition need to be expressible in each adapter (coverage matrix of the baseline card).
