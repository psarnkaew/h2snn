# D11 pre-registrations (written before running; one section per unit)

## U1 — finish D7 (2026-11-07)
Change: `_base_config` pins `data.fact_layer=untied`, `path_sampler=dfs`, `resample_paths=false` for
every baseline bundle and rejects overrides of these keys; `DataConfig` defaults flip to
`tied` / `walk` / `resample_paths: true` (both the dataclass and `from_dict`).
Baseline path-builder check (inventory): no module in `src/baselines`, `scripts`, notebooks reads
`face_groups`/`FaceGroup` (grep) -> nothing needs the walk sampler.
Measurement: `u1_bundle_dump.py` dump (72 entries: ACM/DBLP/IMDB tensors + face groups, MUTAG edges)
taken at the pre-flip defaults (untied+dfs); expected after the flip through `build_bundle`: 0 mismatches.
Wrong if: any mismatch. Negative control = three-mutation check: removing each pin in turn makes the new
bundle-identity test fail (3/3), otherwise the test is vacuous.
Full suite expected: failure names identical to u0 (the 24 pre-existing test_baselines failures);
tests that relied on old defaults are expected to need an explicit `untied`/`dfs` in their config, not
a changed assertion.

## U2 — learning fix (T-F1)
Inventory: the optimizer is built only in `src/builders.py::build_parameter_groups` (a scalar group already
exists: `model.scalar_parameters()` = θ_η, θ_γ; lr = `optim.scalar_lr_scale` (10) × lr, wd 0). Consumers:
`build_optimizer` -> `build_trainer`/`build_*_trainer`; tests `tests/test_v2_model.py:994-1010` (asserts the
group has 2 params -> becomes 2 + gate-head params) and `:1248` (asserts `coupling_gate_init == 0.95` -> 0.6).
Change: `scalar_parameters()` also returns the listen/broadcast gate-head weights and biases (present heads
only); `model.coupling_gate_init` default 0.95 -> 0.6 (dataclass + from_dict). `coupling` default unchanged.
T-F1 (tests/test_cooperative.py): ACM-like toy (`_random_pap_data`) with labels = argmax of the 1-hop
neighbour-mean of the first 3 features (signal that needs diffusion), tied-free fixture, 50 full-batch epochs,
AdamW lr 1e-3, grad_clip 1. Moved := |η_50 − η_0| and |mean g_50 − mean g_0| and |mean b_50 − mean b_0| each
> 0.05 (mean over nodes of the gate on the current h0). Control: init 0.95 and ONE optimizer group for all
parameters (base lr, wd 1e-4): all three must be < 0.05. Expectation: fix moves all three; η is the marginal
one (D10 smoke moved η by 0.021 in 20 epochs at the same group lr). Wrong if: any fix quantity <= 0.05 or any
control quantity >= 0.05. Also checked: loss decreases (signal is learnable), so the test is not a noise artefact.
U2 attempt log (T-F1): attempt 1 (seed 0, statistic = change of the MEAN gate): fix moves eta 0.088, g 0.186, b 0.0458 -> b FAILS (<0.05); control 0.0009/0.0042/0.0027.
Diagnosis (VERIFIED-RUN, scratch tf1_diag): b head moves on all seeds (mean b 0.586->0.553->0.633 on seed 0; 0.61->0.78 and 0.61->0.71 on seeds 1,2) but its gradient sign oscillates, so the mean change cancels.
Attempt 2 (pre-registered before running): statistic = mean over nodes of |gate_v(50)-gate_v(0)| for g and b, |eta(50)-eta(0)|; seeds 0,1,2; fix must exceed 0.05 for every seed and quantity, control (init 0.95, one group) must stay below 0.05 for every seed and quantity.

## U3 - theory tests (T-F2, T-F3, T-F3b), tests/test_cooperative.py
T-F2: E from the production normalized operator, grad E by autograd, M^-1 grad E = (g/b) * grad E vs the production step (x - diffuse(x))/eta and vs a dense G Delta_dense B x; float64 <= 1e-10.
Controls: the swapped metric (b/g) and the identity metric differ from the step by > 1e-3. Expected: pass (algebra). Wrong if > 1e-10 or a control within 1e-3.
T-F3: 50 fixtures x 50 steps, eta in {0.25, 0.7, 1.0}: largest one-step increase of E <= 1e-12 (absolute, float64); eta = 3: some fixture's E increases by > 1e-6. Wrong if an increase > 1e-12 at eta <= 1, or no increase at eta = 3.
T-F3b: 30 fixtures (edges + signed mixed-length cells, random gates in [0.02, 1]); effective weight w(u<-v) = ||A_uv||_F / sqrt(d) (scalar factor of the block; A = (I - Phi)/eta from `diffuse`);
least-squares metric from log m_u - log m_v = log w(v<-u) - log w(u<-v) (gelsd); residual <= 1e-10, ||MA - (MA)^T|| / ||MA|| <= 1e-10, and the metric equals b/g up to a constant (<= 1e-8 relative).
Control available at U3: free per-direction block factors -> residual > 1e-3 on some fixture. The pair_attention control (required by the card) is added in U4b.
(U2 outcome) T-F1 after attempt 2: fix moves every quantity by >= 0.088 on seeds 0-2; control <= 0.013; mutation (gate heads outside the group) caught 2/2 (`u2_mutation_output.txt`).
(U3 outcome) T-F2, T-F3, T-F3b and the free-weights control passed at first run (4 tests, 288 s). Full suite after U2+U3 (one run, U3 adds tests only): 267 tests, failure names identical to the 24 pre-existing (`u3_full_suite.txt`).

## U4 - diagnostics (outcome; pre-registered expectation: logs contain per-step edge/path energy, eta, gamma, gate mean/min; kernel script runs on a trained model)
Smoke (ACM, 20 epochs, cuda, tied+walk, cooperative, defaults): `u4_smoke_metrics_summary.txt` - energy edge/path logged for each of the K+1 = 5 steps every 5 epochs (edge 0.956 -> 0.549, path 0.022 -> 0.013 at epoch 0);
eta 0.5025 -> 0.5087, gamma 1.006 -> 1.056, g mean 0.613 -> 0.431 (min 0.119), b mean 0.606 -> 0.670 (min 0.186): the gates move after the learning fix, eta barely.
Kernel script on that checkpoint (`u4_kernel_smoke.json`, k = 16, 60 LOBPCG iterations, 3 min on the GPU): smallest Ritz eigenvalue 0.014, dim ker (cut 1e-6) = 0 as an estimate (residual-norm check added afterwards; the smoke JSON predates it).
Scope notes: graph classification logs no per-step energy (per-batch operators); `pair_attention` has no Laplacian, so the trace is not defined for it.

## U4b - coupling: pair_attention (T-F4 ... T-F7, 3 mutations)
NOTE: this section was written into the file after the first test run of tests/test_pair_attention.py (the design and tolerances were taken from the card; nothing was changed after seeing results).
Design (src/models/pair_attention.py): GATv2 score a^T LeakyReLU(W [.]) (Eq. 7 of references/GATv2.pdf, opened and read: VERIFIED-SOURCE; bias omitted as the card specifies); single head; one W per entry kind (inputs differ in width), one shared a; hidden width = d;
LeakyReLU slope 0.2 (GAT default); embeddings rel_emb (num_relations x d_r), dir_emb (2 x d_r, shared-role relations use role 0 on both ends), pos_emb (max_len+1 x d_r), self_emb (d_r). Softmax per receiving node over {self} + {facts at u: 2 incidences per stored fact, a self-loop fact gives two} + {cell positions containing u}; log m_f added to cell scores.
Q_{u<-v}: orthogonal parts of the maps (F_u^-1 F_v with per-edge scale), cells via D9 (transport_sum / adjoint_prefix). Requires map_scale per_edge; config rejects per_side_scale, free_path_maps, endpoint/holonomy edges, uniform_path_gate, h0_only_readout with pair_attention.
DECISION (fairness of the A4 test): the attention parameters join the fast optimizer group (lr 10x, wd 0) exactly like the gate heads they replace; otherwise the test would handicap one arm. Flagged for the author.
Pre-registered expectations = the card: T-F4 rows > 0 and sum to 1 within 1e-12; T-F5 section fixed point within 1e-10 (tree, mixed-direction hops); T-F6 max node norm nonincreasing for eta in (0, 1] on 50 fixtures x 50 steps, control (unnormalized exp) increases on some fixture;
T-F7 operator equals the explicit matrix within 1e-10 (mixed lengths, signed hops, self-loops), state dict = D11's minus the 4 gate-head tensors plus the attention tensors exactly; T-F3b control: pair_attention residual > 1e-3 on some fixture.
Extra: independent loop-based softmax reference (roles, positions, shared roles, self-loops); merged weight 2 == two duplicate unit cells.
Mutations: M1 self outside the softmax -> T-F4/T-F5; M2 Q instead of Q^T on -1 hops -> T-F5; M3 log m_f omitted -> the duplicate-cell test.
Outcome: 13 tests pass (165 s); mutations 3/3 caught (u4b_mutation_output.txt).

## U5 - core-claim screen (written BEFORE any screen run, 2026-11-07)
Baseline precondition (u5_baseline_probe_output.txt): SeHGNN cannot run (vendor/SeHGNN is empty; clones are left alone per CLAUDE.md); CSNN cannot run (needs a C compiler, `where cl` fails); every other vendor clone except HINormer and csnn is absent; DGL cannot be imported by itself (graphbolt DLL) but the HINormer adapter runs on CPU: DBLP seed 0, published parameters, test macro-F1 0.933. Strongest baseline that runs = HINormer (WWW 2023; its paper evaluates DBLP, IMDB, Freebase, AMiner, not ACM). Reported as such; not a stop condition.
Runs: ACM, DBLP, IMDB x {full, ablation.no_path_cells (gamma = 0), ablation.endpoint_edges, ablation.h0_only_readout, ablation.coupling = pair_attention} x seeds 42, 43, 44; the fixed config configs/experiments/<dataset>_h2snn.yaml (identical across datasets except the name; d 32, K 4, 200 epochs, early stopping patience 100, tied + walk + resampling, cooperative gates init 0.6, scalar group 10x); nothing tuned; no early peeking at test metrics for any decision.
Primary metric: test macro-F1 at the epoch of best validation macro-F1 (trainer's best_test_metrics.json); micro-F1 reported alongside. Per (dataset, row): mean and sample std (ddof = 1) over the 3 seeds. Parameter count per row from a built model.
Comparison convention (fixed now): for rows a, b on a dataset, delta = mean_a - mean_b and s = max(std_a, std_b). "a beats b" iff delta > s; "a ~ b" iff |delta| <= s; "a loses" iff delta < -s.
Triggers of plan 12.6, operationalised:
 T1 "path cells do not beat gamma = 0 or endpoint_edges": full does NOT beat BOTH gamma = 0 and endpoint_edges on at least 2 of 3 datasets  -> STOP, report.
 T2 "full ~ h0_only (within 1 std)": full ~ h0_only on at least 2 of 3 datasets (reported; it routes to expressivity work, not a stop).
 T3 "path cells win and full > h0_only": not T1 and full beats h0_only on at least 2 of 3 datasets -> Tier 0 with 10 seeds (a recommendation to the author, not started here).
 T4 "gates still do not move after the learning fix": the mean over runs of max over epochs of |g_mean - init| and |b_mean - init| stays below 0.05 -> report (expected NOT to hit: the smoke run moved g mean by 0.18 in 20 epochs).
 T5 "pair_attention > full by > 1 std on >= 2 of 3 datasets": pair_attention beats full (convention above) on at least 2 of 3 datasets -> STOP, report (A4 in doubt; confirmation needs 10 seeds, the author decides).
 T6 "pair_attention <= full + 1 std": otherwise A4 retained (the row is reported as the test of A4).
Weakness stated up front: with 3 seeds a 1-std rule is weak evidence either way; the triggers decide what to do next (stop / confirm with 10 seeds), they do not establish the claims.
Logged per run (already in metrics.jsonl): eta, gamma, g/b mean and min per epoch, eta*lambda_max and normalized energy per step (edge, path) every 10 epochs (not for pair_attention / h0_only). One ACM model per variant (seed 42): analyze_kernel.py (dim ker, probe) where defined.
Expectations (not claims): the full model is above h0_only on ACM and DBLP (PAP/APA signal) and about level on IMDB (HGB IMDB is multilabel and heterophilic); pair_attention is competitive because outside evidence leans toward attention (UNVERIFIED here). I will be wrong if the screen contradicts these; the triggers above fix what that means.
