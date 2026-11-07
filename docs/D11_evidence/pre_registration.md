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
