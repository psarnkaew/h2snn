# G3b — end-to-end functional check: pre-registration

Written 2026-11-08, before any G3b run (after G1/G2 tests, before G3's measurement). Training is
allowed; it checks that the new parts work, not predictive performance. Accuracy is logged only,
never reported as a result. Nothing below is changed after a result; additions get a dated heading.

## (a) Real data
**a1 Smoke, ACM, seed 42, 5 epochs, defaults** (`filter_bank`, `geometry_context: true`; config
`configs/experiments/acm_h2snn.yaml`, `epochs: 5`, `log_every: 1`), in-process with the production
`build_optimizer` / `build_trainer` (`docs/D12_evidence/g3b_smoke.py`). Pass iff all of:
1. every logged loss and metric finite (no NaN/Inf); train loss of epoch 5 < train loss of epoch 1
   (the logged training loss of epochs index 4 and 0);
2. filter diagnostics every epoch: `filter_gain ≤ 1 + 1e-5` (float32 power iteration, 30 steps),
   `filter_norm_sum ≤ 1 + 1e-6`, `lambda_max_A ≤ 2`;
3. every `W_k` moves: `‖W_k(end) − W_k(init)‖_F / ‖W_k(init)‖_F > 1e-3` for every k;
4. gates and γ move: `|γ(end) − γ(init)| > 1e-4`; each gate head's parameters relative change > 1e-3
   (mean gate shift reported);
5. gradients reach the context inputs: on the first training step `‖∂L/∂c‖ > 0` and finite (forward
   hook on `model.relational_context`, `retain_grad`);
6. a reloaded checkpoint reproduces the logits exactly: final `state_dict` saved with `torch.save`,
   loaded into a freshly built model; eval-mode logits on CPU `torch.equal` (GPU max |diff| reported).
**a2 DBLP, 2 epochs, defaults**: runs without error; the data contain path lengths 2 and 4 (APA,
APTPA); losses finite.
**a3 ACM, 2 epochs, every `V_k` initialised at 2 × its D11 value** (Σ‖V_k‖₂ = 2, cap active from the
start), manual loop with the production optimizer: every gradient finite at every step;
`filter_gain ≤ 1 + 1e-5` and `filter_norm_sum ≤ 1 + 1e-6` after each epoch.
**a4 Comparison switches, ACM, 2 epochs each** (`scripts/train_h2snn.py`): `encoder: first_order`,
`geometry_context: false`, `coupling: symmetric`, `no_path_cells`, `endpoint_edges`,
`h0_only_readout`, `coupling: pair_attention`, `identity_maps` (+ `no_path_cells`, required). Pass:
exit code 0, `best_test_metrics.json` written, losses finite.

## (b) Synthetic known-answer task (`docs/D12_evidence/g3b_synthetic.py`)
Graph: C = 4 classes, 16 venues (featureless; class of venue v = v mod 4), 600 target papers. For
each i: author `a_i` (16-dim N(0,1) noise features), target paper `t_i` and carrier paper `q_i`
(16-dim N(0,1) noise features), facts `(a_i, writes, t_i)`, `(a_i, writes, q_i)`,
`(q_i, published_in, V_i)`, `V_i` uniform. Label(`t_i`) = class(`V_i`); carriers are unlabelled.
`t_i` has no venue edge: the label is reachable only through the typed path t–a–q–V (3 hops).
Stored edges in both directions → production tied fact layer; path cells from the production walk
sampler (`resolve_schemas` + `build_schema_path_groups`) with schemas PAP and PAPV, fixed draw
(seed = run seed, no resampling). Split of targets 60/20/20 (fixed seed 0). Model block of
`acm_h2snn.yaml` (d 32, K 4, …), production trainer, up to 200 epochs, early stopping patience 100,
selection on validation accuracy. Seeds 42, 43, 44 (the graph is tiny; 3 seeds make the pass
criterion robust to one lucky run).
Arms: full (defaults), `h0_only_readout`, `encoder: first_order`, `no_path_cells`.
**Pass iff** full test accuracy ≥ 0.90 on every seed and `h0_only` test accuracy ≤ 0.35 on every
seed (chance 0.25). first_order and no_path_cells are reported, not gated.
**If full fails**: diagnose before anything else — (1) train accuracy (fit vs no fit); (2) coverage:
fraction of test targets with at least one PAPV cell; (3) reach without cells (no_path_cells result);
(4) gradient to the venue embeddings — classify as bug vs insufficient reach, with evidence.
Expected: full ≥ 0.9, first_order ≥ 0.9 (same cells), h0_only ≈ 0.25, no_path_cells uncertain
(3 hops within K = 4 applications, through degree normalisation).

## Addendum 2026-11-08: float64 re-check of a1 check 2 (author decision 1 on G1–G3b)
Written after the G3b results and before the re-check run. a1 check 2 failed under the
pre-registered float32 measurement (`filter_norm_sum` up to 1.0000017 > 1 + 1e-6). Decision: the
tolerance 1 + 1e-6 is kept. The logged norm-sum diagnostic now casts the d×d blocks `W_k` to
float64 before taking the spectral norms (`ChebyshevFilterBank.diagnostic_norm_sum`, used by both
logging paths). The cap in `weights()` is unchanged, so training is unchanged.
Re-check: a1 and a3 are rerun exactly as pre-registered above (same config, seed 42, same script
with `--tag f64`, fresh experiment directory `g3b_smoke_acm_f64`, output `g3b_smoke_f64.json`). Every
criterion and tolerance is unchanged; check 2 now reads the float64 diagnostic. It is reported as
"FAIL under the pre-registered float32 measurement; float64 re-check: <result>". The original a1
result stays recorded as FAIL.
Found while preparing the re-check: the original a1 checks were evaluated over 10 metrics rows, not
5. The logger appends to `metrics.jsonl`, and the directory held the rows of both attempts: the
first, which crashed with a device mismatch after training, and the second, completed one. Their
losses are identical; `filter_norm_sum` differs in the 7th decimal. Both values above 1 + 1e-6
(1.0000017, 1.0000013) are from the second attempt; the first attempt's maximum was 1.0000005.
The re-check uses a fresh directory, and the script now refuses to run if `metrics.jsonl`
already exists.
