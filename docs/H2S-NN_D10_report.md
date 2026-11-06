# D10 report — cooperative coupling with a real spectrum

Written for: the H2S-NN author (task card Part E). Evidence in `docs/D10_evidence/`; hardware: RTX 4060 Laptop GPU (benchmarks), CPU (tests).

## 1. Summary
`x ← x − η G Δ̂ B x` with node-level sigmoid gates from `h₀`, in the shared encoder (all three tasks), `ablation.coupling: cooperative | listen_only | symmetric` (default `cooperative`), gate init `model.coupling_gate_init` (0.95). T1–T6 pass (14 tests), the 3-mutation check catches 3/3 (one only after I strengthened T3), full suite 256 tests with only the 24 pre-existing `test_baselines` failures. Overhead on ACM: +0.6% time, +0.5% memory. 20-epoch smoke: loss finite, η·λ_max ≤ 0.88. **No accuracy claim; the §10.5 decision rule (10 seeds, Holm) has not been run.**

## 2. Files changed
- `src/configs/config.py`: `ablation.coupling` (string, validated), `model.coupling_gate_init` ∈ (0,1), `reject_never_tuned_settings` now also rejects `coupling`/`coupling_gate_init` in search spaces.
- `src/models/h2snn_model.py`: gate heads (created last, none under `symmetric`/`h0_only_readout`), `coupling_gates`, gated `diffuse`, `lambda_max` (power iteration on `(GB)^{1/2}Δ̂(GB)^{1/2}`), `eta_lambda_max`, gate stats in `diffusion_scalars`.
- `src/trainers/trainer.py`, `link_prediction_trainer.py`: log `eta_lambda_max` every `log_every` epochs.
- `tests/test_cooperative.py` (new, 14 tests); `tests/test_v2_model.py`: 2 tests adapted to the new non-boolean field (same intent: every boolean switch off by default; all coupling modes train).
- `docs/heads_evidence/v2_golden.py`: harness names `coupling: symmetric`; `v2_golden.pt` untouched.

## 3. Deviations from the spec
1. The card names `src/models/h2snn_v2.py`; the file that runs is `h2snn_model.py` (VERIFIED-CODE). `docs/H2S-NN_D7_report.md` does not exist yet (D7 report and default flip await the author's decision A/B/C); D10 does not depend on it. Measurements use explicit `tied`+`walk`, the intended post-D7 model.
2. `η·λ_max` is logged every `cfg.log_every` (10) epochs in the node/link trainers (each log builds one extra operator), every epoch in the smoke script. Graph classification logs gate mean/min only (per-batch operators). Gate mean/min is logged every training epoch in all three tasks.
3. `λ_max` is a 30-step power-iteration Rayleigh quotient, a lower bound: 1.756 vs 1.780 at 300 steps (1.3% low) at the smoke run's end.
4. T3 was strengthened after the mutation check (§4).

## 4. Verification evidence
- Golden (T1): `python -m unittest tests.test_cooperative` → `Ran 14 tests … OK`; `compare` = 844 entries, 0 mismatches with `coupling: symmetric`; control: the default model has the 4 extra gate-head keys, so the golden would detect it.
- T2 dense reference (independent `Δ₀`, signed transports with general inverse, L ∈ {2,3,4}): ≤ 1e-10, per-edge and per-side scales, 3 modes.
- T3 (`spectrum_output.txt`, 60 fixtures, gates 0.0100–0.9900): max |Im λ|/max|λ| = 0.0; max λ / (2 max gb) = 0.930; min λ = 0.0. CSNN control (Eq. 6–8 of the paper, VERIFIED-SOURCE): max |Im λ| = 1.43 > 1e-3.
- T4 exact zeros (`x_u` unchanged bitwise; Jacobian blocks exactly 0) vs controls > 1e-8; T5: kernel dimension d with gap ≥ 1e3·cut (cut 1e-9, pre-registered); T6: both heads' weight and bias gradients non-zero.
- Mutations (`mutation_check_output.txt`): swapped sides → T4 + T2 fail; B without G → T4 + T2 fail; gates without sigmoid → T3 fails. **First attempt: the sigmoid mutation was NOT caught.** Diagnosis (DERIVED): `GΔ̂B = (GB⁻¹)(BΔ̂B)` is a diagonal times a PSD matrix, hence similar to a symmetric matrix, so even indefinite gates keep the spectrum *real*; only `λ ≥ 0` and `gates ∈ (0,1)` break. T3 now asserts both (the card's stated theory `[0, 2 max gb]`), one fix attempt.
- Full suite: `Ran 256 tests … failures=11, errors=13`, names identical to the pre-D7 baseline.

## 5. Measurements (ACM, d=32, factorized k=2, tied+walk, 3 repeats after 1 warm-up, alternating A/B)
| mode | params | peak GiB | s / fwd+bwd |
|---|---|---|---|
| symmetric | 302,695 | 1.4870 | 1.0526, 1.0605 |
| cooperative | 302,761 | 1.4941 | 1.0664, 1.0590 |
Overhead +0.6% time (spread between the two symmetric runs 0.8%), +0.48% memory; pre-registered ≤ 5% / ≤ 2%: met. Smoke (`smoke_output.txt`): loss 1.109→0.293; η 0.4975→0.4762; γ 1.006→1.053; g mean 0.9524→0.9493, min 0.9439→0.9404; b mean 0.9512→0.947; mean w_f 0.565→0.423; η·λ_max 0.88→0.84.

## 6. Self-review checklist
PASS `symmetric` bit-identical (T1). PASS spectrum real and bounded, CSNN control complex (T3). PASS listen/broadcast exact, swapped-sides caught (T4, mutation). PASS no new tuning knob besides `coupling_gate_init`; neither is in any sweep/notebook (test). PASS §7 filled. PASS 3 mutations caught, 0 vacuous (after the T3 strengthening).

## 7. Assumptions imposed for theory and reality checks
- `THEORY-CONSTRAINT: node-level gates only (no per-edge strength) — may cost relation- or pair-specific asymmetry, e.g. a hub that should broadcast to some neighbours but not others.` Steelman: per-edge directional strength is strictly more expressive. It breaks the similarity to a symmetric operator (complex spectrum possible, plan §10.4). No variant keeping both found; stays an open ablation candidate.
- `THEORY-CONSTRAINT: gates ∈ (0,1) (sigmoid) — may cost amplification (g>1 impossible); only attenuation.` Needed for `λ ≤ 2 max gb < 2`.
- `THEORY-CONSTRAINT: gates from h₀, fixed through the K steps (plan §11 item 2) — may cost state-dependent cooperation.` Untested here; separate decision.
- Default `cooperative` is itself unproven: `symmetric` and `listen_only` stay as ablations (theory never removes one); §10.5 decides.
- Reality checks: (a) data (`reality_check_output.txt`): ACM subject degree mean 54, max 1,195; term max 3,001; endpoint label homophily PAP 0.82, PSP 0.67 — skew and partial heterophily, compatible with unequal influence but not proof. (b) CSNN paper opened, p.4–6 (VERIFIED-SOURCE): undirected SNNs cannot listen without propagating (Prop. 3.1). (c) Directional attention in HAN/HGT/Simple-HGN/SeHGNN: UNVERIFIED (`vendor/HGB` empty, only an `edge_softmax` import seen in HINormer).
- Smoke shows the gates stay ≈ 0.95 over 20 epochs: this run gives no evidence that they specialize.

## 8. Open items
UNVERIFIED: directional-attention claim; gate behaviour beyond 20 epochs; benefit over `symmetric` (needs the 10-seed Tier 0/1 test); GC `η·λ_max`; D7 items (LastFM/PubMed/amazon/youtube data, default flip). Noticed, not fixed: gate heads sit in the weight-decayed group (1e-4, negligible against bias 2.94). `docs/D4_evidence/smoke_d4.py` and `measure_d4.py` still use removed config keys (`architecture`).

## 9. Decisions needed
(a) D7 default flip: A (baselines pinned to untied+dfs in `bundles.py`), B or C. (b) Keep `cooperative` as default before the §10.5 evidence, as the card says, or default `symmetric` until then? (c) Exclude gate heads from weight decay?
