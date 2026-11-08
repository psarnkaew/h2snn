# H2S-NN — WISP capacity report (task card Part K, spec v1.6)

**Branch:** `wisp-capacity` from `bf7f333`: K0 `f73c48f`, K1 `7d3b64b`, K2 `13b322f`+`af9504e`, K2b `f846844`+`1ee8b6c`, K3 `c2fdf02`, K4 `465d13e`, K5 `25cab8a` + review. Not pushed. `wisp-followup` is on `origin`.
**Evidence:** `docs/WISP_evidence/k_*`. RTX 4060 8 GB.
**Tags:** RUN, CODE, SRC, DERIVED, UNVERIFIED, PROPOSAL. Synthetic accuracies are functional checks, not results.

## 1. Summary
1. **Gate (ii) passes** (RUN): at n = 2000, a fixed-β oracle gives 0.992 / 0.988 / 0.996 at β = log 512 (relevant cell ≈ 0.87 of its target's `D`). log 64 passes on 2/3 (share ≈ 0.45; seed 43: 0.698). β = 0 is at chance.
2. **The probe at the checkpoint tracks the model:**
   - at initialisation: probe ≈ 0.5–0.6, dispersion 0.87–0.92;
   - at log 512: probe .96–.996, dispersion .42–.52;
   - at log 64: probe .72–.99, dispersion .34–.69.
3. **K2b:** σ\* is strictly first by mean log-weight on 3/3 seeds, but only 1.15–1.24× above its sibling. `uniform` also solves the task (0.996) through relation-specific facts (an audit miss), so selection was not needed.
4. **K1:** `endpoint_edges` / `free_path_maps` maps and pair-attention cell latents read `[h₀, c]`; goldens bit-identical.
   **K3:** `r` is never tuned and has a per-ℓ_max table. DBLP has no valid `r` at ℓ_max 2.
5. **Mutations:** 12/12 caught. **Suite:** 365 tests, only the 24 pre-existing `test_baselines` failures. **Review:** 0 blockers, 0 MAJOR, all 5 MINOR accepted.

## 2. Files changed
- **`src/models/h2snn_model.py`:** the map encoder (`_map_latent`, created last, only when `g`'s input ≠ the geometry); pair-attention cell latents read the geometry; `build_model` refuses `path_source == ""`.
- **`src/configs/config.py`:** `WISP_WALKS_PER_ANCHOR` (by ℓ_max); the tuning guards; the `path_weight_context` combinations.
- **`src/data/dataset.py`:** the unresolved-source check.
- **Tests:** `test_v2_model.py` T11-ctx (5 tests); `test_wisp.py` `WispGuardTests` (5 tests). Two old `r` assertions changed (E11).
- **Evidence scripts:** `k2_capacity.py`, `k2b_s2_n2000.py`, `k3_calibrate_r.py`, and the mutation checks.
- **K4:** an erratum, plus a post-review correction, appended to the Part I report. The D2-row default in AGENTS.md/CLAUDE.md is changed and left **uncommitted**.

## 3. Deviations
- **K2:**
  - β is a buffer that requires grad, never in the optimizer. The oracle's own `b` is in the scalar group (as in I3).
  - n = 2000 fits (1,946 MB peak), recorded before the gated runs.
- **K2b:** r = 200 already gives σ\* coverage 1.0 on every seed's draw (set-up only).
  - **The route audit missed** that d, e and k are distinct relations. The fact maps can carry C\*'s class without any cell weight, and `uniform` at 0.996 confirms it.
- **K3:**
  - **DBLP ℓ_max 2 has no incidence-matching r** (DERIVED: all walks, revisits allowed, give ≤ 511,887 < 862,788). The first calibration run doubled r without end and segfaulted; the script now checks the bound and caps r.
  - The "0 or the calibrated value" rule is enforced in the tuning check only (card), narrower than spec E11 (§9).
- **Environment:** a Windows Application Control policy blocks the optional `torch-scatter` / `torch-sparse` DLLs. Model code does not use them.

## 4. Verification evidence and commands (RUN)
- **K1:**
  - `python -m unittest tests.test_v2_model.TwoLatentContextTests` → 5 OK.
  - `k1_mutation_check.py` → 3/3 caught on assertions.
  - `k1_param_counts.py before|after`.
  - Affected modules: 149 OK, goldens bit-identical (they pin `geometry_context: false`).
- **K2:** `k2_capacity.py --n {500,2000} --betas … --seeds 42 43 44` and `--arm full`, preceded by `--fit-check` → `k2_results.json`, `k2_run_output.txt`.
- **K2b:** `k2b_s2_n2000.py --choose-r`, then `--r 200 --arms full|uniform` → `k2b_results.json`.
- **K3:**
  - `k3_calibrate_r.py acm|dblp|imdb` → `k3_calibrate_r.json`.
  - `WispGuardTests` 5 OK; `k3_mutation_check.py` 9/9 caught (one as a `KeyError`); 221 affected OK.
- **K5:** `unittest discover` → `Ran 365 tests … failures=11, errors=13`, all in `test_baselines` (`k5_suite_output.txt`; pre-existing since D10, SRC `H2S-NN_D10_report.md:27`).

## 5. Measurements
**K1: parameter counts** at the default context, before → after (T11 fixture): `endpoint_edges` 5,480 → 8,456, `free_path_maps` 6,779 → 12,315, pair attention 5,917 → 10,013. `full` and every context-off arm are unchanged.

**K2: gate (ii)** (test accuracy, seeds 42 / 43 / 44; probe = logistic regression on `u = A h₀ − h₀`, LBFGS, L2 1e-3):

| n | β | Model | Probe, ckpt (init) | Dispersion init → ckpt | Relevant share of `D`, ckpt |
|---|---|---|---|---|---|
| 2000 | 0 | .506 / .512 / .472 | .52 / .51 / .50 (≈ .50) | .87–.92, unchanged | .001 |
| 2000 | log 64 | .996 / **.698** / .982 | .99 / .72 / .98 (.51–.54) | → .34 / .69 / .39 | .45 |
| 2000 | log 512 | **.992 / .988 / .996** | .996 / .956 / .994 (.58–.63) | → .42 / .52 / .43 | .87 |
| 500 | 0, log 8, log 64 | .46–.52 | .42–.46 (.40–.48) | .87–.90, unchanged | ≤ .08 |
| 500 | log 512 | .864 / .648 / .824 | .83 / .61 / .76 (.50–.57) | → .59 / .69 / .55 | .87 |

- The `h₀` control probe stays at .43–.55.
- **`full`** (learned `g`, n = 2000, seed 42): 0.508; the relevant − irrelevant gap is ±0.002.

**`−∂L/∂β > 0`**, as the fraction of epochs before the best epoch (that number of epochs in parentheses):
- **n = 2000:** log 64 1.00 (49) / .99 (85) / .99 (81); log 512 .97 (32) / 1.00 (69) / .98 (48); β = 0 .80 (59) / .86 (51) / – (0).
- **n = 500:** log 512 .89 (72) / .86 (65) / .99 (81). The others rest on 1–4 epochs and are uninformative.
- **Reading** (fixed at K0): consistently positive, so a learnable scale would move the right way given enough reach.

**K2b: S2 at n = 2000** (r = 200, coverage 1.0), by mean learned log-weight (lift):

| Seed | 1st | 2nd | 3rd | σ\* rank (mean / lift) | Test acc |
|---|---|---|---|---|---|
| 42 | **σ\* a b c d** 6.954 (×12.3) | a b c e 6.817 (×10.7) | a b m k 4.12 | 1 / 1 | 1.00 |
| 43 | **σ\*** 5.909 (×12.9) | a b c e 5.697 (×10.5) | a b m k 1.97 | 1 / 1 | 1.00 |
| 44 | **σ\*** 6.965 (×12.3) | a b c e 6.795 (×10.4) | a b m k 4.09 | 1 / 1 | 0.99 |
| `uniform` 42 | all tie (−0.427) | | | 11 / 9 | 0.996 |

**K3: r by ℓ_max** (the smallest r reaching the `schemas` budget, seed 42):

| ℓ_max | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| ACM | 64 | 47 | 42 | 38 | 36 |
| DBLP | none (≤ 511,887 reachable) | 93 | 65 | 55 | 48 |
| IMDB | 25 | 13 | 12 | 11 | 11 |

**Max log-weight:** 5.8 (K2), 10.3 (K2b), against the stop level of 30.

## 6. Self-review checklist
- **K1, maps and cell latents read `[h₀, c]`; `g` reads `h₀`:** PASS (T11-ctx, 3/3 mutations). Goldens: PASS.
- **K2, β never trained; nothing clamped; reachability and route audit before the runs:** PASS. Commit order verified by the reviewer.
- **K2b, criterion strict and unchanged:** PASS. Its route audit: **FAIL** (relation-specific route missed; §3).
- **K3, one test per rule, each failing under a targeted mutation:** PASS (9/9).

## 7. Assumptions imposed for theory and reality checks
- **THEORY-CONSTRAINT** (the normalised operator, positive `w_f`): a cell acts through its share of `D`. Selecting one instance in 64 worked when that share reached about 0.87, and only partly at about 0.45.
  - Cost on real data: sparse decisive evidence needs a large range in `g`.
  - Gate (ii) tests capacity, not learning: `full` still learns no selection in S1′ (gap ±0.002).
- **THEORY-CONSTRAINT** (the idealised one-application SNR e^β/√63) is optimistic for finite samples (log 64: 2/3). It is not a design target (Part I erratum, item 3).
- **Reality checks:**
  - With 250 training targets every arm memorises; log 512 reaches at most 0.86.
  - With 1,000, S2 is solvable without the weights through relation-specific fact maps.

## 8. Open items
- **Learning:** `g` does not learn the S1′ selection. Cluster C2 runs.
- G0b H1 (ego route) stays deferred.
- **PROPOSALs:**
  - an S2 variant whose sibling carriers share σ\*'s last relation type, so only cell weights can separate them;
  - an r rule for (DBLP, ℓ_max 2).
- The stale comment at `h2snn_model.py:912` (pair attention); no code change after the review.

## 9. Decisions needed from the author
1. Accept gate (ii) as passed (log 512 3/3; log 64 2/3 recorded)?
2. C2b's S2: count K2b as met, or rerun the PROPOSAL variant on the cluster?
3. DBLP ℓ_max 2 for U3: exclude it, or define r there?
4. E11: enforce "0 or the calibrated value" in every config, not just in tuning?
5. Push `wisp-capacity`?
