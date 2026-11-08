# H2S-NN — WISP capacity report (task card Part K, spec v1.6)

**Branch** `wisp-capacity` from `bf7f333`: K0 `f73c48f`, K1 `7d3b64b`, K2 `13b322f` (set-up) + `af9504e`, K2b `f846844` (set-up) + `1ee8b6c`, K3 `c2fdf02`, K4 `465d13e`, K5. Not pushed.
**Backup:** `wisp-followup` (`bf7f333`) is on `origin`.
**Evidence:** `docs/WISP_evidence/k_*`. Hardware: RTX 4060 8 GB.
**Tags:** RUN, CODE, DERIVED, UNVERIFIED, PROPOSAL. No number here is a performance result; synthetic accuracies are functional checks.

## 1. Summary
1. **WISP gate (ii) passes.** At n = 2000 (1,000 training targets), a fixed-β oracle reaches 0.992 / 0.988 / 0.996 at β = log 512, and β = 0 stays at 0.47–0.51 (no leak). If a cell's weight is right, the model uses the one decisive instance.
2. **The operator probe shows how.** The information is scattered at initialisation (probe ≈ 0.5–0.6, dispersion 0.87–0.92). Training aligns the transported class directions (dispersion 0.33–0.52; probe 0.96–0.996).
3. **K2b:** at n = 2000, σ\* is strictly first by mean learned log-weight on 3/3 seeds (I4: 1/3). The margins over its last-relation sibling are small (0.14–0.21), and `uniform` also learns the task (0.996).
4. **K1:** the maps of `endpoint_edges` and `free_path_maps` and the pair-attention scores read `[h₀, c]` again (E10); the goldens are bit-identical. **K3:** `r` is never tuned, and has a per-ℓ_max table. DBLP has no valid `r` at ℓ_max = 2.
5. **Mutations:** 12/12 caught. **Suite:** 365 tests; the only failures are the 24 pre-existing `test_baselines` ones.

## 2. Files changed
- **Code:**
  - `src/models/h2snn_model.py`: the map encoder (`_map_latent`); pair-attention latents read the geometry; `build_model` refuses an unresolved `path_source`.
  - `src/configs/config.py`: `WISP_WALKS_PER_ANCHOR`; the tuning guards; the `path_weight_context` combinations.
  - `src/data/dataset.py`: the unresolved-source check.
- **Tests:** `tests/test_v2_model.py` (T11-ctx, 5 tests) and `tests/test_wisp.py` (`WispGuardTests`, 5 tests). Two old `r` assertions were changed (`:249`, `:1202`, authorised by E11).
- **Evidence scripts:** `k2_capacity.py` (oracle and probe live here), `k2b_s2_n2000.py`, `k3_calibrate_r.py`, and the mutation checks.
- **K4:** an erratum appended to `docs/H2S-NN_WISP_followup_report.md`. The D2-row default in `H2S-NN/AGENTS.md` / `CLAUDE.md` is changed to `path_source: wisp` (node classification) and left **uncommitted**.

## 3. Deviations and decisions
- **K1:** the second encoder exists only when `g`'s input differs from the geometry (context on, appendix off). It is created last, so the context-off model and the goldens are unchanged. Pair attention: its encoder now reads `geometry_dim`.
- **K2:**
  - β is a buffer that requires grad but is not in the optimizer. The oracle's own scalar `b` sits in the scalar group, as in I3.
  - n = 2000 fits (peak 1,946 MB), written down before the gated runs.
- **K2b:** r = 200 already covers σ\* on every seed's evaluation draw (set-up from the draws only).
- **K3:**
  - **DBLP ℓ_max = 2 has no incidence-matching r** (DERIVED bound: every walk, revisits allowed, gives ≤ 511,887 < 862,788). The first calibration run doubled r without end and segfaulted; the script now checks the bound and caps r. The table omits that entry, so an unset r raises.
  - The calibrated-value rule is enforced in the tuning check only: the synthetic scripts set their own r.
  - NIT: the consumers raise on `path_source == ""` (a directly built `DataConfig`).
- **Environment:** a Windows Application Control policy blocks the optional `torch-scatter` / `torch-sparse` DLLs (also in Part I's final suite). Only two baseline adapters import them.

## 4. Verification evidence (RUN)
- **K1:**
  - T11-ctx (a)–(d) at the default context, plus a context-off check: 5 OK.
  - `k1_mutation_check.py` 3/3 caught on assertions (map latent fed `h₀`; `g` fed the geometry, which also fails T-H7; pair-attention latents fed `h₀`).
  - Affected modules: **149 OK**, goldens bit-identical (they pin `geometry_context: false`).
- **K3:**
  - `WispGuardTests` 5 OK; `k3_mutation_check.py` **9/9 caught**. One shows as an error: removing the uncalibrated-ℓ_max raise turns it into a `KeyError`.
  - Affected modules: **221 OK**.
- **K5:** `python -m unittest discover -s tests` → `Ran 365 tests … FAILED (failures=11, errors=13)`, all 24 in `test_baselines` (`k5_suite_output.txt`).

## 5. Measurements
**K1: parameter counts** (T11 fixture; `k1_param_counts_*.json`), context on, before → after:
- `endpoint_edges` 5,480 → 8,456;
- `free_path_maps` 6,779 → 12,315;
- pair attention 5,917 → 10,013;
- `full`, and every arm with the context off: unchanged.

**K2: gate (ii)** (`k2_results.json`; test accuracy of the model / probe on `u = A h₀ − h₀` at the selected checkpoint; seeds 42 / 43 / 44):

| n | β | Model | Probe (checkpoint) | Probe (init) | Dispersion init → ckpt | Relevant share of `D` (ckpt) |
|---|---|---|---|---|---|---|
| 2000 | 0 | .506 / .512 / .472 | .52 / .51 / .50 | ≈ .50 | .87–.92 → .87–.92 | 0.001 |
| 2000 | log 64 | .996 / **.698** / .982 | .99 / .72 / .98 | .51–.54 | → .34 / .69 / .39 | 0.45 |
| 2000 | log 512 | **.992 / .988 / .996** | .996 / .956 / .994 | .58–.63 | → .42 / .52 / .43 | 0.87 |
| 500 | 0 / log 8 / log 64 | .46–.52 | .42–.46 | .40–.48 | .87–.90 (unchanged) | ≤ 0.08 |
| 500 | log 512 | .864 / .648 / .824 | .83 / .61 / .76 | .50–.57 | → .59 / .69 / .55 | 0.87 |

- The `h₀` control probe stays at .43–.55 everywhere.
- **`full`** (learned `g`, n = 2000, seed 42): 0.508; the relevant − irrelevant log-weight gap stays within ±0.002.
- **Max log-weight:** 5.8.

**`−∂L/∂β > 0`**, as the fraction of epochs before the best epoch (that number of epochs in parentheses):

| n | β = 0 | log 8 | log 64 | log 512 |
|---|---|---|---|---|
| 2000 | .80 (59) / .86 (51) / – (0) | — | 1.00 (49) / .99 (85) / .99 (81) | .97 (32) / 1.00 (69) / .98 (48) |
| 500 | uninformative (1–4 epochs) | same | same | .89 (72) / .86 (65) / .99 (81) |

Reading (fixed in K0): at n = 2000 with β > 0 the sign is consistently positive, so a learnable scale would move the right way given enough reach.

**K2b: S2 at n = 2000** (r = 200, coverage 1.0; `k2b_results.json`). Top schemas by mean learned log-weight (lift):

| Seed | 1st | 2nd | 3rd | σ\* rank (mean log w / lift) | Test acc |
|---|---|---|---|---|---|
| 42 | **σ\* a b c d** 6.954 (×12.3) | a b c e 6.817 (×10.7) | a b m k 4.118 | 1 / 1 | 1.00 |
| 43 | **σ\*** 5.909 (×12.9) | a b c e 5.697 (×10.5) | a b m k 1.965 | 1 / 1 | 1.00 |
| 44 | **σ\*** 6.965 (×12.3) | a b c e 6.795 (×10.4) | a b m k 4.086 | 1 / 1 | 0.99 |
| `uniform` 42 | all tie at −0.427 | | | 11 / 9 | 0.996 |

**K3: r by ℓ_max** (`k3_calibrate_r.json`; the smallest r reaching the `schemas` budget, seed 42):

| ℓ_max | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| ACM (403,122) | 64 | 47 | 42 | 38 | 36 |
| DBLP (862,788) | none (≤ 511,887 reachable) | 93 | 65 | 55 | 48 |
| IMDB (143,052) | 25 | 13 | 12 | 11 | 11 |

## 6. Self-review checklist
- **K1, maps and scores read `[h₀, c]`; `g` reads `h₀`:** PASS (T11-ctx; mutations). Goldens: PASS (context off).
- **K2, β never trained; nothing clamped; reachability and route audit before the runs:** PASS (`k_preregistration.md`, committed before any run; β is not in the optimizer).
- **K2b, criterion unchanged and strict (ties count against σ\*):** PASS; `uniform` is not first.
- **K3, one test per rule, each fails under a targeted mutation:** PASS (9/9).
- **Every test fails when its property breaks:** PASS for the 12 mutations; the K2/K2b runs are measurements.

## 7. Assumptions imposed for theory and reality checks
- **Gate (ii) answers capacity, not learning.** The oracle sets the weights; whether `g` learns them is still open. In S1′, `full` learns no selection at n = 2000 (gap ±0.002), as in Parts H and I.
- **The idealised SNR (e^β/√63) undershoots the finite-sample need.** At n = 2000, log 64 passed on 2/3 seeds (s43: probe .72, dispersion .69: the class direction did not align). log 512 passed on 3/3.
- **Memorisation still limits n = 500** (250 targets): log 512 reached at most 0.86.
- **K2b:** with 1,000 targets the label is learnable without the weights (`uniform` 0.996). σ\*'s lead over its sibling is small. "Recovered" means strictly first, as pre-registered, not by a margin.

## 8. Open items
- **Learning (not capacity):** why `full` does not learn the S1′ selection; the cluster protocol's C2 runs.
- G0b H1 (readout ego route) stays deferred (erratum).
- **PROPOSAL:** a rule for ℓ_max values without an incidence-matching r (DBLP ℓ_max 2), e.g. all cells, for the U3 curve.
- **Compatibility:** configs saved before K3 that carry an `r` other than 0 or the table value are still valid outside tuning.

## 9. Decisions needed from the author
1. Accept gate (ii) as passed (β = log 512, 3/3), with log 64 at 2/3 recorded?
2. DBLP at ℓ_max = 2: exclude it from the U3 curve, or define r there (e.g. "all cells")?
3. Push `wisp-capacity` after reading this report?
