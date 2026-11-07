# H2S-NN — D12 report (task card Part G)

**Status:** G0 only (2026-11-08). G1–G4 not started; they wait for the author's reply to this section.
Branch `d12-encoder` (off `tier0-screen-d11`). Evidence: `docs/D12_evidence/`. Hardware: RTX 4060 8 GB, Windows, env `h2snn` (torch 2.6, PyG 2.8.0, torchmetrics 1.9.0).

## G0 — Pipeline-correctness gate

### 1. Summary
- **(a) Splits: PASS.** For ACM, DBLP and IMDB, our test set is exactly HGB's `label.dat.test`, train ∪ val is exactly `label.dat`, the validation share is HGB's 20 % (same sizes as HGB's `int(0.2 n)`), and all 3025 / 4057 / 4573 labels match. HGB's own loader was the reference.
- **(b) Reference baseline: outside the band on every dataset; per the pre-registered rule, ACM = FAIL (diagnosed), DBLP = INCONCLUSIVE.** R-GCN (HetSheaf's code and committed config, seed 42) on our pipeline scored ACM 89.04 and DBLP 92.08, against bands [91.63, 94.91] and [93.84, 94.76].
- **Diagnosis:** I found no defect in our data. The gap between our pipeline and HetSheaf's comes from two HetSheaf choices (a 10 % validation share instead of HGB's 20 %, and `ToUndirected`, which duplicates every relation). The rest of the gap to the published number is the unpublished sweep configuration: HetSheaf's own pipeline also misses with that configuration.
- **What this means for D12:** on the identical split, labels, metric and selection rule, a feature-free R-GCN beats H2S-NN (D11 screen, seed 42) by **5.0 points on ACM and 9.9 on DBLP**. The 6–10-point deficit therefore lies in the model, training or feature path, not in splits, labels, metric or selection.

### 2. Files changed (no `src/` or `tests/` change)
`docs/D12_evidence/g0_preregistration.md` (rules, written before any run; dated addendum written before the diagnostics) · `g0_splits.py` + `g0_splits.json`/`_output.txt` (a) · `g0_rgcn.py` + `g0_rgcn_<ds>_<arm>.json` (per-epoch curves) + `g0_rgcn_*_output.txt`, `g0_swaps_output.txt`, `g0_graph_decomp_output.txt` (b) · this report.

### 3. Deviations from the card
- **Published configuration.** HetSheaf selected R-GCN with a 150-run sweep and publishes only the search space (App. C.1.1, Table 5; VERIFIED-SOURCE). I used the only configuration on record, the repo's committed defaults (2 layers, hidden 32, dropout 0.2, embedding mode, AdamW 1e-3 / 5e-4, cosine, 200 epochs, early stopping on validation loss with patience 100, test with final weights). Whether this matches the sweep winner is UNVERIFIED. Arm P was added to measure what this configuration gives on HetSheaf's own pipeline.
- **Lightning not installed.** HetSheaf's training loop was re-implemented without it: same optimizer, scheduler, early stopping and loss; metric checked against torchmetrics.
- **SimpleHGN not run.** Its code is DGL-based, and DGL does not import in `h2snn`.
- **Extra runs.** Ten single-seed diagnostic runs (swaps). There were no multi-seed runs.

### 4. Verification evidence
`python ../docs/D12_evidence/g0_splits.py` (from `H2S-NN/`): `test_equal: true`, `train_union_val_equal_label_dat: true`, `train_val_disjoint: true`, `label_mismatch_count: 0` on all three datasets; sizes are 726/181/2118 (ACM), 974/243/2840 (DBLP) and 1097/274/3202 (IMDB). The test class counts equal HGB's. `python ../docs/D12_evidence/g0_rgcn.py` also checked that our tied facts plus their reversals reproduce the stored HGB edge multiset exactly (`reconstructs_stored_edges: true`; 547,872 / 239,566 / 86,642 edges).

### 5. Measurements (test macro-F1 ×100, seed 42, RTX 4060; O = our pipeline, P = HetSheaf's pipeline, final-epoch weights unless marked)
| | Band (pub. ± 2 std) | O best-val | O | P | O graph + P split | P graph + O split | P graph without `ToUndirected` + O split | H2S-NN full / h₀-only (D11, s42) |
|---|---|---|---|---|---|---|---|---|
| ACM | [91.63, 94.91] | 89.04 | 88.47 | 91.20 | 89.77 | 90.61 | 89.71 | 84.01 / 83.77 |
| DBLP | [93.84, 94.76] | 92.08 | 91.96 | 91.93 | — | — | — | 82.18 / 76.98 |
| IMDB | [59.25, 60.09] | 50.74 | 51.03 | 55.30 | 52.81 | 53.96 | 51.89 | not run |

**Diagnosis, one component at a time:**
- **Metric:** ruled out. sklearn and torchmetrics agree to < 1e-6.
- **Split:** worth +1.3 (ACM) and +1.8 (IMDB). HetSheaf's `SubsetTrainSplit(val_ratio=0.1)` trains on 817 ACM nodes instead of 726. Their paper says "canonical HGB 24/6/70" (App. C.3.1), but their code gives 27/3/70. That is a HetSheaf paper–code discrepancy (VERIFIED-SOURCE vs VERIFIED-CODE `hetsheaf/datasets/hgb.py:55`).
- **Graph:** worth +2.1 (ACM) and +2.9 (IMDB), all of it from `ToUndirected`. HGB already stores both directions, so the transform gives every direction a second relation type: 14 relations instead of 8 on ACM, with twice the edges. Removing self-loops or duplicates changed nothing (90.56 / 90.61 vs 90.61).
- **Features:** not involved (embedding mode).

**Noise caveat:** HetSheaf's graph without `ToUndirected` is our graph minus 16 self-loops, yet it scored 89.71 against 88.47. Single-seed noise or ordering noise is therefore about 1 point, so the individual swap effects (1.3–2.9) are only 1–3× noise.

### 6. Self-review checklist
- Split ids compared with an independent parser (HGB's loader, not PyG): PASS.
- Labels compared on every labelled node, IMDB multi-hot exactly: PASS.
- Tolerance fixed before any run: PASS (`g0_preregistration.md`). The diagnostic plan was added after the gate runs, before the diagnostics, and is marked as such.
- Test labels never used for selection: PASS. Selection uses validation loss or validation macro-F1 only; test is logged per epoch for reporting.
- Arm O exercises our loader, split, labels and metric code: PASS. It does **not** exercise our trainer (`src/trainers/trainer.py`) or the feature path: NOT CHECKED.
- Seed noise for this configuration: NOT CHECKED (needs approved multi-seed runs). One near-duplicate pair suggests about 1 point.
- Independent sub-agent review: not done for G0; planned for G4.

### 7. Assumptions imposed for theory and reality checks
No THEORY-CONSTRAINT was touched.
- **Reality check (protocol):** published numbers on HGB are not protocol-equivalent. HetSheaf's R-GCN uses a different validation share from HGB's and relation-duplicated inputs, and is selected by a 150-run sweep. A "within 2 std of published" gate tests the configuration as much as the pipeline. This supports spec §6 rule "baselines all rerun under this protocol; no quoted numbers".

### 8. Open items
- **Feature path unverified.** `h₀`-only is 15 points below the feature-free R-GCN on DBLP (76.98 vs 92.08). Our features are raw: HetSheaf row-normalises, and HGB's recommended `feats_type` for ACM is 2 (as transcribed in `src/baselines/bundles.py:44`; HGB source not opened). A features-using reference on our pipeline (e.g. HetSheaf's GCN, ACM 92.25) would test this. It is not run because the card allows one baseline.
- **Our trainer is unverified as a component.** R-GCN was trained by the gate script's loop, not by `trainer.py`.
- **Number reproduction not achieved on any pipeline.** It needs either HetSheaf's selected R-GCN configuration (UNVERIFIED whether it can be obtained) or HGB's own R-GCN/SimpleHGN with their commands (needs an HGB clone and working DGL).
- **D11 screen incomplete.** 19 of 45 runs are done; the driver is not running and was not restarted. Stray empty `C:\c\…` folder (from D11) is still present.

### 9. Decisions needed from the author
1. Should G0 count as passed on the data leg (splits, labels, edges, metric verified exactly) with the reproduction leg recorded as "not achievable without the sweep winner"? If not, which route: (a) ask HetSheaf's authors for the selected configurations, (b) set up HGB's R-GCN/SimpleHGN, or (c) approve 5 seeds of arms O and P to measure noise?
2. Should a feature-path check (one features-using baseline on our pipeline, plus `h₀`-only with row-normalised features) run before G1? R-GCN without features outscores every D11 variant on the same pipeline, so the deficit may be partly in how H2S-NN uses features and the graph rather than in the sheaf design.
3. Which split should the protocol cite when comparing to HetSheaf's paper numbers (C7): HGB's 24/6/70, which we use, or HetSheaf's code split of 27/3/70?
