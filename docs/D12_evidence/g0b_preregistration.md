# G0b — gap localization ladder: pre-registration

Written 2026-11-08, before any G0b code change or run. Author-approved multi-seed runs (G0b only).
Nothing below is changed after seeing a result; additions go under a dated heading.

## Common protocol
- Pipeline: ours (`build_dataset` with `configs/experiments/<ds>_h2snn.yaml`: HGB split, 20 % of
  `label.dat` to validation with split seed 42, identical masks for every rung and seed).
- Datasets: ACM, DBLP. Metric: test macro-F1 (sklearn, `src/utils/misc.py`). Selection: test at the
  epoch of best validation macro-F1 (first maximum), as in G0 arm O and `trainer.py`
  (`checkpoint_metric: macro_f1`). Logged per run: train / val / test macro-F1 at that epoch, best epoch.
- Per rung and dataset: mean and sample std (ddof 1) over seeds. "Within 2 points" / "more than
  2 below" compare means, in macro-F1 points (×100).

## Rungs
| Rung | What | Seeds |
|---|---|---|
| L0 | R-GCN feature-free = G0 arm O (vendored HetSheaf `RGCN.py`, committed config, G0 loop) | 42–46 |
| L1 | HetSheaf's `GCN.py` with its committed config (`model/gcn.yaml`: 2 layers, hidden 32, dropout 0.2, output 32; lr 1e-3, wd 5e-4, cosine, 200 epochs, early stop on val loss patience 100) on our pipeline: our raw features zero-padded to a common width (`stack_node_features`, the same padding convention as PyG's `to_homogeneous`), our facts + reversals as one homogeneous graph, G0 loop | 42–44 |
| L2-raw | `ablation.h0_only_readout: true` through `trainer.py`, raw features (current pipeline) | 42–44 |
| L2-row | same, with `data.feature_normalization: row` (new switch, default `none`): PyG `NormalizeFeatures` per node type on the raw HGB features before stacking — the transform HetSheaf applies | 42–44 |
| L3 | sheaf-free H2S-NN through `trainer.py`: `ablation.identity_maps: true` (new diagnostic switch: every restriction map `F = I`, scale 1, so `Δ₀` is the graph Laplacian over facts ⊗ `I_d`), `ablation.no_path_cells: true` (γ = 0), `ablation.coupling: symmetric`, the D11 first-order update (K = 4, learned η), readout `[x_K, h₀]` | 42–44 |
| L4 | full D11 through `trainer.py` (config unchanged) | 42–44 |

Published bands (HetSheaf Table 2, mean ± 2 std, macro-F1): GCN ACM 92.25 ± 0.82 → [90.61, 93.89];
DBLP 90.87 ± 1.77 → [87.33, 94.41]. R-GCN: G0 bands (reference only; L0 is the internal reference).

## Reading rules (card G0b, operationalized now)
Each rule is evaluated per dataset. A conclusion is drawn when ACM and DBLP agree; if they disagree,
both are reported and no single conclusion is drawn.
- (i) L2-row mean > L2-raw mean + 2 → **feature-preprocessing defect**: fix it for every model (a
  pipeline fix). In G0b the switch is added and the result reported; flipping the default for all
  models is proposed in the report, not done (G0b stops after reporting).
- (ii) |L3 − L0| ≤ 2 and L4 < L3 − 2 → **the learned sheaf (maps, gates, 2-cells) hurts**: design-level, stop.
- (iii) L3 < L0 − 2 → **propagation or training** (trainer settings, reach of K steps, readout).
  Diagnose with train macro-F1 at the selected epoch (mean over seeds): < 90 = underfitting;
  ≥ 98 with val macro-F1 more than 2 below L0's val = overfitting; otherwise reported unlabelled.
- (iv) L1 mean < its band's lower edge → **feature path of the baselines also suspect**.
- Not covered by (i)–(iv) (e.g. L3 > L0 + 2, or L3 ≈ L0 and L4 ≈ L3): reported as is.

## Expectations (stated so they can be wrong)
- L0: ACM ≈ 88–90, DBLP ≈ 91–92 (G0 seed 42: 89.04, 92.08); seed std ≈ 1 on ACM.
- L2: row vs raw differ by < 2 (the type encoder is linear per type; a per-row rescale should matter
  little) — I may be wrong; this is the test.
- L3: below L0 by more than 2 on both datasets (D11 variants without a working sheaf already sit at
  84–86 on ACM), so (iii) fires; L4 ≈ L3.
Wrong if: L3 ≈ L0 (then the deficit is in the learned sheaf, rule ii), or L2-row ≫ L2-raw (rule i).

## Implementation notes fixed now
- `identity_maps` requires `no_path_cells` and forbids `h0_only_readout` and `pair_attention` (the
  path transports and attention are not defined for it in this unit); the map module is still built
  (state dict unchanged) but not called. Rejected by `reject_never_tuned_settings` (base config and
  search space), like `coupling`.
- `feature_normalization` ∈ {`none`, `row`}, default `none` (behaviour unchanged); rejected in search spaces.
- Unit tests: identity maps give `Δ̂ = I − D^{-1/2} A D^{-1/2}` ⊗ I on a small graph (dense check);
  default configs build the same model state dict as before; the row switch matches PyG's transform.
- Runs: one worker, sequential (host memory limits seen in D11).
