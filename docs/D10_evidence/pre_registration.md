# D10 pre-registration (written before any code or run)

Branch D1_add after the D7 commits (HEAD 64620a2). Defaults after Part D are the interim
`untied` / `dfs` / no resampling (D7 default flip is awaiting the author's decision A/B/C);
D10 does not depend on them. The card names `src/models/h2snn_v2.py`; the file that runs is
`src/models/h2snn_model.py` (VERIFIED-CODE: `diffuse` at h2snn_model.py:613, `encode` :621).

## Reality check (Part A, "Reality before theory")
Done before the design is touched; results in the report, evidence in this directory.
- Data: ACM subject degree skew, label homophily of PAP / PSP cells (script `reality_check.py`).
- Published models: CSNN paper opened (VERIFIED-SOURCE `references/cooperative_sheaf_neural_network.pdf`
  p.4-6: Prop. 3.1 says undirected SNNs cannot LISTEN-without-PROPAGATE or the reverse; the
  CSNN layer composes `(Δ_in)^T Δ_out`, Eq. 6-8). Simple-HGN / HGT / HAN directional attention:
  checked in vendored code if present, else UNVERIFIED.
- Steelman: per-edge directional strength would be more expressive than node-level gates
  (THEORY-CONSTRAINT: node-level only, because per-edge breaks the similarity to a symmetric
  operator and the real spectrum; may cost expressiveness on relation-specific asymmetry).

## Units (smallest independently testable changes)
| Unit | Change | Test / measurement | Expected | Wrong if |
|---|---|---|---|---|
| U1 | `ablation.coupling`, `model.coupling_gate_init`, gate heads, gated update `x - eta G Δ̂ B x`, symmetric = no heads | T1 (golden with `coupling: symmetric`), T2 (dense reference), T6 (gradients) | golden bit-identical; T2 ≤ 1e-10 (float64); both heads get non-zero grads | any golden entry differs (stop); T2 > 1e-10 |
| U2 | spectrum tests | T3 (60 fixtures, gates in [0.01, 1]), T5 | max abs Im λ / max abs λ ≤ 1e-9; λ_max ≤ 2 max(gb) + 1e-9; CSNN control (Eq. 6-8) has abs Im λ > 1e-3 on ≥ 1 fixture; T5: kernel dim of `G Δ̂ B` = dim ker Δ̂, eigenvalue cut 1e-9 (stated now) | T3 or T5 fails after 3 attempts (stop); control real (test vacuous) |
| U3 | T4, logging, measurements, smoke | T4 exact zeros vs > 1e-8 control; 3 mutations; overhead; 20-epoch ACM smoke | overhead ≤ 5% time, ≤ 2% memory (gate heads: 2 linears on h0 + 2 vector products per step); η·λ_max < 2 and loss finite | mutation not caught; overhead above bound (reported, not hidden); smoke diverges (stop) |

## Choices the card leaves open (stated now)
- Gate-head init: bias = logit(`coupling_gate_init`), weight ~ N(0, 0.01²) (init choice, like the
  rotation init at sheaf_maps.py; not a config key). Gates start at 0.95 ± ~0.003.
- `AblationConfig.coupling` is a string: the bool-only validation loop exempts it and it is
  validated against {cooperative, listen_only, symmetric}.
- Gates are instantiated only when diffusion exists (`not h0_only_readout`); `listen_only`
  instantiates only the listen head; `symmetric` none.
- `coupling` and `coupling_gate_init` are rejected in tuning search-space names (extends
  `reject_never_tuned_settings`).
- λ_max by power iteration on the symmetric similar form `(GB)^{1/2} Δ̂ (GB)^{1/2}` (Rayleigh
  quotient, a lower bound of λ_max; steps stated with each number). Logged every
  `cfg.log_every` epochs in the node / link trainers (cost: each log builds one extra operator),
  every epoch in the smoke script; graph classification logs gate mean/min only (per-batch
  operators). Gate mean/min logged every training epoch from the training forward.
- T3 CSNN control: flat vector bundle per Eq. 7-8, node-level orthogonal S_i, T_i times positive
  scalars, undirected edges as two directed ones, normalized by D_out / D_in block scalars.
- Golden harness `docs/heads_evidence/v2_golden.py` `build_config` sets `coupling: symmetric`
  whenever the field exists; its recorded tensors are not re-captured (T1 = bit-for-bit).
