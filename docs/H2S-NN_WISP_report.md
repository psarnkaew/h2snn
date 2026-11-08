# H2S-NN — WISP report (task card Part H)

**Status (2026-11-08):** H0–H3 done and committed. **Stopped at a Part H stop condition: the H3 variance gate was missed** (see H1–H3 below). H4–H7 not run. Branch `wisp`, not pushed.
Evidence: `docs/WISP_evidence/`. Hardware: RTX 4060 8 GB, Windows, env `h2snn`.
No number in this report is a performance result: H2S-NN is untuned and the baselines are tuned.

## H0 — Preconditions, inventory, pre-registration (2026-11-08)

### 1. Summary
- **Preconditions:** Part G is closed (G4 committed at `1c106d2`; no open units). There are no
  uncommitted changes in `src/` or `tests/`.
- **Inventory:** every consumer named in the card, with file:line (`WISP_evidence/h0_inventory.md`).
  There are no config-key collisions. The D12 goldens round-trip new `FaceGroup` fields safely.
- **Data facts (VERIFIED-RUN):** no duplicate facts on ACM, DBLP or IMDB, so "uniform unvisited
  neighbour" equals "uniform over the action's facts". The `r` budgets are 403,122 / 862,788 /
  143,052 incidences.
- **Pre-registration:** `WISP_evidence/preregistration.md` (H1–H6 with your reduced settings).
  It raises three decisions; two of them block H5 only.

### 2. Files changed
- `docs/WISP_evidence/h0_inventory.md`, `preregistration.md`, `h0_data_facts.py/.json` (new).
- This report (new).
- **D2 rows (not committed, as instructed):**
  - `H2S-NN/AGENTS.md:72` and `H2S-NN/CLAUDE.md:734` now read "WISP (spec v1.3 §3.2) …", with
    the interim default `path_source: schemas` until the H7 decision.
  - Status: "Agreed 2026-11-08; implementation: task card Part H, branch `wisp`".

### 3. Deviations and author overrides
**Overrides by the author** (message of 2026-11-08):
- H3 uses the 5-epoch ACM smoke checkpoint and 20 draws.
- H4 uses T = 3.
- H5 runs under 1 minute per run with ≤ 100 epochs. It has 3 seeds for `full` and for the added
  oracle-weight arm, and 1 seed for the other arms (+2 only if a control reaches the pass level).
- H6 is reduced: ladder values on ACM only, 2 epochs; DBLP/IMDB with `full` only; LP/GC 1 epoch;
  cost on ACM only.
- The full suite runs after H2, H6 and H7.
- No accuracy is reported. The spec §6 WISP gate stays closed.

**Deviation: branch base.** `wisp` starts at `19a9d36`, not `1c106d2`.
- `19a9d36` is your docs-only commit on top of `1c106d2`: specs, plan, task cards, AGENTS.md,
  CLAUDE.md. Code is identical.
- Branching at `1c106d2` would have removed Part H and your edits from the working tree.

**AGENTS.md and CLAUDE.md** were committed by you in `19a9d36`, so they no longer hold uncommitted
edits of yours. My D2-row edits to them are left uncommitted.

### 7. Assumptions imposed for theory and reality checks
- **S1 as carded.** Its rank argument assumes positive per-edge weights. The transports are signed
  orthogonal maps conditioned on each edge's endpoints, and they can express a 3-bit parity, so
  `endpoints` may solve S1 without intermediate-aware weights (DERIVED in the pre-registration).
- **Reading the ladder.** In this model the intermediates shape every arm's transport, so the C2
  ladder isolates the weight, not "whether intermediates matter" in general.

### 9. Decisions needed from the author
1. **S1:** run the parity S1 as carded (with the recorded prediction of an `endpoints` miss), the
   key-matching S1′ (G3c v2, where the expected miss is learning; the oracle shows capacity), or
   both?
2. **S2 criterion:** I_σ cannot rank σ\* strictly first (its sub-path schemas receive at least its
   weight, by Prop. K). The options are: I_π first; I_σ first among length-4 schemas; lift
   relative to initialisation; or a combination.
3. **Branch base** `19a9d36` instead of `1c106d2`: OK?


## H1–H3 — Sampler, weights and ladder, adaptive proposal (progress record, 2026-11-08)
Your H0 decisions are recorded as a dated addendum in `WISP_evidence/preregistration.md`. That
addendum covers the S1 expectation change, S1′, the S2 criteria, the note on what the C2 ladder
isolates, and the branch base. Commits: H1 `683fe44`, H2 `027a58c`, H3 `5c39d1c`.

- **H1 PASS** (`683fe44`): schema-free sampler with exact float64 counts and length grouping.
  - T-H1, T-H2 (q = p), T-H4 and T-H9 pass; mutations allow-revisits and drop-summed-merge are both
    caught. The test file is `tests/test_wisp.py`.
  - **r = 42 on ACM** (411,257 incidences ≥ 403,122).
  - **T-H10:** draw time is linear in r (R² = 0.9992). The only Python loops are over hops and
    visited positions.
- **H2 PASS** (`027a58c`): `path_weight: exp` (the default; γ removed, b in the scalar group),
  the C2 ladder, and per-cell action embeddings for WISP groups (relation plus hop direction).
  - The D12 goldens are bit-identical under `sigmoid_gamma` + `schemas`.
  - T-H5…T-H8 pass; the endpoints mutation is caught.
  - The suite found a real bug: a saved default made configs unreloadable. It is fixed.
  - Full suite: 329 tests, only the 24 pre-existing failures.
- **H3** (`5c39d1c`): the adaptive walk-level defensive proposal.
  - T-H2 (biased q, plus the no-IS control) and T-H3 (an independent reference plus the per-step
    control) pass. All 5 mutation checks are caught.
  - **Variance gate MISSED:** V(adaptive) 963.7 vs V(uniform) 1013.8, a −4.9 % reduction against
    the required −20 % (5-epoch ACM checkpoint, 20 draws, 16 probes; `h3_variance_gate.json`).
    `wisp_proposal` stays `uniform`.
  - This is a stop condition of Part H.

**Prepared but not committed:** `src/analysis/wisp_readouts.py` (H4) and
`WISP_evidence/h5_synthetic.py` (H5), with their tests in the session scratchpad.

**Decision needed:** continue with H4–H7, with the default staying `uniform` as the card
prescribes? Or stop here?
