# H2S-NN — WISP report (task card Part H)

**Status (2026-11-08):** H0 done; waiting for the author's reply before H1. Branch `wisp`, not pushed.
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
