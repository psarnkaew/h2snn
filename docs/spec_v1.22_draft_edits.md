# Spec v1.22 — draft edits (constants of mini-batch mode; LP decoder), for the author's approval

**Status:** APPLIED as spec v1.22 (author approval 2026-11-13); was: awaiting approval of the wording (decisions N1–N6 approved by the author on 2026-11-13, updated after the Part MB review (D-MB1, D-MB2, analysis flag; 2026-11-13), `docs/experiment_settings_decisions.md`). Written against spec **v1.21** on disk (md5 `73fae0e13da59886563a578367305ec6`); the anchor of W-1 occurs exactly once (checked 2026-11-13); back up to `docs/spec_backups/H2S-NN_methodology_spec_v1.21.md`. **The method is unchanged.**

## W-1 §6 Node classification, Training mode — constants of mini-batch mode
**Insert after:** `a fresh training draw per batch; PCG inner products and stopping per scope.`
> (1.22, author) Constants of mini-batch mode, never tuned: walks are drawn in chunks of 4,096 anchors, each chunk with its own generator seeded `seed·2²⁰ + k` for chunk `k`; the chunk changes which random numbers a walk uses, never `p(ω|t)`, and full-batch draws are not chunked; training targets are taken in a seeded permutation (seed `base·100,003 + pass`) that continues across epochs and is reshuffled when exhausted; a batch never crosses a pass boundary: the last batch of a pass holds the remaining targets and counts as one step (as H2GB's `LoaderWrapper` and PyG loader); the training draw of batch `b` in epoch `e` has seed `(base·100,003 + e)·65,536 + b` (at base seed 0 the first training draw shares the evaluation draw's seed; this changes no `p(ω|t)` and is kept); evaluation runs in batches of 1,024 targets; scopes are built on the graph they are evaluated on (PDNS: each test window's graph). An analysis flag runs ACM and DBLP in mini-batch for the gap of L5 (C5), with B = all training targets (one step per epoch), evaluation every epoch and the full-batch budget of the dataset, so that only the scope differs; the tuner refuses it. The batch size and epoch of the baseline adapters on the H2GB datasets are implemented with the H2GB loaders (the Part M M3 card).

## W-2 §3 Task readouts and §6 search space — H2S-NN's LP decoder (author decision 2026-11-13, decisions log "H2S-NN LP decoder")
**W-2a, §3 "Task readouts and losses (1.15)". Replace:** `then the decoder (dot or DistMult; `concat_mlp` open);`
**with:**
> then the decoder, tuned in {dot, DistMult}, the LP head shared with every baseline (1.22, author; `concat_mlp` is removed: no baseline has it and HGB uses only dot and DistMult);

**W-2b, §6 Tuning, H2S-NN's space. Replace:** `GC pooling ∈ {mean, max, add, attention}; and the shared axes`
**with:**
> GC pooling ∈ {mean, max, add, attention}; LP decoder ∈ {dot, DistMult} (1.22; the tuner refuses `concat_mlp`); and the shared axes

Implementation (tuner always tunes the decoder on LP tasks, refuses `concat_mlp`; test) is a unit of the M3 card. Anchors W-2a and W-2b each occur exactly once in spec v1.21 (checked 2026-11-13, md5 `73fae0e13da59886563a578367305ec6`, unchanged since W-1 was drafted).

## Proposed changelog entry (prepended above the 1.21 entry; version line → 1.22)
> **Changes from 1.21 (to 1.22; author decisions of 2026-11-13: N1–N6, the Part MB and Part MB-fix review rows, the H2S-NN LP decoder row; `docs/experiment_settings_decisions.md`, draft `docs/spec_v1.22_draft_edits.md`; the method is unchanged):** (W1) mini-batch constants: sampler chunks of 4,096 anchors (mini-batch only), seed formulas, seeded continuing permutation of training targets with batches inside a pass (H2GB's loader), evaluation batches of 1,024, scopes on the evaluated graph, an analysis flag for the L5 gap on ACM/DBLP (one batch of all training targets, full-batch budget); baselines' H2GB batching moves to the M3 card. (W2) H2S-NN's LP decoder is tuned in {dot, DistMult}, the head shared with every baseline; `concat_mlp` removed.
