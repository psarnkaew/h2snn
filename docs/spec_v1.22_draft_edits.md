# Spec v1.22 — draft edits (constants of mini-batch mode), for the author's approval

**Status:** DRAFT, awaiting approval of the wording (decisions N1–N6 approved by the author on 2026-11-13, `docs/experiment_settings_decisions.md`). Written against spec **v1.21** on disk (md5 `73fae0e13da59886563a578367305ec6`); the anchor of W-1 occurs exactly once (checked 2026-11-13); back up to `docs/spec_backups/H2S-NN_methodology_spec_v1.21.md`. **The method is unchanged.**

## W-1 §6 Node classification, Training mode — constants of mini-batch mode
**Insert after:** `a fresh training draw per batch; PCG inner products and stopping per scope.`
> (1.22, author) Constants of mini-batch mode, never tuned: walks are drawn in chunks of 4,096 anchors, each chunk with its own generator seeded from (seed, chunk index); the chunk changes which random numbers a walk uses, never `p(ω|t)`, and full-batch draws are not chunked; training targets are taken in a seeded permutation that continues across epochs and is reshuffled when exhausted (as H2GB's `LoaderWrapper` continues its iterator); evaluation runs in batches of 1,024 targets; scopes are built on the graph they are evaluated on (PDNS: each test window's graph). An analysis flag runs ACM and DBLP in mini-batch for the gap of L5 (C5); the tuner refuses it. The batch size and epoch of the baseline adapters on the H2GB datasets are implemented with the H2GB loaders (the Part M M3 card).

## Proposed changelog entry (prepended above the 1.21 entry; version line → 1.22)
> **Changes from 1.21 (to 1.22; author decisions N1–N6 of 2026-11-13, `docs/experiment_settings_decisions.md`, draft `docs/spec_v1.22_draft_edits.md`; the method is unchanged):** (W1) mini-batch constants: sampler chunks of 4,096 anchors (mini-batch only), seeded continuing permutation of training targets, evaluation batches of 1,024, scopes on the evaluated graph, an analysis flag for the L5 gap on ACM/DBLP; baselines' H2GB batching moves to the M3 card.
