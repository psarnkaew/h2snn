# Draft spec edits → v1.17 (mini-batch design; author decisions of 2026-11-12)

**Status:** draft **approved by the author 2026-11-12 08:41 JST** (as `spec_v1.16_draft_edits.md` Z-0–Z-8 of this session; renumbered because another session's draft for v1.16, Part M decisions K1–K6, took the file name `docs/spec_v1.16_draft_edits.md` at 08:34). **APPLIED 2026-11-12 as spec v1.17** (on top of v1.16, md5 `64ceb5a3b81c17b4eef223157ea008b7`; backup `docs/spec_backups/H2S-NN_methodology_spec_v1.16.md`; every Old string matched exactly once; Z-7b first variant). Apply **after** v1.16 is in the spec: re-read `docs/H2S-NN_methodology_spec.md`, check every "Old" string occurs exactly once (Z-7b has two variants), back up to `docs/spec_backups/H2S-NN_methodology_spec_v1.16.md`, merge, never overwrite.
**Source:** `docs/minibatch_design_v3B.md` (MB0–MB6). **The method is unchanged** (energy, solver, theorems).

## Z-0 Changelog entry (insert above "Changes from 1.15") and header
**Changes from 1.16 (to 1.17; mini-batch design, author decisions of 2026-11-12 recorded in `docs/minibatch_design_v3B.md`, draft `docs/spec_v1.17_draft_edits.md`; the method is unchanged):** (Q1) mini-batch = one scope per target (lower set of the incidence poset: the target, its facts, the cells of its walks), with the scope's own degrees, context and normalisation; the predictor is the sheaf of each scope, not of the whole graph (L5); whole-graph alternatives considered and not adopted (`minibatch_design_v3B.md` §4.4); (Q2) Lemma R corrected: a batch equals one scope at a time **within the solver tolerance**, not bit for bit (CG amplifies round-off; checked); (Q3) training mode follows the source protocol: HGB NC, small LP tasks and GC full batch / per graph; the eight H2GB NC datasets and the two large LP tasks mini-batch; Freebase mini-batch only if full batch does not fit at the largest configuration; never a tuning value; (Q4) batch size is tuned on the eight H2GB NC datasets, `{32, 256, 1024}`, the same range for every method (supersedes "batch size not tuned" of J4 for these datasets); there 1 epoch = 8,192 training targets (H2GB's 256 iterations × 32); (Q5) every method is evaluated on the whole validation and test split, in batches (H2GB's code evaluates a random subset per evaluation; documented in `minibatch_design_v3B.md` §7.2); (Q6) large LP: each endpoint is a target with its own scope, the decoder scores `(x*_u, x*_v)`; (Q7) in mini-batch the evaluation warm start is off and the log-weight range stop condition is checked per forward, i.e. per batch. Supersedes `docs/minibatch_design_decisions.md` and `docs/minibatch_v3B_addendum.md` (kept as history; their v2 reasoning does not apply to v3-B).

Header: set **Version** to 1.17; in the "Open:" list of the 1.15 entry nothing is edited (history); the current open list (wherever v1.16 keeps it) drops "mini-batch design".

## Z-1 §2 R7
**Old:** `a mini-batch applies the same model to each target's closed local subcomplex |`
**New:** `a mini-batch applies the same model (same rule, same parameters) to each target's scope, a lower set of the incidence poset whose degrees, context and normalisation are its own (1.17) |`

## Z-2 §4 Lemma R row
**Old:** `| Lemma R (mini-batch) | A scope is a closed subcomplex; the §3 construction on it is a sheaf and Thms 1–8 hold on it (`minibatch_design_decisions.md` M1); PCG with per-scope inner products makes a batch bit-identical to one scope at a time | DERIVED |`
**New:** `| Lemma R (mini-batch; 1.17) | A target's scope (the target, its facts, the cells of its walks with their nodes and facts) is a lower set of the incidence poset; the §3 construction on it is a sheaf on that poset, and Thm 1, 3, 4, 5, 6, Props. 7/7′, Thm 8, Prop. 10, Prop. 11 hold on it, Thm 9 / Prop. S per anchor (`minibatch_design_v3B.md` §4.1). It is not the restriction of the whole-graph sheaf (context and degrees are the scope's), so Thm 3 and Thm 6 concern sections of the scope. PCG with per-scope inner products and stopping solves each scope of a batch to its own tolerance; a batch equals one scope at a time within the solver tolerance, not bit for bit (CG amplifies round-off: 1e-6–7e-5 relative at tol 1e-4 in float64 on a test of four blocks) | DERIVED; checked |`

## Z-3 §4 limits, L5
**Old:** `**L5** Mini-batch is the same model on a closed local scope; degrees and context are the scope's (M3).`
**New:** `**L5** On the eight H2GB NC datasets and the two large LP tasks each target is predicted by the same model, with the same parameters, on its own scope (its facts and the cells sampled from it); degrees, relational context and normalisation are the scope's, so a node with many facts outside the scope weighs more there than in the whole graph, and information that no fact of the target and no sampled cell carries is not seen. Training and evaluation use the same rule. The gap to full batch is reported on ACM and DBLP (C5) (1.17).`

## Z-4 §6 Node classification, mini-batch sentence
**Old:** `The mini-batch design is reopened (author, 2026-11-11): the M0–M4 design and its v3-B addendum are reference material until the author confirms them. ogbn-mag, mag-year, H-Pokec, oag-cs, oag-eng, oag-chem (and IEEE-CIS-G and PDNS if full batch does not fit) require mini-batch training (Lemma R; WISP cells drawn inside each batch as a closed subcomplex): a required engineering unit after Part I. `ℓ_max` is tuned on every dataset (1.15).`
**New:** `**Training mode (1.17, author).** The mode follows the source protocol and never depends on a trial: HGB ACM, DBLP, IMDB full batch; Freebase full batch if it fits at the largest configuration (`d` 64, `ℓ_max` 5, `r` 64), else mini-batch (stated); the eight H2GB datasets mini-batch (H2GB trains every dataset with a sampler). Mini-batch (Lemma R, L5; `docs/minibatch_design_v3B.md`): one scope per target; walks drawn on the whole graph by the unchanged sampler and cut per target (cells merged per anchor); one fixed evaluation draw per run (base seed) for the validation and test targets; a fresh training draw per batch; PCG inner products and stopping per scope. On the H2GB datasets 1 epoch = 8,192 training targets (H2GB's 256 iterations × 32), i.e. `8192/B` optimizer steps for batch size `B`, the same definition for every method. **Evaluation:** every method is evaluated on the whole validation and test split, in batches (H2GB's code evaluates `val.iter_per_epoch × 32` randomly drawn nodes per evaluation; documented in `minibatch_design_v3B.md` §7.2). PDNS: scopes and the evaluation draw are built on each test window's graph. `ℓ_max` is tuned on every dataset (1.15).`

## Z-5 §6 Link prediction, training sentence (not touched by v1.16's Z-1, which edits the Metrics fragment before it)
**Old:** `Training: batches of 8,192 positive links, 40 epochs, patience 40 (HGB's LP defaults); for H2S-NN one step is one solve on the whole message graph.`
**New:** `Training: batches of 8,192 positive links, 40 epochs, patience 40 (HGB's LP defaults); for H2S-NN on the six small tasks one step is one solve on the whole message graph; on ogbn-mag P–P and H-Pokec U–U (mini-batch, 1.17) each endpoint of the step's positive and negative links is a target with its own scope, and the decoder scores `(x*_u, x*_v)` as in full batch; evaluation computes the scopes of all validation/test endpoints once and scores every link.`

## Z-6 §6 Tuning: H2S-NN space, shared batch size, H2GB epochs
**Old:** `H2S-NN's space: `d` ∈ {16, 32, 64}; `ℓ_max` ∈ {2, 3, 4, 5}; `r` ∈ {8, 16, 32, 64}; GC pooling ∈ {mean, max, add, attention}; and the shared axes below.`
**New:** `H2S-NN's space: `d` ∈ {16, 32, 64}; `ℓ_max` ∈ {2, 3, 4, 5}; `r` ∈ {8, 16, 32, 64}; GC pooling ∈ {mean, max, add, attention}; and the shared axes below (including the batch size on the H2GB datasets, 1.17).`
**Old:** `batch size not tuned (HGB NC full graph; LP 8,192 positive links; GC 1; H2GB per the mini-batch design)`
**New:** `batch size: HGB NC full graph, LP 8,192 positive links, GC 1 (not tuned); on the eight H2GB NC datasets tuned in {32, 256, 1024} for every method (1.17, author; the effective batches of H2GB's configs: 32 targets × accumulation 1, 8 or 32)`
**Old:** `H2GB 500 without early stopping`
**New:** `H2GB 500 epochs of 8,192 training targets without early stopping, checkpoint at the best validation evaluation, evaluation every `eval_period` epochs of H2GB's config (4; OAG 2) (1.17)`

## Z-7 §6 Solver settings
**(a) Old:** `warm start from the previous epoch's `Y*` on the same evaluation draw`
**New:** `warm start from the previous epoch's `Y*` on the same evaluation draw in full batch; none in mini-batch (1.17)`
**(b)** If v1.16's Z-4 is applied (expected), **Old:** `(range = max − min of `log w_f` over the cells of one forward, §3)` **New:** `(range = max − min of `log w_f` over the cells of one forward, i.e. of one batch in mini-batch, §3; 1.17)`. If v1.16's Z-4 is not applied, **Old:** `a log-weight range of `g` above 30 stops (§3)` **New:** `a log-weight range of `g` above 30 stops (§3; per forward, i.e. per batch in mini-batch, 1.17)`.

## Z-8 References
Only the Lemma R row cites `minibatch_design_decisions.md` as current (handled by Z-2); the 1.11 changelog (G10) stays as history.

## After applying (not the spec)
- `docs/experiment_settings_decisions.md`: rows MB0–MB6 (2026-11-12), appended to the version on disk.
- `docs/experiment_settings.html` §1, §4.1, §4.3: update boxes.
- Baseline card: the shared batch range and epoch definition on H2GB for every adapter; evaluation on whole splits.
