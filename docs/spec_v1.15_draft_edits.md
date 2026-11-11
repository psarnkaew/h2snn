# Spec v1.15 — draft edits (experiment settings, round 2), for the author's approval

**Status:** APPLIED 2026-11-12 as spec v1.15 (author approval 03:47 JST); backup of v1.14: `docs/spec_backups/H2S-NN_methodology_spec_v1.14.md`. Written against spec **v1.14** on disk (md5 `b21d7914244a5237ca1054eaa0eb377e`, modified 2026-11-11 09:03 JST; the file is re-read immediately before applying, and these edits are merged onto any newer version). Nothing in G17 (incidence poset) or in the method's equations is touched.
**Source of the decisions:** `docs/experiment_settings.html` (round 2, §4.1–4.3, §5.2) and `docs/experiment_settings_decisions.md` (rows of 2026-11-11 "chat (2026-11-11d)" and 2026-11-12).
**Scope:** WISP on every task, the H2S-NN search space and its fixed constants, the shared training protocol of every method, HGB's link-prediction negatives, the GC readout, the baseline list, the heterophily computation, stale wording. The method's energy, solver and theorems are unchanged.
**Closed 2026-11-12 (author):** O1 = (a) ℋ² with H2GB's own meta-path rule; O2 = the four H2GB datasets are added; every heterophily value is recomputed and no earlier value (ours or published) is reused.
**OPEN (marked in the text; not decisions yet):** (O3) `feats_type` in five baseline spaces and the LP decoder `concat_mlp`; (O4) GC epochs 200 or 500; (O5) mini-batch design (blocks the H2GB datasets, the two large LP tasks and `r`'s upper bound on large graphs).

---

## Y-1. §1 Meta-path set and graph (bullet "Meta-path set and graph (1.14, replaces 1.8)")
**Old (fragment):**
> The same `M₂` is used for MLH, ℋ² (H2GB's computation fed this set) and MDE. [...] The only check is the reproduction of Hetero²Net Table 1 (DBLP/ACM/IMDB: MLH 0.87/0.60/0.24, MDE 0.80/0.74/0.59) within a tolerance fixed before running. Values published by H2GB (ℋ², MLH on its own rule) are cited as context, not as a reproduction target.

**New (fragment):**
> **Each measure is computed with its source's code, exactly as published (1.15).** MLH and MDE with Hetero²Net's code (`homophily.py`) on Hetero²Net's `M₂` (its list where it has one, else `get_metapaths`). ℋ² with H2GB's code (`calcHomophily.py`) on **H2GB's own meta-path rule** (`extract_metapath`: every length-2 type sequence from target to target, same-type relations included), so MLH/MDE and ℋ² of one dataset may use different meta-path sets (stated in the caption). **Every value is recomputed** (author, 2026-11-12): no earlier value, ours (J0) or published, is reused or used as a check; no tolerance gate; a value is reported as "computed with <code> at <commit>".

## Y-2. §1 Where reported
**Add after the first sentence:**
> (1.15) Every dataset of every task is recomputed (NC, LP, GC), including those whose values are published; graphs of H2GB size are computed on the server. Freebase reports the labeled-only diagnostic beside the filled value (unchanged).

## Y-3. §3 component table, row "WISP sampler", and §3.2 (b) "Knobs"
**Old (§3.2 b, fragment):**
> Knobs: `ℓ_max` sets the **support** (a modelling choice, pre-registered per dataset; the shape of the target inside the support is largely absorbed by `g`, which sees the relation sequence) and `r` (precision only). The ½ is fixed (it gives the bound 2), never tuned.

**New:**
> Knobs: `ℓ_max` sets the **support** and `r` (walks per anchor in one draw) the **precision**; both are hyperparameters in the search space (1.15, author: `ℓ_max ∈ {2, 3, 4, 5}`, `r ∈ {8, 16, 32, 64}`; supersedes E11, E14 and the pre-registered `ℓ_max` of §6). The ½ is fixed (it gives the bound 2), never tuned.

**Add to §3.2 (a), after "Anchors: every node of the target type (node classification).":**
> Link prediction: every node of the two endpoint types of the target relation, on the message graph (held-out facts removed in both orientations). Graph classification: every node of the graph. WISP is the path source of **every** task (1.15); the length-2 schema-free walks previously used by the LP and GC trainers are not part of the method.

## Y-4. §3 component table, row "Readout `[x*, h₀]`" — add task readouts
**Add after the table (new paragraph):**
> **Task readouts and losses (1.15).** NC: `[x*, h₀]` → Linear(2d, 128) → GELU → Dropout → Linear; cross-entropy (IMDB: per-genre BCE, genre predicted iff logit > 0). LP: per endpoint Linear(2d, d) → GELU, then the decoder [O3: {dot, DistMult} or {dot, DistMult, concat_mlp}]; BCE with 1:1 training negatives. GC: `[x*, h₀]` of every node → pooling (mean, max, add, or attention as HetSheaf's `AttentionPooling`: Linear → Tanh → Linear, 2 heads averaged) → **one Linear layer**; no BatchNorm, no MLP (as HetSheaf's `GraphClassifier` and every GC baseline); cross-entropy. All fixed constants of the method that are not tuned are listed in `docs/experiment_settings.html` §4.2.

## Y-5. §6 Node classification (datasets)
**Old (fragment):**
> H2GB IEEE-CIS-G, mag-year, ogbn-mag, H-Pokec with their loaders and splits; [...] No further H2GB dataset is added (author, D4).

**New (fragment):**
> H2GB IEEE-CIS-G, mag-year, ogbn-mag, H-Pokec, **oag-cs, oag-eng, oag-chem, PDNS** (1.15, author; 12 NC datasets in all) with their loaders and splits (data under `H2S-NN/data/NC/`). OAG labels are the journal venue (`PV_Journal` edges); venue nodes and edges are not in the message graph (H2GB's loader skips them), checked by the leak audit; H2GB's `oag_dataset.py:55-57` maps `engineering` → `OAG_Chemistry.pt` and `chemistry` → `OAG_Engineering.pt`, so each OAG file's identity is verified against H2GB Table 1 (node and edge counts) before any number. PDNS follows H2GB's temporal protocol (training graph over the training windows, one test graph per later window with previously labelled domains removed); it is the one inductive NC dataset.

## Y-6. §6 IMDB protocol
**Old (fragment):**
> validation macro-F1 selects checkpoints and is the statistic of the paired tests.

**New (fragment):**
> validation macro-F1 selects checkpoints. A genre counts in Macro-F1 when at least one movie of that split carries it (§6 averaging rule).

## Y-7. §6 Link prediction, Metrics (1.14 bullet)
**Old (fragment):**
> test negatives are 2-hop neighbours of the head, 1:1 with the test positives, drawn within the target relation's tail type [...] To check before running: the share of 2-hop draws that fall back to random on cross-type tasks; agreement of our 2-hop sampler and MRR with HGB's own code (not available locally, `vendor/HGB` is empty);

**New (fragment):**
> test negatives follow HGB's `get_test_neigh` (`LP/benchmark/scripts/data_loader.py`) in every point: 2-hop candidates from the symmetric adjacency of train + validation + test facts, direct links and self-loops removed, filtered to the relation's head and tail types, drawn **with replacement**, as many as the head's test positives, seed 1; a head without 2-hop candidates gets **no** negatives (no random fallback), and the number of such heads is reported per task. Validation negatives follow `get_valid_neg`: one random tail of the correct type per positive. Training: batches of 8,192 positive links, 40 epochs, patience 40 (HGB's LP defaults); for H2S-NN one step is one solve on the whole message graph. To check before running: agreement of our MRR with HGB's description (`src/utils/misc.py:178-236`);

## Y-8. §6 Graph classification
**Old (fragment):**
> Readout: the node-classification readout `[x*, h₀]` pooled over the graph's nodes (pooling ∈ {mean, max, add, attention}, as HetSheaf's baselines) and an MLP; no new readout component;

**New (fragment):**
> Readout: `[x*, h₀]` pooled over the graph's nodes (pooling ∈ {mean, max, add, attention}, as HetSheaf's baselines) and one Linear layer, as HetSheaf (1.15; no BatchNorm, no MLP); no new readout component; [...] Epochs (200 or 500): [O4] decided at training time.

## Y-9. §6 Tuning — replace the H2S-NN space sentence
**Old (fragment):**
> H2S-NN's space: `d`, `lr`, `weight_decay`, `encoder_dropout`, `classifier_dropout` (ranges in `docs/experiment_settings.html` §4.2); `K` and `layer_dropout` are removed (no `K` in v3-B; `layer_dropout` is not used by the anchored path). Epoch and patience budgets of HGB, H2GB and GC: decided at training time, the same for every method within a task.

**New:**
> **Rule (1.15, author):** the search space of H2S-NN contains only components of the method; constants of the method are fixed and listed once (`docs/experiment_settings.html` §4.2); keys that are not part of the method are never in the space. H2S-NN's space: `d` ∈ {16, 32, 64}; `ℓ_max` ∈ {2, 3, 4, 5}; `r` ∈ {8, 16, 32, 64}; GC pooling ∈ {mean, max, add, attention}; and the shared axes below. `K`, `layer_dropout` and every legacy key (gates, filter bank, old samplers, diagnostics) are refused by the tuner.
> **Shared by every method (1.15, author):** optimizer AdamW (baselines whose original code uses Adam switch, stated in the paper); `lr` log-uniform [1e-4, 1e-2]; `weight_decay` ∈ {0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 0.1, 0.8}; dropout (hidden and input) uniform [0, 0.9]; no learning-rate scheduler; no gradient clipping; batch size not tuned (HGB NC full graph; LP 8,192 positive links; GC 1; H2GB per the mini-batch design [O5]). Epochs per source protocol: HGB NC at most 300 with patience 30; LP 40 with patience 40; H2GB 500 without early stopping; GC as HetSheaf. NC and LP: early stopping and checkpoint on the validation metric (NC Macro-F1, LP ROC-AUC), not validation loss; GC: as HetSheaf (validation loss, final weights). Baseline-specific axes keep each adapter's space, which contains the published values [O3: `feats_type`]. The scalar group of the method (θ_τ and `b`: 10× lr, no weight decay) is part of the method, not of the shared protocol.

## Y-10. §6 WISP settings (pre-registered, never tuned)
**Old (fragment):**
> **WISP settings (pre-registered, never tuned):** `ℓ_max = 4` on HGB (covers every literature schema of ACM, DBLP, IMDB, incl. DBLP's length-4 schemas); `r` = the walks per anchor that matches the incidence budget of the fixed-literature-schema run on the same dataset (reported; never in a search space; recalibrated by the same rule for each `ℓ_max` of the U3 curve, 1.6); `proposal` = `uniform`

**New (fragment):**
> **WISP settings (1.15):** `ℓ_max` and `r` are tuned (Y-3). The incidence-budget rule for `r` (E11, E14) and the U3 calibration apply only to the fixed-literature-schema comparison row of the discussion part and are restated when that part is designed. `proposal` = `uniform`

## Y-11. §6 Solver settings
**Old (fragment):**
> `τ = e^θ`, `θ` init 0, scalar optimizer group at 10× the base learning rate; `τ > 50` stops.

**New (fragment):**
> `τ = e^θ`, `θ` init 0, scalar optimizer group at 10× the base learning rate (constant: no scheduler, 1.15); `τ > 50` stops; the log-weight range of `g` above 30 stops (§3). Both stop conditions are implemented and tested before the first protocol run (not found in `src/` on 2026-11-12).

## Y-12. §6 Baselines
**Add after the first sentence:**
> (1.15) Baselines: GCN, GAT, ChebNetII, NSD, CSNN, HAN, SeHGNN, HINormer, PSHGCN, H2G-former, Hetero²Net, H2SGNN, HetSheaf. LatGRL is removed: it is unsupervised (label-free encoder, then a probe on frozen embeddings), unlike every task of this protocol. Hetero²Net stays (supervised; its masked terms are auxiliary losses).

## Y-13. Superseded wording (no new content)
- §0 changelog entries E11, E14, F3's "No further H2GB dataset" and 1.14's "within a tolerance fixed before running" stay as history; the 1.15 entry names what supersedes them.
- §6 "WISP gate" and the S1′ capacity text are unchanged (discussion part).

---

## Proposed changelog entry (prepended above the 1.14 entry when applied)
> **Changes from 1.14 (to 1.15; experiment settings round 2, author decisions of 2026-11-11/12 recorded in `docs/experiment_settings.html` §4–§5 and `docs/experiment_settings_decisions.md`; the method's energy, solver and theorems are unchanged):** (J1) WISP is the path source of every task; anchors for LP (both endpoint types, message graph) and GC (every node); (J2) `ℓ_max` ∈ {2, 3, 4, 5} and `r` ∈ {8, 16, 32, 64} are tuned (supersedes E11, E14 and the pre-registered `ℓ_max`); (J3) H2S-NN's search space contains only method components; method constants listed once in `experiment_settings.html` §4.2; (J4) shared protocol for every method: AdamW, shared `lr`/weight-decay/dropout ranges, no scheduler, no clipping, batch size not tuned, epochs per source protocol, checkpoint on the validation metric (NC/LP); (J5) LP negatives exactly as HGB's `get_test_neigh`/`get_valid_neg`, no random fallback; LP trained in batches of 8,192 links for 40 epochs; (J6) GC readout: pooling and one Linear layer, as HetSheaf; (J7) LatGRL removed; (J8) heterophily recomputed for every dataset with each source's code (MLH, MDE: Hetero²Net; ℋ²: H2GB incl. its meta-path rule), no earlier value reused, no tolerance gate; (J9) four H2GB NC datasets added (oag-cs, oag-eng, oag-chem, PDNS; 12 NC datasets), with an identity check of the OAG files, a venue-leak audit and PDNS's temporal inductive protocol; number of Welch tests ≈ 48 (NC 24, LP 16, GC 8); (J10) stop conditions `τ > 50` and log-weight range > 30 to be implemented; (J11) stale IMDB wording ("paired tests") corrected.
