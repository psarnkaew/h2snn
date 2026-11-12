# Spec v1.16 — draft edits (after Part M), for the author's approval

**Status:** APPLIED 2026-11-12 as spec v1.16 (author approval 09:26 JST); backup of v1.15: `docs/spec_backups/H2S-NN_methodology_spec_v1.15.md`. Written against spec **v1.15** on disk (md5 `e4d46e8d62d97fd80ee749787955829c`); re-read immediately before applying, merged onto any newer version.
**Source of the decisions:** Part M report (`docs/H2S-NN_partM_report.md`, "Decisions for the author" 1–9) and the author's answers of 2026-11-12 08:20–08:33 (`docs/experiment_settings_decisions.md`). WISP is not touched (the author is consulting his advisor on predefined meta-paths).
**Scope:** LP test and training negatives exactly as HGB's runner; GC Macro-F1 as HetSheaf; stop conditions in tuning and a non-finite stop; correction of 1.15's "symmetric adjacency" (the designer's error: HGB adds validation facts in their stored orientation only).

---

## Z-1. §6 Link prediction, Metrics — HGB's two test sets, HGB's adjacency, HGB's training negatives
**Old (fragment, 1.15):**
> test negatives follow HGB's `get_test_neigh` (`LP/benchmark/scripts/data_loader.py`) in every point (1.15): 2-hop candidates from the symmetric adjacency of train + validation + test facts, direct links and self-loops removed, filtered to the relation's head and tail types, drawn with replacement, as many as the head's test positives, seed 1; a head without 2-hop candidates gets no negatives (no random fallback), and the number of such heads is reported per task.

**New:**
> test negatives follow HGB's runner (`LP/benchmark/methods/baseline/run_new.py`) in every point (1.16): **every task is evaluated on two test sets**, both reported. (i) **2-hop** (`get_test_neigh`): candidates from the adjacency of train and test facts in both orientations plus validation positives **in their stored orientation only** (as HGB's code; corrects 1.15's "symmetric"), direct links and self-loops removed, filtered to the relation's head and tail types, drawn with replacement, as many as the head's test positives, seed 1; a head without 2-hop candidates gets no negatives and the share of such positives is reported. On dblp_ap and imdb_ma no head type reaches the tail type in two hops (schema fact), so their 2-hop set has no negatives and its metrics are reported as "–" with that reason (never as MRR = 1). (ii) **Random** (`get_test_neigh_w_random`): one random tail of the tail type per test positive, redrawn while it is a known neighbour of the head (train, validation or test), seed 1.

**Add after "Validation negatives follow `get_valid_neg` (...); model selection on validation ROC-AUC.":**
> Training negatives follow HGB's `get_train_neg` (one random tail of the tail type per training positive, known positives not excluded), drawn as HGB's runner draws them (per epoch or once, as its code does; recorded), with the run's seed, for every method (1.16).

## Z-2. §6 Seeds and statistics — number of tests
**Old:** `(≈ 48: NC 24, LP 16, GC 8; 1.15)`
**New:** `(≈ 60: NC 24, LP 28 = 8 tasks × 2 metrics × 2 test sets − 4 undefined 2-hop cells, GC 8; 1.16)`

## Z-3. §6 Graph classification — metric definition
**Old (fragment):**
> validation Macro-F1 selects configurations; test Macro-F1 and Micro-F1.

**New (fragment):**
> validation Macro-F1 selects configurations; test Macro-F1 and Micro-F1, both computed as HetSheaf's code computes them (torchmetrics `F1Score(task="multiclass", average="macro" | "micro", num_classes=C)`, version recorded); the per-split class rule of node classification does not apply to GC (1.16). Batch size 1 is kept; its cost is carried by the lab cluster (author, 2026-11-12).

## Z-4. §6 Solver settings — stop conditions
**Old (fragment, 1.15):**
> `τ > 50` stops; a log-weight range of `g` above 30 stops (§3). Both stop conditions are implemented and tested before the first protocol run (not found in `src/` on 2026-11-12).

**New:**
> `τ > 50` stops; a log-weight range of `g` above 30 stops (range = max − min of `log w_f` over the cells of one forward, §3); a non-finite `g(ξ_f)` stops (fp32 `exp` overflows above ≈ 88.7) (1.16). Implemented in Part M (M6) and Part N.

## Z-5. §6 Tuning — failed trials
**Insert before "**Shared by every method (1.15, author):**":**
> **Failed trials (1.16, author):** a trial in which a stop condition fires (`T_max`, `τ > 50`, log-weight range > 30, non-finite `g`) or, for any method, the training loss becomes non-finite, is recorded as failed: it gets no score, the Bayesian search continues, and the number of failed trials is reported per method and dataset.

---

## Proposed changelog entry (prepended above the 1.15 entry)
> **Changes from 1.15 (to 1.16; after Part M, `docs/H2S-NN_partM_report.md`; author decisions of 2026-11-12 in `docs/experiment_settings_decisions.md`; the method is unchanged):** (K1) LP evaluated on HGB's two test sets, 2-hop and random, as HGB's runner; 2-hop is "–" on dblp_ap and imdb_ma (no length-2 head→tail type path); (K2) HGB's adjacency for 2-hop candidates adds validation positives in their stored orientation only (corrects 1.15 J5 "symmetric"); (K3) LP training negatives as HGB's `get_train_neg`; (K4) GC Macro/Micro-F1 as HetSheaf's torchmetrics call; GC batch size 1 kept (lab cluster); (K5) stop conditions: range defined per forward, non-finite `g` added; a trial that hits a stop condition or a non-finite loss is failed (no score) and counted; (K6) ≈ 60 Welch tests.
