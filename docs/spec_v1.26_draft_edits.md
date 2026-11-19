# Spec 1.26 — draft edits (AA-1 – AA-10), APPROVED and APPLIED 2026-11-19

Written 2026-11-19 by the model session. Base: `docs/H2S-NN_methodology_spec.md` v1.25 (design-docs). The method is unchanged in 1.26: every item is protocol or wording. Sources: decisions log rows "Card 2 review (dry run, `9d48ca4`)" (2026-11-16: B1a, B2a, range rule 1), "Card 1f (B1a baselines) and card 6 (heterophily harness) written" (2026-11-17), "Card 1e F3 + 1f review" and "Card 1f U5 review" (2026-11-18: PDNS (a), sweep objective), "TSUBAME limits", "TSUBAME card decisions 1–4" and "Model change: `R_r` init, H2GB timing, bounded `g`" (2026-11-18 (b): N comes from the first TSUBAME trial). Author input of 2026-11-19: the memory rule uses the **H100 94 GB** of TSUBAME, not 24 GB.

**Decided 2026-11-19:** AA-5 (N = 100) and AA-6 (large LP use N). **Pending:** the meaning of "values from published papers" (AA-5), and approval of AA-1–AA-4 and AA-7–AA-9.

---

## AA-1 Header: version and change line
- **Replace** `**Version:** 1.25 (2026-11-18).` **with** `**Version:** 1.26 (2026-11-19).`
- **Insert before** `**Changes from 1.24 (to 1.25;` **a new paragraph:**
> **Changes from 1.25 (to 1.26; author decisions of 2026-11-16 to 2026-11-19 in `docs/experiment_settings_decisions.md`, draft `docs/spec_v1.26_draft_edits.md` AA-1–AA-9; the method is unchanged):** (AA1) §1: graph classification reports no heterophily value (1.23 X9 had not reached §1). (AA2) §6: validation during training on the H2GB NC datasets uses the fixed B1a subset (same nodes for every method; PDNS per run seed); the trial is selected by the subset Macro-F1. (AA3) §6: 100 trials per (method, dataset) on every task, including H2GB NC, the large LP tasks and GC (B2a's reduced N withdrawn); a projected wall time from the first TSUBAME trials is recorded before those cells are tuned. (AA4) §6: search-range rule under memory limits, for every method, on one H100 94 GB GPU (replaces the 24 GB rule of 2026-11-16). (AA5) §6: the server is TSUBAME 4.0: environment and timing GPU stated.

## AA-2 §1 Heterophily, "Where reported": graph classification
**Replace** `Graph classification reports the 1-hop `H_edge` on the original bonds with label = atom type (the label-homophily formula of Hetero²Net without a meta-path), marked "1-hop, label = atom type, not the graph class"; MLH over meta-paths would be 1 by construction there (1.14).` **with:**
> Graph classification reports no heterophily value (1.23 X9): its node "types" are atom labels and the class belongs to the graph, not to a node, so no label-based measure applies; the type-mixing ratio (share of bonds between different atom types) is reported instead, marked as such (1.26).

## AA-3 §6: new bullet after "Node classification" — B1a validation subset
**Insert after the bullet beginning** `- **Node classification:** HGB ACM, DBLP, IMDB` **:**
> - **Validation during training on the H2GB NC datasets (B1a; author 2026-11-16, card 1f; 1.26).** Checkpoint selection and the tuning objective use a **fixed subset** of the validation split of H2GB's own size, `min(split, size)`: ogbn-mag 16,384, mag-year 8,192, oag-cs, oag-eng, oag-chem 4,096, IEEE-CIS-G 8,192, PDNS 8,192, H-Pokec 16,384. The subset is drawn with seed 0 and is the same for every method (`src/evaluation/validation_subset.py`, shared by the baselines and H2S-NN). PDNS draws its split from the run seed (H2GB's code), so its subset is drawn from that seed's validation split with an RNG seeded by (0, run seed): the same run seed gives the same nodes for every method. No test pass runs during training. The selected checkpoint is evaluated once on the whole validation and test splits, and those numbers are reported. In the search, the trial is selected by the subset Macro-F1 (the sweep objective), for every method. The HGB datasets and link prediction are unchanged (whole validation split).

## AA-4 §6 Tuning: the objective on H2GB NC
**Replace** `tuning on seed 0 only, by the validation metric;` **with** `tuning on seed 0 only, by the validation metric (on the H2GB NC datasets: the B1a subset Macro-F1, bullet above);`.

## AA-5 §6 Tuning: trial count — N = 100 everywhere (author 2026-11-19; B2a's reduced N withdrawn)
**Author 2026-11-19:** N = 100, and the two large link-prediction tasks use N (AA-6 option (a)). Every task therefore keeps the 1.14 rule, "100 trials per (method, dataset)", and the B2a rule "N from central-point timings" is withdrawn. The wording of the Tuning bullet stays as it is. **Insert after the Tuning bullet:**
> - **Trial count on the large tasks (1.26, author 2026-11-19).** The H2GB NC, the two large LP and the GC datasets keep 100 trials per (method, dataset), as every other task (B2a's reduced N of 2026-11-16 is withdrawn). The first TSUBAME trial of every method at its central point on these cells gives a projected wall time, which is recorded in the decisions log before their tuning starts. If the projection ends after 2026-12-20, the author decides before any of these trials runs; trials are never cut silently, and no search is stopped early.

**Answered (author 2026-11-19):** option (b), published result numbers as placeholders; see AA-10.

## AA-6 Large link prediction — decided: option (a), they use N (= 100)
See AA-5.

## AA-7 §6: memory rule (range rule 1) on H100 94 GB, for every method
**Insert, as a new bullet after the N bullet (AA-5):**
> - **Search range under memory limits (range rule 1; author 2026-11-16, GPU changed 2026-11-19; 1.26).** On every cell, for every method, the search range is the largest grid point that trains on **one H100 SXM5 94 GB GPU** of TSUBAME 4.0 (one GPU per trial, with a quarter of a `node_f` node's host memory). An axis is stepped down one grid step at a time, in the order stated per method, and the range is stated per cell. This replaces the 24 GB rule of 2026-11-16. Ranges and N/A cells that were derived on 24 GB GPUs are derived again on TSUBAME before their cells are tuned. A cell that does not fit even at the grid minimum is N/A with the reason. H2S-NN's Freebase NC runs full batch if the largest configuration fits by this rule, else mini-batch (§6 "Ablation statistics"); the same check decides.

**Consequence for the ops session (not spec text):**
- The 24 GB OOM ladders of card 3 A5 and the 24 GB N/A cells must be re-derived on H100: HAN LP `freebase_bb`, and HAN NC Freebase on CPU because of GPU memory.
- TSUBAME decision (4) already plans this ("revisited at phase 2"). Phase 1 (HGB NC + LP baselines) should not be affected; the ops session confirms that no phase-1 range was stepped down on 24 GB.

## AA-8 §6 Environment and timing GPU
- **Replace** `- **Environment (1.23).** Every run uses conda env `h2snn-b` on the lab cluster:` **with** `- **Environment (1.23; 1.26: server).** Tests, known-answer runs and references run on the lab cluster; main-table, ablation, analysis and timing runs run on TSUBAME 4.0 (H100 SXM5 94 GB; `docs/tsubame_guide.md`). Both use conda env `h2snn-b` built from the same lock:`
- **Replace** `comes from separate timing runs on one GPU type, stated in the paper;` **with** `comes from separate timing runs on one GPU type (1.26: H100 SXM5 94 GB on TSUBAME), stated in the paper;`

## AA-9 §6 Freezing: base commit
**Append to the Freezing bullet:** ` (1.26) The H2S-NN freeze is a commit of `v3b` at or after model 1.25 (`894fd74`) that includes card 5; no TSUBAME run of H2S-NN uses an earlier commit.`

---

## Not in the spec (card 5, after approval)
- **B1a and PDNS in H2S-NN's H2GB NC loop.** It uses the shared `validation_subset`, records the subset metric and fingerprint, runs no test pass during training, and restores the best weights for the whole-split evaluation. The sweep objective is `val_subset_macro_f1`, as card 1f U5/U6 does for baselines.
- **Freebase NC.** A full-batch or mini-batch switch, decided by the AA-7 check.
- **N2b analysis hooks.** `τ` is fixed ∈ {1, 10, 100}, the `τ > 50` stop is lifted and `T_max` = 500, for N2b runs only.
- **H2S-NN TSUBAME cell list.** Coordinate with the ops session, because the TSUBAME card owns the launcher.

## AA-10 §6: published numbers as placeholders (author 2026-11-19, option (b))
**Insert after the memory-rule bullet (AA-7):** the bullet "Published numbers as placeholders (1.26, author 2026-11-19)", as applied in the spec. The register is `docs/published_number_register.md`.

**Applied 2026-11-19** (author: "approve ร่าง 1.26"): AA-1–AA-10 in spec v1.26. Backup: `docs/spec_backups/H2S-NN_methodology_spec_v1.25.md`.
