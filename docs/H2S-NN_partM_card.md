# Task card — Part M: align the code with the experiment protocol of spec v1.15

**Author decisions:** 2026-11-11/12, recorded in `docs/experiment_settings_decisions.md`. **Design authority:** `docs/H2S-NN_methodology_spec.md` **v1.15** (changelog J1–J11) and `docs/experiment_settings.html` §4.1–§4.3, §5.2, §6. Where `H2S-NN/AGENTS.md` / `CLAUDE.md` name older documents as sources of truth (v2 plan, fix spec), the spec v1.15 governs design questions; environment, coding and testing rules of `AGENTS.md`/`CLAUDE.md` still bind.
**Branch:** `part-m`, created from `v3b` at the commit made in M0 (record the hash in the report). **Report:** `docs/H2S-NN_partM_report.md`, one section per unit.

## Goal
Make the code run the protocol of spec v1.15 on every task, and compute the heterophily statistics of every dataset. Correctness, function and cost only: **no tuning sweeps, no multi-seed runs, no performance numbers.**

## Scope rules
- Every unit: one commit; tests first or with the change; mutation check where marked; report section with VERIFIED-RUN output.
- Never tuned / never changed by this card: the method's energy, solver, maps, WISP target measure and theorems; `proposal`, `path_weight_input`, solver tolerance schedule, `T_max`, `τ` init.
- Open (do **not** change): `feats_type` in baseline spaces; LP decoder `concat_mlp`; GC epochs 200 vs 500; mini-batch design. If a unit needs one of them, stop that unit and list it under "Decisions for the author".
- Forbidden: loosening tolerances, deleting/skipping tests, silencing exceptions, special-casing inputs, editing the spec or the HTML to fit the code. Out-of-scope problems are listed, not fixed.

**Deferred (author, 2026-11-12): M2 and M3** move to after the mini-batch design. H2GB datasets need preprocessing that the laptop cannot do; they will be prepared on the lab cluster. In this card: no H2GB data is loaded, no heterophily statistics are computed; M2 and M3 below are kept as the specification for that later card.

## Units (in this order; a unit blocked by missing data is skipped and the next one starts)

### M0 — Preconditions (no code)
Read spec v1.15 (whole file), `experiment_settings.html` §4–§6, the decisions log. Commit, by explicit path only, the author's docs: `docs/H2S-NN_methodology_spec.md`, `docs/spec_v1.15_draft_edits.md`, `docs/spec_backups/H2S-NN_methodology_spec_v1.14.md`, `docs/experiment_settings.html`, `docs/experiment_settings_decisions.md`, `docs/H2S-NN_partM_card.md`, `docs/handoff_2026-11-12.md` (message "docs: spec v1.15, experiment settings round 2, Part M card"). Create `part-m` from that commit. Run the full suite once and record the pre-state (known: `test_baselines` failures under Windows Application Control).

### M1 — Data layout (`data/NC`, `data/LP`, `data/GC`)
The author moved the data: `data/NC/{acm,dblp,imdb,freebase,mag/{ogbn-mag,mag-year},OAG/raw,...}`, `data/LP/{LastFM,PubMed,lp_derived}`, `data/GC/{MUTAG,PROTEINS,ENZYMES,NCI1}`. Update every loader, preset, script default and `data/PROVENANCE.md` to the new layout (one place for the roots; no compatibility shims for the old paths).
Scope: the local datasets only (HGB NC 4, LP 6 local tasks, TU 4); H2GB folders (`NC/mag`, `NC/OAG`, ...) are not touched.
**Gate:** each local dataset loads, and its node/edge/label counts equal the counts recorded before the move (PROVENANCE.md, J-report, `docs/WISP_evidence/h0_data_facts.json`); a difference stops the unit.

### M2 — Heterophily harness, local datasets (H-1, H-2) — DEFERRED (later card)
Script `scripts/heterophily_stats.py` + module under `src/analysis/`. Call the reference code **unchanged** (import or vendored call; no re-implementation of the formulas):
- MLH and MDE: `vendor/Hetero2Net/hetero2net/homophily.py` on Hetero²Net's `M₂` (its predefined list where it has one: DBLP APA; ACM PAP, PSP, PTP; IMDB MDM, MAM, MKM; ogbn-mag PFP, PAP — else `get_metapaths`, `homophily.py:51-63`).
- ℋ²: `vendor/H2GB/H2GB/calcHomophily.py` with **its own** meta-path rule (`extract_metapath`).
- Conventions of spec §1: symmetrised graph as the loaders build it; IMDB per genre (binary) then mean; unlabeled target nodes get H2GB's random fill (`calcHomophily.py:110-113`), fill seed recorded, spread over 5 fill seeds, labeled-only value beside it; MDE on raw features, "–" without target features.
- LP tasks: label measures on the task's graph where both endpoint types carry labels, task-level `H_edge` (Freebase book–book without self-loops); MDE on the F17 files per endpoint type that keeps features; PubMed/LastFM label measures "–".
- GC: 1-hop `H_edge` on original bonds, label = atom type (Hetero²Net's label-homophily formula without a meta-path); MDE "–".
- **No published value is used** (author): no comparison column, no tolerance. Each value records code path and commit.
**Tests:** wrapper tests on hand-built graphs with values computed by hand (e.g. 4 target nodes, 2 classes, one meta-path: MLH, `H_adj`, MDE known exactly); the meta-path sets produced for ACM/DBLP/IMDB equal Hetero²Net's lists; H2GB's rule includes a same-type relation where one exists. **Output:** `docs/heterophily_evidence/{stats.json, table.md}` for every local dataset of NC (HGB 4), LP (local tasks), GC (TU 4).

### M3 — H2GB node-classification datasets (8) — DEFERRED (later card, on the lab cluster)
ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem, IEEE-CIS-G, H-Pokec, PDNS from `data/NC/`, through H2GB's loaders (or equal-output wrappers).
- **OAG identity check:** H2GB's `oag_dataset.py:55-57` maps `engineering` → `OAG_Chemistry.pt` and `chemistry` → `OAG_Engineering.pt`. Count nodes and edges per type of each OAG file and compare with H2GB Table 1 (arXiv 2407.10916; record what was read). A mismatch is reported with the evidence and the unit continues with the files labelled by their verified identity.
- **Venue leak audit (F18 style):** OAG labels come from `PV_Journal` paper–venue edges; confirm venue nodes and every paper–venue relation are absent from the message graph; fingerprint probe as F18.
- **PDNS:** H2GB's temporal protocol (training graph over training windows; one test graph per later window, previously labelled domains removed). Implement the loader; the trainer path for an inductive test graph is listed as a decision for the author if it needs more than evaluating the same model on a second graph.
- Splits as H2GB; record split sizes and class counts.
- Heterophily of these graphs (H-3): run M2's script **only if** it fits the laptop (state peak memory); otherwise write the exact server command in the report.

### M4 — WISP on every task (J1)
Replace `build_free_path_groups(length=2)` in `link_prediction_dataset.py:914` (and the LP trainer's resampling) and `graph_classification_trainer.py:203` with the WISP sampler of spec §3.2. Anchors: LP = every node of the two endpoint types on the message graph; GC = every node of the graph. Evaluation draw fixed (seed of the run); training draws resampled per epoch. `ℓ_max` and `r` read from config (both tunable now; no calibrated-`r` enforcement for LP/GC).
**Tests:** anchor sets exact on a toy graph; no held-out fact (either orientation) in any LP cell (`assert_no_held_out_facts`); evaluation draw identical across epochs; GC cells of a graph independent of other graphs; mutation: anchors restricted to one endpoint type must fail a test. **Smoke (v3-B anchored, `arm: full`):** `acm_pa` LP and MUTAG GC, 3 epochs, CPU or GPU: finite loss, parameters move, PCG iterations ≤ `T_max`; record time and peak memory.

### M5 — GC protocol and readout (J6, spec §6 GC)
Readout: `[x*, h₀]` → pooling ∈ {mean, max, add, attention} (attention = HetSheaf's `AttentionPooling`, Linear → Tanh → Linear, 2 heads averaged, `hetsheaf/graph_classification/pooling.py`) → **one Linear**; remove BatchNorm and the MLP from `graph_readout.py`. Protocol: TU via PyG `TUDataset` (`use_node_attr=False`), node types from one-hot label argmax, MUTAG bond types, constant feature 1, one random 80/10/10 split with seed 42 for every training seed, batch size 1, early stopping on validation loss with patience 50, test with final weights, validation Macro-F1 selects configurations. Epochs: leave configurable (open O4).
**Tests:** pooling outputs on a hand graph; split identical across training seeds and equal to HetSheaf's `randperm(seed 42)` order; mutation: re-inserting BatchNorm fails a test.

### M6 — Stop conditions (J10)
`τ > 50` and log-weight range of `g` > 30 raise a named exception (as `T_max` does), logged with epoch and value. **Tests:** forced values trigger each; normal smoke does not.

### M7 — Link-prediction protocol (J5)
Test negatives: port HGB's `get_test_neigh` exactly (2-hop from symmetric train + valid + test adjacency, direct links and self-loops removed, type-filtered, `random.choices` with replacement, count = head's test positives, seed 1; no fallback). Validation: `get_valid_neg` (one random tail of the correct type per positive). Remove the random fallback (`link_prediction_dataset.py:720, 751`). Training: batches of 8,192 positive links, 40 epochs, patience 40. Write the negative files once per task; every method reads them (bundles). Scope: the 6 local tasks (acm_pa, dblp_ap, imdb_ma, freebase_bb, PubMed, LastFM); ogbn-mag P–P and H-Pokec U–U wait for the H2GB data. Report per task: heads without 2-hop candidates (count, share of test positives). MRR: test on a hand example against HGB's description. Rerun the F18 leak audit per task after the change.
**Tests:** a toy graph where the 2-hop candidate set is known by hand; seed-1 determinism; no candidate is a known positive or self-loop; mutation: re-enabling the fallback fails a test.

### M8 — Metrics (spec §6, 1.14/1.15)
Macro-F1 for every NC dataset and method: classes fixed per split from that split's true labels, passed as `labels=L`; IMDB: a genre counts when at least one movie of the split carries it. Shared metrics module used by H2S-NN and every adapter.
**Tests:** a split missing a class (prediction of that class does not enter the average); IMDB genre absent from a split.

### M9 — Shared training protocol (J4)
Every method: AdamW (replace `torch.optim.Adam` in adapters; list every changed file), no LR scheduler, no gradient clipping (H2S-NN's `optim.scheduler`/`grad_clip` defaults become none). `budget.py`: budgets per task and source protocol — HGB NC max 300, patience 30; LP 40, patience 40; H2GB NC 500, no early stopping (select the best validation epoch); GC as M5. NC/LP: early stopping and checkpoint on the validation metric (NC Macro-F1, LP ROC-AUC). H2S-NN's scalar group (θ_τ, `b`: 10× lr, no weight decay) unchanged.
**Tests:** every adapter's optimizer is AdamW (runtime check over the registry); budget table; no scheduler object is created.

### M10 — Search spaces (J2, J3)
`scripts/run_h2snn_wandb_tuning.py`: plain Bayes, **no Hyperband**; H2S-NN space exactly: `d` {16, 32, 64}; `lr` log-uniform [1e-4, 1e-2]; `weight_decay` {0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 0.1, 0.8}; `encoder_dropout`, `classifier_dropout` uniform [0, 0.9] (encoder dropout absent for featureless datasets); `wisp_max_length` {2, 3, 4, 5}; `wisp_walks_per_anchor` {8, 16, 32, 64}; GC pooling {mean, max, add, attention}; LP `decoder_type` unchanged (open). `NEVER_TUNED_*`: allow `wisp_max_length`, `wisp_walks_per_anchor`; refuse `K`, `layer_dropout` and every legacy key of `experiment_settings.html` §4.2 (second table). Tuning = seed 0 only, then the best configuration on seeds 0–4 (script support; do not run it). Shared axes in `search_grids.py` (`LR`, `WEIGHT_DECAY`, `DROPOUT`) applied to every adapter's shared axes; baseline-specific axes unchanged; `feats_type` unchanged (open).
**Tests:** the generated sweep config equals the table above; `SearchSpaceCoverageTests` (published values reachable) still pass; refusing `K` raises.

### M11 — Baseline registry (J7)
Remove LatGRL (adapter, registry entry, notebooks via the generator, tests that name it). `registry.py`: `DATASETS` = NC 12 (HGB 4 + H2GB 8), LP 8 (acm_pa, dblp_ap, imdb_ma, freebase_bb, PubMed, LastFM, ogbn-mag P–P, H-Pokec U–U), GC 4; remove YouTube/Amazon; H2GB datasets and the two large LP tasks are registered with the unsupported reason "data pending (lab cluster, after mini-batch design)"; HetSheaf moves from `CITED_ONLY` to "adapter pending" (the adapter itself is the next card). Unsupported cells keep a reason.
**Tests:** registry consistency; no cell names LatGRL.

### M12 — Full suite and report
Full suite (record known Application Control failures separately). Report: per unit what was done, VERIFIED-RUN output, deviations from this card with reasons, smoke cost table (time, memory, PCG), heads without LP negatives, "Decisions for the author", out-of-scope findings.

## Stop conditions
- A gate fails (M1 counts, a leak audit, an OAG file whose identity cannot be established): stop that unit, report, continue with the next independent unit.
- A change would alter the method (energy, maps, solver, WISP target) or needs an open decision: stop the unit, report.
- `T_max`, `τ > 50` or the log-weight stop fires in a smoke run: report with the values; not a reason to change settings.
**Not stop conditions:** any heterophily value (no published comparison); smoke-run losses or metrics (not performance); a baseline number (untuned vs tuned; never a design signal).
