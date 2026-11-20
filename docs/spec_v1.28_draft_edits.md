# Spec 1.28 — draft edits (PC-1 – PC-11), APPROVED and APPLIED 2026-11-20

Drafted 2026-11-20. Base: spec v1.27 (origin/design-docs `4e8f935`). **Source:** decisions log rows of 2026-11-20 from
"Published tuned configurations; LP and GC model sets" to "Card 8 amendment 4 (second C1 stop)", and card 8 with
amendments 1–4 (`docs/H2S-NN_card8_published_rgcn_magnn_card.md`; cell inventory `docs/card8_evidence/c0_inventory.md`
on `card8` `11cb744`). **The method is unchanged.** 1.28 changes §5 C7's evidence column and §6 (tuning, evaluation
protocol, baselines, LP metrics and training, statistics, published numbers).

Two points need an author decision before this draft can be applied: **OPEN-1** (statistics with published numbers,
PC-9) and **OPEN-2** (HetSheaf's C7 rerun against its filled cells, PC-11).

---

## PC-1 Header
- Replace `**Version:** 1.27 (2026-11-19).` with `**Version:** 1.28 (2026-11-20).`
- Insert before `**Changes from 1.26 (to 1.27;`:
> **Changes from 1.27 (to 1.28; author decisions of 2026-11-20 "Published tuned configurations; LP and GC model sets", "Card 8: R-GCN and MAGNN", "Card 8 amended: no MAGNN in GC", "Card 8 amendment 1", "Baseline DistMult as HGB", "LP test on random negatives only", "Card 8 amendment 4"; draft `docs/spec_v1.28_draft_edits.md` PC-1–PC-11; the method is unchanged):** (PC1) A published configuration (HGB, H2GB, the method's own repository) is the tuned configuration of its cell: 5 seeds, run exactly as the file states; tuning only where none exists. (PC2) Evaluation protocol option 1: data, splits, test negatives and model code follow HGB/H2GB; metrics, selection, AdamW for tuned cells and seeds are shared; an appendix table lists every difference. (PC3) Baselines: NC 15 (R-GCN, MAGNN added), LP 12, GC 6 = HetSheaf's Table 4 rows as published numbers (no baseline GC run; no MAGNN in GC). (PC4) LP reported on the random test set only. (PC5) Baseline small-LP training in batches of 8,192 positives; baseline DistMult a full `d × d` matrix per relation. (PC6) Statistics: [OPEN-1]; LP family 16 tests. (PC7) The 76 cells with published numbers keep them (author decision under AA-10).

## PC-2 §6 "Every number is new (1.5)"
Replace the bullet with:
> - **Every number is new (1.5; qualified 1.26, 1.28):** every cell we run (H2S-NN and every baseline) is run under this protocol; no number of the submitted result tables is reused, quoted or audited. Exception (1.28, author 2026-11-20): a baseline cell that already holds a published number keeps it (76 cells: NC 52, GC 24; `configs/baselines/paper_filled_cells.tsv`, register rows), as an author decision under the placeholder rule (§6 "Published numbers").

## PC-3 §6 "Tuning" bullet — published configurations first
Insert at the start of the bullet, before `plain Bayesian search`:
> **Which cells run, and how (1.28, author 2026-11-20).** A cell of the paper tables that already holds a published number is not run (PC-2). An empty cell with a **published configuration** (a file that gives the values for that dataset: HGB's run scripts and README commands, H2GB's `configs/<dataset>/<dataset>-<Method>.yaml`, or the method's own repository) runs **5 seeds (0–4) at that configuration, exactly as the file states**: model, every hyperparameter and every training-schedule key (optimizer, learning-rate scheduler and warm-up, gradient clipping, accumulation, batch size, epochs, evaluation period, sampler fan-out); no search. Argparse defaults alone are not a configuration unless the source's README runs that script for that dataset without flags. A source key our runner cannot execute exactly makes the cell tuned, never approximated (R-GCN × LastFM LP: HGB's command gives width 64 with 10 `bdd` bases, which DGL 2.4 refuses). Only empty cells without a published configuration are tuned as below. Cell counts (card 8, after amendments 1, 3 and 4): NC 52 filled, 33 published, 86 tuned, 8 probe, 1 N/A; LP 0 filled, 5 published, 88 tuned, 2 probe, 1 N/A; GC 24 filled, no baseline run.

In the same bullet:
- Replace `the same N for H2S-NN, every baseline and HetSheaf` with `the same N for H2S-NN and every tuned baseline cell, HetSheaf included`.
- Replace `Search spaces are published per method (each baseline's space contains its published values).` with `Search spaces are published per method; a tuned cell's space contains that method's published values where they lie inside the shared axes (a published cell is not tuned, so its values need not lie in the grid).`
- Replace `**Shared by every method (1.15, author):** optimizer AdamW (baselines whose original code uses Adam switch, stated in the paper);` with `**Shared by every tuned cell (1.15, author; 1.28: published cells run their source's optimizer, scheduler, clipping and accumulation instead):** optimizer AdamW (tuned baselines whose original code uses Adam switch, stated in the paper);`
- Replace `batch size: HGB NC full graph, LP 8,192 positive links, GC 1 (not tuned);` with `batch size: HGB NC full graph, LP 8,192 positive links per optimizer step for every method (H2S-NN and baselines; 1.28 PC-5), GC 1 (not tuned);`.

## PC-4 §6 new bullet after "Tuning" — evaluation protocol (option 1)
> - **Evaluation protocol (1.28, author 2026-11-20, option 1).** Data files, splits, test negatives and each model's code follow HGB and H2GB. Shared by every method we run: the metrics (Macro- and Micro-F1 with the per-split class list; LP ROC-AUC and MRR on the random test set), checkpoint and configuration selection (B1a validation subset on the H2GB NC datasets; the fixed validation split and validation negatives drawn once on HGB; LP selection on validation ROC-AUC; validation once per epoch, or every `eval_period` epochs on H2GB), AdamW for tuned cells, seeds 0–4 on the H100. A published cell keeps its source's model and training schedule (PC-3) and this evaluation. An **appendix table** lists every point where a run differs from HGB's or H2GB's own runner, with the reason (e.g. HGB's GCN LP code validates every 10 batches with early stopping on validation loss, `homoGNN.py:46-65`, ours once per epoch on ROC-AUC; H2GB evaluates `val.iter_per_epoch × 32` sampled nodes, ours the whole split and the B1a subset; on the binary H2GB datasets H2GB's networks give one logit and BCE, ours `C` logits and cross-entropy). Published numbers in the tables were produced under the benchmarks' own protocols; the paper states this beside the tables.

## PC-5 §6 "Link prediction" — Metrics and Training
In *Metrics (1.14 …)*:
- Replace `**every task is evaluated on two test sets**, both reported. (i) **2-hop** (` … up to and including `(never as MRR = 1). (ii) **Random** (` with `**every task is reported on the random test set** (1.28, author 2026-11-20; the 2-hop set of 1.16 is dropped: it is empty on dblp_ap and imdb_ma by schema, and on acm_pa 23.3 % of the test positives have no 2-hop negative and would get reciprocal rank 1; the code may still compute it, no table reads it): **random** (`.
- Keep the rest of the random-set definition, the validation and training negatives unchanged.

At the end of the *Metrics* sub-bullet, replace `Training: batches of 8,192 positive links, 40 epochs, patience 40 (HGB's LP defaults); for H2S-NN on the six small tasks one step is one solve on the whole message graph;` with:
> Training (1.28 PC-5, every method): an epoch takes `ceil(P / 8,192)` optimizer steps, each on 8,192 shuffled training positives and its share of the epoch's training negatives, one encode per step (HGB's runner and H2S-NN; the baseline loop took one step per epoch on all positives before card 8 C2b); a published LP cell uses its source's batch (PSHGCN LastFM 10,000 positives, `hgb-lp/training.py:179`; R-GCN PubMed 1,024; GCN 512 positives + 512 negatives, `GNN/utils.py:166`; MAGNN LastFM 8); 40 epochs, patience 40 (HGB's LP defaults) for tuned cells, the source's epochs and patience for published cells; for H2S-NN on the six small tasks one step is one solve on the whole message graph;

Add to *Model* sub-bullet:
> **Decoder (1.28, author, card 8 amendment 2).** Every baseline scores pairs with the shared head: `dot`, or DistMult `uᵀ W_r v` with a **full `d × d` matrix per relation**, initialised as HGB's (`LP/benchmark/methods/baseline/GNN.py:10-20`, xavier normal, gain 1.414); the diagonal form used before is a strict special case (symmetric in u, v) and would favour H2S-NN by decoder capacity on same-type targets (PubMed, freebase_bb, ogbn-mag P–P, H-Pokec U–U). H2S-NN's decoder is unchanged; its endpoint projection `Linear(2d, d) + GELU` before the decoder is listed in the appendix differences.

## PC-6 §6 "Graph classification" — baseline rows
Append to the bullet:
> **Baselines (1.28, author option (a)):** the GC baseline rows are HetSheaf's Table 4 (GCN, GAT, NSD, R-GCN, HAN, HetSheaf; Macro- and Micro-F1, mean ± std over seeds 42–46), whose protocol this bullet copies in every point (differences: their 150-run sweep vs our 100 trials; seeds 42–46 vs 0–4). No baseline GC cell is run; H2S-NN's four GC cells are run. ChebNetII, CSNN, SeHGNN, HINormer, PSHGCN, H2G-former, Hetero²Net and H2SGNN are not in the GC table. MAGNN is not in GC: the TU graphs define no meta-paths (node types = atom types, one edge type except MUTAG), and HetSheaf's GC HAN row uses PyG `HANConv` on edge types, i.e. length-1 paths (`hetsheaf/models/gnn_baselines/HAN.py:11,53`).

## PC-7 §6 "Baselines (1.14)" bullet — model sets and implementations
- Replace the opening `all rerun under this protocol, including HetSheaf (an adapter for node and link tasks is required; `src/baselines/registry.py:107-110` still lists it as cited-only); no quoted numbers in result tables.` with `every empty cell is run under this protocol (published configuration or tuning, PC-3); cells that hold a published number keep it (PC-2).`
- Replace `(1.15) Baselines: GCN, GAT, ChebNetII, NSD, CSNN, HAN, SeHGNN, HINormer, PSHGCN, H2G-former, Hetero²Net, H2SGNN, HetSheaf.` with:
> (1.28) **Model sets per task.** NC (15): GCN, GAT, ChebNetII, NSD, CSNN, R-GCN, HAN, MAGNN, SeHGNN, HINormer, PSHGCN, H2G-former, Hetero²Net, H2SGNN, HetSheaf. LP (12): GCN, GAT, NSD, CSNN, R-GCN, HAN, MAGNN, PSHGCN, H2G-former, Hetero²Net, H2SGNN, HetSheaf. GC (6, published): GCN, GAT, NSD, R-GCN, HAN, HetSheaf (PC-6). **R-GCN:** HGB's code (`NC/benchmark/methods/RGCN`, `LP/benchmark/methods/RGCN`) on the HGB datasets and LP tasks, its decoder replaced by the shared head; on H2GB, H2GB's own R-GCN (`HomoGNNModel`, `layer_type: RGCN`) at its yaml. **MAGNN:** HGB's code (`NC/benchmark/methods/MAGNN`, `LP/benchmark/methods/MAGNN`) with HGB's meta-path lists (`docs/predefined_metapaths.md`; LastFM: MAGNN's own); an LP endpoint type without a listed path uses the symmetric length-2 schemas `t–X–t` over its neighbour types on the message graph (held-out facts removed); its H2GB NC cells and the two large LP tasks are probed on the H100 at the smallest grid point and are N/A if it does not fit (range rule 1). **GCN and GAT rows use two implementations** (stated in the paper): HGB's models on the tuned cells (decision S1) and H2GB's `HomoGNNModel` on the published H2GB cells (GCN on mag-year, oag-cs, oag-eng, oag-chem, H-Pokec, PDNS; GAT on the same minus H-Pokec, whose yaml says `layer_type: GCN`).
- In `(1.23, author) Baselines on the H2GB datasets …`, after `class A (GCN, GAT, HAN, H2G-former) on H2GB's `hgt` sampler with its YAML fan-out lists;` insert ` (1.28) a class-A cell with a published yaml runs H2GB's own network and schedule at that yaml (PC-3);`.

## PC-8 §6 "Published numbers as placeholders"
Replace the last two sentences (`A placeholder enters no statistical test, and no claim rests on it. Before submission, … for placeholders only.`) with:
> (1.28, author 2026-11-20) The 76 cells that hold a published number keep it by author decision (NC 52, GC 24; `configs/baselines/paper_filled_cells.tsv`, with the register line of each); the register records the decision per cell. The 5 HGB LP numbers (2-hop) left the tables (PC-5) and their cells are run. Statistics with published numbers: §6 "Seeds and statistics" [OPEN-1]. No claim rests on a published number alone.

## PC-9 §6 "Seeds and statistics" — [OPEN-1]
Replace `Committed now: every raw p-value and the number of tests are reported (≈ 60: NC 24, LP 28 = 8 tasks × 2 metrics × 2 test sets − 4 undefined 2-hop cells, GC 8; 1.16), and both raw and adjusted values are given; the adjustment is Holm, fixed before the results (1.23, author), applied per task: NC (24 tests), LP (28) and GC (8) are separate families.` with the text of the option the author chooses.

**The problem.** AA-10 says a placeholder enters no test, but 76 cells now hold published numbers: on NC the best other method is often a published row (e.g. ACM: 13 of the 15 baselines are filled), and on GC every baseline is published, so "H2S-NN against the other method with the best mean" cannot be computed from raw seed values.

| option | rule | tests | for | against |
|---|---|---|---|---|
| A (recommended) | NC and LP: H2S-NN against the best method **run under this protocol**; the best published number is shown and, if it beats H2S-NN, said so without a p-value; the number of (dataset, metric) pairs where a published number is the best is reported. GC: Welch from summary statistics (`scipy.stats.ttest_ind_from_stats`, `equal_var=False`, n = 5 each) against the best HetSheaf Table 4 row, because that protocol equals ours (PC-6); HetSheaf's std is converted as if reported with ddof = 0 (× √(5/4)), the direction that can only widen the variance | NC 24 + LP 16 + GC 8 = **48** | every test compares numbers produced under one protocol (NC/LP) or an identical one (GC); no raw seeds are needed | NC/LP: the test is not against the overall best row when that row is published |
| B | Welch from summary statistics wherever the source reports mean ± std over 5 runs (HGB Table 3, H2GB Table 3, HetSheaf Tables 2 and 4, HINormer, Hetero²Net, SeHGNN), against the overall best | about 48 (fewer where the source's ± is unclear: PSHGCN and H2SGNN label it standard error; SeHGNN averages 5 data partitions) | always against the best row | mixes protocols inside a test (splits, selection, 150-run sweeps); the ± definition is uncertain for several sources |
| C | no test where the best other method is a published number; report the count | NC ≤ 24 (count unknown until results exist) + LP 16 + GC 0 | strictly AA-10 | the GC table has no test at all; the NC count depends on the results |

Recommendation: **A**. It keeps AA-10's intent (no test mixes protocols) except on GC, where the protocols are equal by construction, and it fixes the test count before any result (48; Holm per task: NC 24, LP 16, GC 8).

Text for option A (to replace the sentence above):
> Committed now (1.28): every raw p-value and the number of tests are reported (48: NC 24 = 12 datasets × 2 metrics, LP 16 = 8 tasks × 2 metrics on the random test set, GC 8 = 4 datasets × 2 metrics), with raw and Holm-adjusted values; Holm is applied per task (families NC 24, LP 16, GC 8), fixed before the results (1.23, author). On NC and LP the comparison is H2S-NN against the method with the best mean test value **among the methods run under this protocol**; the best published number is shown beside it and, where it is higher than H2S-NN's, the paper says so without a p-value; the number of (dataset, metric) pairs whose best value is published is reported. On GC every baseline is published (HetSheaf's Table 4, same protocol, PC-6): the test is Welch's test from summary statistics (`scipy.stats.ttest_ind_from_stats(equal_var=False)`, n = 5 per group) against the best Table 4 row, with HetSheaf's std multiplied by √(5/4) (treated as ddof = 0, the conservative reading).

## PC-10 §5 C7 evidence
Replace `| C7 Competitive | Rerun baselines under the identical protocol; HetSheaf rerun mandatory | — | — |` with `| C7 Competitive (1.28) | Baselines under the identical evaluation protocol (PC-4): every empty cell run at its published configuration or tuned; cells with a published number keep it (PC-2); statistics as §6 "Seeds and statistics" | — | — |` — subject to OPEN-2.

## PC-11 §6 "G0 outcome" and HetSheaf — [OPEN-2]
**Conflict.** The G0 bullet and C7 say "HetSheaf is **rerun** on our pipeline (C7), its published numbers appear only in a labelled non-comparable note". The register now fills HetSheaf × ACM, DBLP, IMDB NC (Table 2) and its GC rows (Table 4) with published numbers, so on those cells HetSheaf is not rerun. Its ACM and DBLP NC cells are tuned anyway in phase `baselines-early` (card 7 needs their configurations).

| option | HetSheaf NC ACM, DBLP, IMDB | cost |
|---|---|---|
| A (recommended) | ACM and DBLP: our run replaces the placeholder (seeds 0–4 at the `baselines-early` configuration); IMDB: run as well (one more tuning cell), so HetSheaf's NC row is ours on the HGB datasets and C7's "rerun" stays true there; GC stays published (PC-6) | 3 final-seed lines + 1 tuning cell |
| B | keep the three published numbers; the G0 bullet and C7 drop "HetSheaf rerun mandatory" | none |

Text for option A: in the G0 bullet replace `its published numbers appear only in a labelled non-comparable note` with `on NC its published numbers are replaced by our runs (1.28); on GC its Table 4 rows are the published baselines (PC-6)`; remove HetSheaf × ACM, DBLP, IMDB from `paper_filled_cells.tsv` (ops).

---

**Applied 2026-11-20 by the model session** (author: OPEN-1 = A, OPEN-2 = A; decisions log row "Statistics with published numbers (OPEN-1); HetSheaf rerun (OPEN-2)", card 8 amendment 5). Backup: `docs/spec_backups/H2S-NN_methodology_spec_v1.27.md`. Changes against the text above, all from amendment 5 and the two choices:
- Filled cells 73 (NC 49, GC 24), not 76 (NC 52): PC-1, PC-2, PC-8. PC-3 counts: NC tuned 87 (plus HetSheaf × ACM, DBLP in `baselines-early`).
- PC-1 (PC6, PC7), PC-9 and PC-11 use the option A texts. PC-9 also qualifies the earlier sentence of the same bullet ("H2S-NN against the other method with the best mean", now "among the methods run under this protocol on NC and LP; GC below").
- PC-7 opening also states HetSheaf's NC rerun on ACM, DBLP, IMDB; PC-8 lists HetSheaf's three NC numbers among those that left the tables; PC-10 keeps "HetSheaf's NC cells … rerun (mandatory)".
- Cross-references to "PC-n" inside the spec text are written as §6 bullet names.
