# Spec v1.14 — draft edits (experiment settings), for the author's approval

**Status:** APPLIED 2026-11-11 as spec v1.14 (author answered the open points: ℋ² uses the same meta-path set; Macro-F1 averages the classes present in each split's true labels, recommended and adopted; IEEE-CIS-G's F1 in H2GB is Macro-F1, author). Backup of v1.13: `docs/spec_backups/H2S-NN_methodology_spec_v1.13.md`. The text below is the draft as written before those answers. The spec on disk is **v1.13** (another Cowork session added G17, the incidence poset, after v1.12; md5 `6395c8364db6fc2fb167e263357b57c0`, modified 2026-11-11 08:24 JST). These edits are written against that file and do not touch G17 or any 1.13 text. When approved, the file is re-read immediately before writing; if it changed again, the edits are merged onto the newer version.
**Source of the decisions:** `docs/experiment_settings.html` (updated after the author's answers to D1–D8, 2026-11-11) and this session's chat of 2026-11-11 (ogbn-mag + mag-year kept, GC heterophily as a limitation, compute not a constraint, feature preprocessing per method).
**Scope:** experiment settings only (datasets, heterophily statistics, protocol, baselines, tuning, seeds, statistics, metrics). Discussion, model analysis and ablation (claims C2–C14, known-answer rows) are out of scope and are only marked "to be restated".
**Still open (not written as decisions; marked OPEN in the new text):** ℋ²'s meta-path set; Macro-F1 averaging rule for ogbn-mag (349 classes); the meaning of H2GB's F1 on IEEE-CIS-G; epoch/patience budgets of HGB, H2GB and GC (decided at training time); multiple-comparison adjustment (decided after results, raw and adjusted both reported); mini-batch design; HSB-GDL (waiting for its author); LP validation metric ROC-AUC (taken as accepted; author may object).

---

## X-1. §1 Tasks (line 44)
**Old:**
> **Tasks.** Main: node classification of a target type. Secondary: link prediction on heterogeneous graphs (§6). Appendix: graph classification, reported only as evidence that the model covers the task (generalizability); it carries no heterophily claim, and its "types" are atom and bond labels, stated as such (1.5).

**New:**
> **Tasks.** Main: node classification of a target type. Secondary: link prediction on heterogeneous graphs (§6). Appendix: graph classification, reported only as evidence that the model covers the task (generalizability); it carries no heterophily claim, and its "types" are atom and bond labels ("typed-by-label"), stated as such (1.5). No truly heterogeneous graph-classification benchmark is available; this is stated as a limitation (1.14).
> **Dataset rule (1.14).** A dataset is used only if it has node attributes or target-node class labels (or both). LastFM (link prediction) has neither and is kept as the one stated exception; its heterophily statistics are "–".

## X-2. §1 Where reported (line 54)
**Old (fragment):**
> Without node labels (LastFM; graph classification) the paper states "not measurable" and why.

**New (fragment):**
> Without node labels the label-based measures are "not measurable" (LastFM, PubMed), with the reason. Graph classification reports the 1-hop `H_edge` on the original bonds with label = atom type (the label-homophily formula of Hetero²Net without a meta-path), marked "1-hop, label = atom type, not the graph class"; MLH over meta-paths would be 1 by construction there (1.14).

## X-3. §1 Meta-path set (line 57)
**Old:**
> - **Meta-path set and graph (1.8).** `M₂` is H2GB's full set of length-2 target→target meta-paths for every dataset, on the symmetrised graph as H2GB's loaders build it (Freebase stores no reverse relations, so book→X→book needs it). Hetero²Net's per-dataset sets serve only the reproduction check. Self-pairs follow the reference code (they are what reproduces DBLP 0.87; 0.80 without), stated in the paper.

**New:**
> - **Meta-path set and graph (1.14, replaces 1.8).** `M₂` is Hetero²Net's predefined list where it has one (DBLP: APA; ACM: PAP, PSP, PTP; IMDB: MDM, MAM, MKM; ogbn-mag: PFP, PAP; Hetero²Net Appendix B.1). Otherwise `M₂` is the output of Hetero²Net's `get_metapaths` (`vendor/Hetero2Net/hetero2net/homophily.py:51-63`): length-2 meta-paths that start and end at the target type and skip relations whose two endpoint types are equal (so e.g. paper–cite–paper–cite–paper is not counted). mag-year (same graph as ogbn-mag, not in Hetero²Net) therefore gets its own set by this rule; different values on the same graph are accepted (author, 2026-11-11). The graph is symmetrised as the loaders build it (Freebase needs it). The only check is the reproduction of Hetero²Net Table 1 (DBLP/ACM/IMDB: MLH 0.87/0.60/0.24, MDE 0.80/0.74/0.59) within a tolerance fixed before running. Values published by H2GB (ℋ², MLH on its own rule) are cited as context, not as a reproduction target. **OPEN:** whether ℋ² (H2GB's code) is fed the same `M₂`. Self-pairs follow the reference code.

## X-4. §1 new bullet after line 57: MDE
**New:**
> - **Attribute heterophily: MDE (1.14).** Metapath-based Dirichlet Energy of Hetero²Net (Eq. 3–4; code `homophily.py:4-14`): `MDE = mean_{P∈M₂} ¼ mean_{(u,v)∈G_P} ‖x_u/√|N_u| − x_v/√|N_v|‖²`, on raw features of the target type, same `M₂` as MLH. Reported for every dataset whose target type has features; "–" otherwise (Freebase, LastFM, graph classification, whose features are constant). Link prediction: computed on the task's files after the F17 feature rule (author decision D1), per endpoint type that keeps features, on the training graph; a type whose features are removed is "–". MDE depends on the feature scale and is a descriptor of each dataset, not a cross-dataset heterophily level; ℋ² remains the stratification variable of C6.

## X-5. §6 Node classification (line 207)
**Old (fragment):**
> H2GB IEEE-CIS-G, mag-year, ogbn-mag, H-Pokec with their loaders, splits and metrics.

**New (fragment):**
> H2GB IEEE-CIS-G, mag-year, ogbn-mag, H-Pokec with their loaders and splits; metrics are Macro-F1 and Micro-F1 for every node-classification dataset (1.14; replaces H2GB's Accuracy/F1; on single-label data Micro-F1 equals accuracy, which keeps H2GB's published accuracy usable as a sanity reference). **OPEN:** the Macro-F1 averaging rule for ogbn-mag (classes present in test, or all 349), the same for every method, fixed before running; H2GB's F1 definition on IEEE-CIS-G. No further H2GB dataset is added (author, D4). The mini-batch design is reopened (author, 2026-11-11): the M0–M4 design and its v3-B addendum are reference material until the author confirms them.

## X-6. §6 LP datasets (line 210)
**Old (fragment):**
> ogbn-mag paper–paper citation and H-Pokec user–user friendship (1.10 F21: same protocol, split by unordered pair, HeaRT on a pre-registered positive subsample for cost; built on TSUBAME after the mini-batch card).

**New (fragment):**
> ogbn-mag paper–paper citation and H-Pokec user–user friendship (1.10 F21; 1.14: the HGB protocol below, split by unordered pair; built on TSUBAME after the mini-batch design).

## X-7. §6 LP split (line 211)
**Old (fragment):**
> 10 training seeds per task.

**New (fragment):**
> 5 training seeds per task (seeds 0–4; 1.14).

## X-8. §6 LP metrics (line 212): HGB protocol for every task
**Old:** the whole bullet `- *Metrics.* Primary: MRR and Hits@K against hard negatives per positive (HeaRT, … the random-fill share of HeaRT's lists is reported per task.` (line 212, copied verbatim when applied).
**New:**
> - *Metrics (1.14; replaces 1.5–1.10 F8, F14, F16 and HeaRT everywhere).* The HGB protocol for **every** link-prediction task, including the tasks derived from node-classification graphs and the two large tasks: test negatives are 2-hop neighbours of the head, 1:1 with the test positives, drawn within the target relation's tail type; metrics ROC-AUC and MRR (MRR grouped by head node as HGB describes; `src/utils/misc.py:178-236`). Validation: random tail negatives, model selection on validation ROC-AUC. Every method sees the same negative files. HeaRT is not used (primary or secondary); its files stay on disk unused. Reason (author): one protocol across datasets avoids a bias question from reviewers. To check before running: the share of 2-hop draws that fall back to random on cross-type tasks; agreement of our 2-hop sampler and MRR with HGB's own code (not available locally, `vendor/HGB` is empty); cost of 2-hop sampling on the two large graphs. HGB's `link.dat.test` of PubMed and LastFM lists positives only (8,528 and 18,567 rows, all label 1.0), so their negatives are generated by the same rule.

## X-9. §6 Graph classification (line 216)
**Old:**
> - **Graph classification (appendix; generalizability only):** TU datasets with atom/bond labels as types, stratified 10-fold CV; no heterophily claim; no design work until the author asks (1.5).

**New:**
> - **Graph classification (appendix; generalizability only; 1.14: HetSheaf's protocol in every point, author D7):** MUTAG, PROTEINS, ENZYMES, NCI1 (PyG `TUDataset`, `use_node_attr` False); node types = argmax of the one-hot node-label columns, edge types = MUTAG's 4 bond types, one type elsewhere; node features replaced by the constant 1; one random split 80/10/10 with seed 42 for every training seed (not stratified, not cross-validation; test sets 20 / 112 / 60 / 411 graphs); batch size 1; early stopping on validation loss with patience 50; test with the final weights; validation Macro-F1 selects configurations; test Macro-F1 and Micro-F1. Seeds 0–4 (HetSheaf: 42–46; the split stays seed 42). Readout: the node-classification readout `[x*, h₀]` pooled over the graph's nodes (pooling ∈ {mean, max, add, attention}, as HetSheaf's baselines) and an MLP; no new readout component; SheafPool is not used now (added later only if GC results call for it, with its own smoke test). Epochs (200 or 500): decided at training time. Stated limitations: variance reflects initialisation only (one split); MUTAG's test set is 20 graphs. No heterophily claim.

## X-10. §6 Tuning (line 217) and §7 OPEN-5 (line 235)
**Old (§6):**
> - **Tuning:** identical budget for every method and every comparison arm (`OPEN-5`), selection on validation only; search spaces published.

**New (§6):**
> - **Tuning (1.14; closes OPEN-5 for the main tables):** plain Bayesian search, **100 trials per (method, dataset)**, the same N for H2S-NN, every baseline and HetSheaf; no Hyperband or other early termination of trials (it feeds unconverged scores to the surrogate); tuning on seed 0 only, by the validation metric; the best configuration is then run on seeds 0–4. Search spaces are published per method (each baseline's space contains its published values). H2S-NN's space: `d`, `lr`, `weight_decay`, `encoder_dropout`, `classifier_dropout` (ranges in `docs/experiment_settings.html` §4.2); `K` and `layer_dropout` are removed (no `K` in v3-B; `layer_dropout` is not used by the anchored path). Epoch and patience budgets of HGB, H2GB and GC: decided at training time, the same for every method within a task. Compute is not a constraint (lab server and TSUBAME). Ablation arms are out of scope here.

**Old (§7):**
> - `OPEN-5` Tuning budget and search spaces (the same for all arms). `ℓ_max`, `proposal`, `path_weight_input`, solver tolerance, `T_max` and `τ`'s initialisation are never in a search space.

**New (§7):**
> - `OPEN-5` Closed for the main tables in 1.14 (§6 Tuning). Still open for ablation arms (discussion part). `ℓ_max`, `proposal`, `path_weight_input`, solver tolerance, `T_max` and `τ`'s initialisation are never in a search space.

## X-11. §6 Seeds and statistics (line 218)
**Old:**
> - **Seeds and statistics:** 10 seeds; paired Wilcoxon with Holm correction for each comparison of §5; Friedman + Nemenyi across datasets.

**New:**
> - **Seeds and statistics (1.14; main tables):** 5 seeds, 0–4, every task (as HGB; author: time). Reported: mean ± sample std (ddof = 1). Test: Welch's unequal-variance t-test, two-sided, unpaired (`scipy.stats.ttest_ind(equal_var=False)`), on every reported test metric, H2S-NN against the other method with the best mean test value on that dataset and metric; a method better than H2S-NN is reported too; zero variance in a group → "n/a". Committed now: every raw p-value and the number of tests are reported (≈ 40: NC 16, LP 16, GC 8), and both raw and adjusted values are given; which adjustment (Holm or Bonferroni) is decided after the results (author). Stated limitations: normality cannot be checked at n = 5; variance comes from seeds only (fixed splits). The validation metric that selects configurations and checkpoints is fixed before running: Macro-F1 (NC, GC), ROC-AUC (LP). The statistics of the §5 claims (C2–C14) are restated when the discussion part is designed.

## X-12. §6 Baselines (line 219): preprocessing per method
**Old:**
> - **Baselines:** all rerun under this protocol; no quoted numbers in result tables.

**New:**
> - **Baselines (1.14):** all rerun under this protocol, including HetSheaf (an adapter for node and link tasks is required; `src/baselines/registry.py:107-110` still lists it as cited-only); no quoted numbers in result tables. Data files, splits and negatives are shared; **feature preprocessing follows each method's original code, as part of the method**: H2S-NN uses features as stored (HGB's loader does not normalise); HetSheaf uses PyG `NormalizeFeatures` (shift by the minimum, divide by the row sum clamped at 1), which is what it proposes; every other baseline as its original code (inventoried per adapter, e.g. HGB's `feats_type`). Graph classification: constant features for every method (HetSheaf's protocol). Plan before running: protocol table with confirmed budgets, coverage matrix (which baseline runs on which dataset), one search-space table per baseline, one reproduction sanity check on a small dataset, small datasets first.

## X-13. §5 and §6 known-answer rows: out of scope marker
**Where:** after line 197 (`Removed in 1.11: C4, C4b (gates), C8 (filter bank).`) add:
> *(1.14) The comparisons and statistics of C2–C14 belong to the discussion/analysis part and are restated when that part is designed; the main-table protocol of §6 (5 seeds, Welch) does not define them.*

**And** line 221 (`- **Known-answer rows (1.7; …`): prepend "*(1.14: deferred by the author to the discussion part; pre-registration drafts in `docs/known_answer_preregistration_draft.md`.)*".

## X-14. §7 Designated extensions (line 238)
**Old (fragment):**
> graph-classification readout (SheafCovPool successor; appendix task, on the author's instruction only).

**New (fragment):**
> graph-classification readout (SheafCovPool successor, or HetSheaf's SheafPool if the plain pooled readout of §6 proves insufficient; appendix task, on the author's instruction only).

## Proposed changelog entry (prepended above the 1.13 entry when applied)
> **Changes from 1.13 (to 1.14; experiment settings, author decisions of 2026-11-11 recorded in `docs/experiment_settings.html` and `docs/experiment_settings_decisions.md`; the method is unchanged):** (H1) dataset rule (attributes or target labels; LastFM the exception); GC is typed-by-label, stated as a limitation; (H2) meta-path sets: Hetero²Net's lists where they exist, else its `get_metapaths`; only Hetero²Net Table 1 is a reproduction target; ℋ²'s set OPEN; (H3) MDE added as the attribute-heterophily descriptor; LP MDE on the F17 files; GC reports 1-hop `H_edge` with label = atom type; (H4) node-classification metrics Macro/Micro-F1 on all eight datasets; (H5) link prediction uses HGB's protocol (2-hop 1:1 negatives, ROC-AUC, MRR) on every task; HeaRT removed (supersedes F8, F14, F16 and the HeaRT parts of F21); (H6) graph classification follows HetSheaf's protocol in every point, seeds 0–4, plain pooled readout; (H7) tuning: plain Bayes, 100 trials, seed 0, then seeds 0–4; `K` and `layer_dropout` removed (closes OPEN-5 for the main tables); (H8) 5 seeds, Welch's t-test on every reported test metric, raw and adjusted p-values; (H9) baselines rerun with shared data and per-method feature preprocessing; HetSheaf adapter; (H10) claims C2–C14 and the known-answer rows deferred to the discussion part.
