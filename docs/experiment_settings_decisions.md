# Experiment settings: decisions log (H2S-NN)

Started 2026-11-11 by the design session. Single place where experiment-setting decisions are written as they are taken; merged into the spec (next version) only after the author approves. The spec is not edited from this file directly.

## Pending work (read before every new topic)
- [x] Review of L8 (report section 8) done 2026-11-11: all pass; part-j = v3b = 3250d2b; T3/m4 decision (a); finding: Thm 4 wording (in spec draft E-6).
- [x] Draft spec edits (item 5) -> docs/spec_v1.12_draft_edits.md (awaiting author approval; adds Thm 4/L4' and Part L decision wording): stale lines in spec section 6 (filter-bank cost gate; ablation table C2-C4b, C8, C9), C13 wording without v2 time, T16 / "hub_A = v2" wording; then commit of v1.11 (author).
- [x] Draft pre-registrations S1'-B, S3, S2' (item 6) -> docs/known_answer_preregistration_draft.md (5 open decisions); author to classify the local known-answer runs (now vs cluster).
- [x] Experiment-settings topics of 2026-11-11 (author list): 1 mini-batch, 2 heterophily metrics (MLH, H2, MDE), 3 datasets per task (+ HSB-GDL), 4 search spaces, 5 HGB protocol vs HetSheaf (node, link, seeds), 6 graph classification as HetSheaf. Report written 2026-11-11: docs/experiment_settings.html (author decisions D1-D8 open).

## Decisions (author)
| Date (JST) | Topic | Decision | Source |
|---|---|---|---|
| 2026-11-11 | C13 timing | No paired v2/v3-B timing; memory and PCG from L7 are the results; time vs v1/baselines later on the server | chat |
| 2026-11-11 | Heterophily | Report MLH, H2 and Meta-path Dirichlet Energy (attribute heterophily) for every dataset; "-" where not measurable | chat |
| 2026-11-11 | Datasets | Avoid datasets with a single node type, for every task | chat |
| 2026-11-11 | Spec | E-1–E-7 of docs/spec_v1.12_draft_edits.md approved; applied as spec v1.12 (G11–G16), uncommitted | chat |
| 2026-11-11 | Known-answer | S1'-B approved as drafted (docs/known_answer_preregistration_draft.md); S2' option (a) chosen (rerun K2b graph under v3-B, read-out criterion unchanged, C2b wording narrowed), local, <= 2 h; S3 pending | chat |
| 2026-11-11 | Known-answer | Author: defer S1'-B, S3 and S2' (KA1 prompt not run) to the discussion/analysis part of the experiments. Note: the idealised S1'-B check (d = 1, identity maps) is not a model run; capacity of the weight channel under v3-B is not shown until S1'-B's full+oracle arm runs | chat |
| 2026-11-11 | Datasets | Keep both ogbn-mag and mag-year (same graph, different labels); each gets its own meta-path set by the rule (ogbn-mag: Hetero2Net list PFP, PAP; mag-year: get_metapaths); different MLH/H2 values are acceptable (H2GB also reports different values) | chat |
| 2026-11-11 | Heterophily (GC) | 1-hop H_edge with label = atom type is reported as is; stated as a limitation (no truly heterogeneous GC dataset exists) | chat |
| 2026-11-11 | Compute | Tuning cost not a constraint (lab server + TSUBAME) | chat |
| 2026-11-11 | Feature preprocessing | Shared data files and splits per protocol (NC/LP: HGB, features as stored, no normalisation; GC: HetSheaf, constant features). Each method keeps the preprocessing of its own original code, as part of the method: H2S-NN none; HetSheaf PyG NormalizeFeatures (shift by the global minimum per type, divide by row sum clamped at 1; VERIFIED-SOURCE torch_geometric 2.6.1 normalize_features.py); every other baseline as its original code (to be inventoried per adapter). MDE uses raw features | chat |
| 2026-11-11 | Experiment settings (from experiment_settings.html, author answers D1-D8) | Mini-batch: design first, not implemented; dataset rule: attributes or target labels (LastFM exception); meta-paths: Hetero2Net lists else get_metapaths, reproduce Hetero2Net Table 1 only; seeds 5 (0-4) all tasks; search space: K and layer_dropout removed; GC: HetSheaf in every point, seeds 0-4; D1 LP MDE on F17 files; D2 GC 1-hop H_edge with label = atom type; D3 TU typed-by-label, HSB-GDL waiting for its author; D4 no extra H2GB dataset; D5 plain Bayes 100 trials, tune on seed 0, run seeds 0-4; D6 NC/LP per HGB/H2GB, GC per HetSheaf, rerun all baselines incl. HetSheaf; LP: HGB protocol on every task, no HeaRT; tests: Welch on every reported test metric; metrics NC Macro/Micro-F1 (all 8), LP ROC-AUC + MRR, GC Macro/Micro-F1; validation NC/GC Macro-F1, LP ROC-AUC; readout: no new component; deferred: budgets, p-adjustment (raw + adjusted reported), cost table | experiment_settings.html |
| 2026-11-11 | Spec | Draft v1.14 edits written (docs/spec_v1.14_draft_edits.md, X-1..X-14) on top of v1.13 (another session's G17); awaiting approval | chat |
| 2026-11-11 | Heterophily | H2 uses the same meta-path set as MLH and MDE | chat |
| 2026-11-11 | Metrics | Macro-F1 (every NC dataset, every method): classes fixed per split from that split's true labels, passed as labels=L; IEEE-CIS-G: H2GB's F1 is Macro-F1 (author) | chat |
| 2026-11-11 | Spec | v1.14 applied (H1-H10) on top of v1.13; backup docs/spec_backups/H2S-NN_methodology_spec_v1.13.md; uncommitted | chat |
| 2026-11-11 | Graph classification | Follow HetSheaf's code protocol (details to be documented) | chat |
