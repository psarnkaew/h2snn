# Task card — Part P: baselines runnable on every local cell (HetSheaf adapter, derived LP tasks, Freebase NC)

**Design authority:** spec **v1.19** (once applied; v1.18 + `docs/spec_v1.19_draft_edits.md` until then) (§6 Baselines, Node classification, Link prediction, Graph classification, Tuning) and `docs/experiment_settings_decisions.md`. `H2S-NN/AGENTS.md`/`CLAUDE.md` bind for environment (conda env `h2snn`), coding, testing (`unittest`) and evidence tags. `src/baselines/protocol.py` (module docstring) is the adapter contract.
**Branch:** `part-p`, created from the last commit of `part-o` (hash given in the prompt). **Report:** `docs/H2S-NN_partP_report.md`; evidence `docs/partP_evidence/`.

## Goal and rule
Every **local** cell of the main tables becomes runnable: (1) HetSheaf as a baseline on NC, LP and GC; (2) the four derived LP tasks (acm_pa, dblp_ap, imdb_ma, freebase_bb) for every LP baseline; (3) Freebase NC for H2S-NN and every NC baseline. **Correctness only** (author, 2026-11-12/13): unit tests, tests that run an adapter's `fit_task` for one epoch on a tiny synthetic release in the real file format, mutation checks, the full suite. **No smoke run on real data, no performance experiment, no metric reported as a result, no comparison between methods.** H2GB datasets stay "data pending" (mini-batch card). WISP and the method are not touched.

## Scope rules
- One commit per unit; stage by explicit path; never stage line-ending-only adapter diffs, `Claude outputs/`, `data/`, vendor clones (except a `CLONED_COMMIT.txt` convention file), or other sessions' files.
- Each baseline keeps its own preprocessing as its original code does (spec v1.14 H9); data files, splits and negatives are shared (spec v1.16–1.18). Shared protocol (AdamW, shared lr/wd/dropout axes, no scheduler, no clipping, budgets per source, failed-trial rule) applies to every adapter (spec v1.15 J4, v1.16 K5).
- Forbidden: loosening tolerances, deleting or skipping tests, silencing exceptions, special-casing inputs, editing the spec to fit the code. Out-of-scope problems are listed, not fixed.

## Units

### P0 — Preconditions and two readings
Read spec v1.18 §6, `src/baselines/protocol.py`, `registry.py`, `bundles.py`, one LP adapter end to end (e.g. `hgb_gnn.py` + `_homogeneous_lp.py`), and HetSheaf: `references/HETSHEAF.pdf` (§5 and Appendix C: which model variants appear in the main NC/LP/GC tables and under what names), `vendor/Heterogeneous-Sheaf-Neural-Networks` (commit in the report) `hetsheaf/scripts/run_{sheaf,attention_sheaf,poly_sheaf}_{nc,lp,gc}.py`, `node_classification/sheaf_node_classifier.py`, `link_prediction/sheaf_link_predictor.py`, `graph_classification/sheaf_graph_classifier.py`, `datasets/hgb.py`, `configs/`, `sweeps/{nc,lp,gc}/`. Report, with file:line: (a) the variants the paper's main tables report as HetSheaf rows (diag/bundle/general sheaf, attention sheaf, poly sheaf; names as in the paper); (b) for each variant, its sweep's search space; (c) HetSheaf's NC head, LP decoder and loss, GC readout (`late_token` = SheafPool); (d) every dataset transform (`hgb.py:53-58`: Constant, ToUndirected, RemoveSelfLoops, NormalizeFeatures, RemoveDuplicatedEdges) and where node/edge types enter the model. No code in P0.

### P1 — HetSheaf adapter (NC, LP, GC)
New `src/baselines/adapters/hetsheaf.py` (+ helpers as needed), importing HetSheaf's model classes from the vendor clone (no copy of their code into `src/`), registered in `registry.py` (replaces `ADAPTER_PENDING["hetsheaf"]`), `runtime/vendor_paths.py`, `runtime/dependencies.py`.
- **Variants:** implement every variant P0(a) lists as a parameter `variant` of the one adapter. **Default campaign row (author may change after the report; recorded under decisions):** one registry row per variant the paper's main tables report.
- **Inputs:** the canonical bundle (shared files/splits/negatives) turned into HetSheaf's homogeneous graph with node-type and edge-type attributes exactly as `datasets/hgb.py` builds it, **including HetSheaf's transforms of P0(d)** (they are part of the method, spec H9). State any transform that cannot apply to a shared split (e.g. one that would add a held-out LP edge) and do not apply that one; list it.
- **NC:** HetSheaf's own model and head (protocol: NC runs the authors' head); multi-label IMDB through the project's IMDB rule (BCE per genre, logit > 0) as every adapter does.
- **LP:** HetSheaf's encoder + the shared LP head (`src/baselines/heads.py`), shared training/validation/test negatives (HGB's functions), both test sets (2-hop, random).
- **GC:** HetSheaf's encoder; readout per decision **D-P2** below (default: HetSheaf's own readout options incl. `late_token`).
- **Search space:** each variant's sweep space from P0(b), with the shared axes (lr, weight decay, dropout ranges of `search_grids.py`) replacing theirs where they overlap; their published values must be reachable (`SearchSpaceCoverageTests` pattern).
- **Shared protocol:** AdamW, no scheduler (their `configure_optimizers` builds a cosine scheduler: not used), no clipping, budgets per task/source, failed-trial rule.
**Tests:** a tiny HGB-format NC release, a tiny derived-format LP release, a tiny TU-like GC set: `fit_task` runs one epoch per variant per task and returns a record with the task's keys (values not asserted beyond finiteness); the homogeneous graph's node/edge type attributes equal a hand construction; each transform of P0(d) is applied (and the excluded one is not); AdamW, no scheduler; search space contains the published values. **Mutation:** drop the edge-type attribute → a test fails.

### P1b — H2G-former's encoder as H2GB's configs (spec v1.19 T2)
`src/baselines/adapters/h2gformer.py` (and `_h2gformer_lp.py`): replace the fixed `Hetero_Raw` encoder by H2GB's stack per dataset as spec v1.19 T2 states (H2GB datasets: their config's `gnn.node_encoder_name`; HGB NC: `Hetero_Raw+Hetero_Label`; LP and GC: `Hetero_Raw`), using H2GB's own encoder classes (`H2GB/encoder/hetero_label_encoder.py`, `hetero_pos_encoder.py`) and, for `Hetero_Metapath`, H2GB's `preprocess_Metapath` / `load_Metapath`. The H2GB datasets stay "data pending": wire and test the Metapath path on a tiny HeteroData only.
**Tests:** the encoder stack resolved per (dataset, task) equals the table of T2; **information-route test:** the label channel's input contains no validation or test label (only training labels, masked as H2GB's encoder masks them), checked by poisoning validation/test labels and asserting the encoder input is unchanged; MetaPath2Vec embeddings are produced and consumed on a tiny graph with a walk list. **Mutation:** feed all labels (not only training) to the label channel → the route test fails.

### P2 — Derived LP tasks for every LP baseline
Remove the `SUPPORTED_LP_DATASETS` restriction where it only reflects missing wiring, and make every LP adapter (GCN, GAT, NSD, CSNN, HAN, PSHGCN, H2G-former, Hetero²Net, H2SGNN, HetSheaf) run on acm_pa, dblp_ap, imdb_ma, freebase_bb through the canonical bundle. Points to handle per adapter, each stated in the report: types without features after F17 (ACM author, DBLP author, IMDB actor: removed blocks) handled by the adapter's own featureless rule; reverse relations of the target in the message graph (held-out facts removed in both orientations, as the loader does); PSHGCN/H2SGNN relation keys (`_polynomial_lp.py`, 2-character keys, type chaining) on graphs with 3–4 node types; the 2-hop "–" cells (dblp_ap, imdb_ma) yield `None` with the reason, never 1 or NaN. Registry: the derived-LP pending reason is removed for every runnable cell; a cell that cannot run keeps a specific reason.
**Tests:** a tiny derived-format release (two node types, a target relation and its stored reverse, one featureless type): every LP adapter's `fit_task` runs one epoch and its record has both test sets' keys; no held-out pair reaches any adapter's message graph (either orientation); PSHGCN/H2SGNN keys on the tiny release equal a hand list. **Mutation:** keep the reverse copy of a test fact in the message graph → a test fails.

### P3 — Freebase NC for H2S-NN and every NC baseline
- **H2S-NN:** an NC preset and loader for HGB Freebase (target type book, HGB's labels and its 24/6/70 split rule as for ACM/DBLP/IMDB; no node features: learned per-node vectors, as the encoder already does for featureless types; relations without a stored reverse fall back to untied facts as D7 states, reported per relation). Full batch (spec v1.17 Q3); its cost is the lab cluster's (not measured here).
- **Baselines:** `bundles.py` (`HGB_RAW_DIRNAME`, `HGB_DEFAULT_FEATS_TYPE`), each NC adapter's Freebase handling (featureless), registry pending reason removed where runnable.
- **Unlabelled books** (32,448 of 40,402) stay in the graph and enter neither loss nor metrics, as IMDB's unlabelled movies.
**Tests:** the loader's counts equal the M1 gate's recorded Freebase counts (nodes per type, labelled 7,954, class counts); split sizes and disjointness; the untied-fallback relations listed equal a direct check of the files; each NC adapter's `fit_task` runs one epoch on a tiny featureless HGB-format release.

### P4 — Registry, coverage matrix, full suite, report
Registry after P1–P3: every (model, task, dataset) cell runnable or with a specific reason. Write the **coverage matrix** (model × task × dataset: runnable / reason) to the report and `docs/partP_evidence/coverage.json`, with the count of runnable cells per task (the input of the tuning cost). Full suite once. Report per unit (done, VERIFIED-RUN, deviations), P0's readings with file:line, the coverage matrix, "Decisions for the author", out-of-scope findings.

## Decisions for the author (defaults used unless he changes them; listed again in the report)
- **D-P1 HetSheaf rows:** one row per variant that the paper's main tables report (P0(a)). Alternative: one "HetSheaf" row with the variant as a tuned categorical.
- **D-P3 H2G-former on datasets without an H2GB config:** HGB NC `Hetero_Raw+Hetero_Label`; LP and GC `Hetero_Raw` (spec v1.19 T2, recommended; author to confirm).
- **D-P2 HetSheaf GC readout:** HetSheaf's own readout options, including `late_token` (SheafPool), because the readout is part of their method (spec 1.14 anticipated it). Alternative: the shared GC head (pooling + one Linear) as every other GC baseline.

## Stop conditions
- P0 finds that HetSheaf's code cannot run in `h2snn` (dependency conflict): stop P1, report the conflict; P2–P3 continue.
- A transform or protocol point cannot be followed without changing shared data/splits/negatives: do not change them; report.
- A change would alter H2S-NN's method or needs an open decision: stop the unit, report.
**Not stop conditions:** any metric value in a test (only finiteness and keys are asserted).
