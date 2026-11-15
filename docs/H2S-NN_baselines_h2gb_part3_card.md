# Task card — Baselines on H2GB, part 3: the stop items of part 2 (card 1c)

**Design authority:** `docs/H2S-NN_baselines_h2gb_card.md` (part 1) and `docs/H2S-NN_baselines_h2gb_part2_card.md` (part 2) with their rules, as amended by the decisions log row "Baselines on H2GB part 2: E0-E4 stop items (G-1..G-10)" (2026-11-16); part 2 report `docs/H2S-NN_baselines_h2gb_part2_report.md` and evidence `docs/baselines_h2gb_evidence/p2_*`. `docs/working_preferences.md` binds, incl. **"Slim preparation"**. `H2S-NN/AGENTS.md`/`CLAUDE.md` bind.
**Branch:** `baselines-h2gb-3` from the `v3b` SHA in the prompt (= approved card 1b), then `git merge --no-ff` of the pinned `design-docs` SHA. **Report:** `docs/H2S-NN_baselines_h2gb_part3_report.md`; evidence `docs/baselines_h2gb_evidence/` (prefix `p3_`).
**Runs in parallel** with `dry-run` (card 2, D4b) and `hetsheaf` (card 3 Part A). **Files:** as part 2 (`src/baselines/**`, `configs/baselines/**`, baseline tests, evidence, report) **plus** regeneration of `notebooks/baselines/` with `scripts/generate_baseline_notebooks.py` (G-9; the generator itself is not edited unless the regeneration fails, then stop item) and `H2S-NN/requirements/` lock lines for `sparse_tools` (G-2). PSHGCN outside MAG and H2SGNN are **not** in this card (card 3 Part B).

## Units (in order; Slim tests only; a unit that stops is reported and the next independent unit starts)
### F0 — Preconditions (no code)
Branch check and merge. For each item below record the vendor file:line it relies on. A file:line that contradicts a decision is a stop.
### F1 — HINormer on H2GB (G-1)
The original HINormer code (vendor `HINormer`, the HGB adapter's model) with its own batching as its source runs it (local-structure encoder and relation encoder as written, node sequences sampled by its own routine, target batches as its training script), class B, on the 8 H2GB NC datasets; H2GB's `HINormerModel` not used (reason stated in the registry). **Tests:** one step finite and moves; no held-out label reaches a sequence (if its inputs carry labels: route test + mutation; otherwise state "no label input", file:line).
### F2 — PSHGCN on ogbn-mag, mag-year (G-2, MAG part)
Their ogbn-mag code in full (S3 (b) widths, MAG relation list, label keys, self-effect removal). Install `sparse_tools` from its source as their README says (record version/commit and build log; `pip check` unchanged otherwise; torch stays 2.4.1+cu124). If the build fails: stop item for the label part, the cells stay pending. **Tests:** PSHGCN equivalence of part 2 E2 (if not yet done); label part: no validation/test label enters the label features (route test + mutation).
### F3 — SeHGNN ogbn-mag, mag-year (G-4)
Their README's "without extra embeddings" command (inputs for featureless types as that command builds them). **Test:** one step finite and moves.
### F4 — Hetero2Net (G-5)
Read its metapath lists for HGB, MAG, RCDD (`hetero2net/dataset.py:20-67`). If one rule generates all of them (state the rule and show it reproduces each list exactly, test), apply it to OAG×3, IEEE-CIS-G, H-Pokec, PDNS; otherwise those cells are N/A with the reason. MAG cells read `mag_metapath2vec_emb.zip` (sha256 at first use) and wait while it is absent. NeighborLoader path as its `main.py:147-175`. **Tests:** rule reproduces the source lists; label channel `y_emb` on sampled batches: no held-out label reaches a target (route test + mutation); one step finite and moves.
### F5 — ChebNetII IEEE-CIS-G, H-Pokec (G-6)
Featureless types: per-type projection first (an embedding table for an identity block), propagation in the forward. **Test:** the existing linearity equality covers it (state); one step finite and moves on a synthetic graph with an identity block.
### F6 — H2G-former mag-year (G-7)
H2GB's `preprocess_Metapath` with ogbn-mag's schema for mag-year (same graph), stated. **Test:** one step finite and moves.
### F7 — Large LP (G-8, G-9, G-10)
On `large_lp_training.py`: GCN, GAT, NSD, CSNN, HAN, H2G-former, Hetero2Net (methods with a local-LP path). H-Pokec U-U: H2GB's `hgt` sampler with its per-type lists unchanged, seeded by each batch's endpoints on the message graph (G-8). Regenerate the baseline notebooks (G-9). **Tests:** the held-out-fact test of E4 run with each new loader configuration (incl. the H-Pokec hgt path) and its mutation once; one LP step finite and moves per method.
### F8 — Suite and report
Registry: every H2GB NC and large-LP cell is runnable, waiting for a named file, moved to card 3 Part B (PSHGCN non-MAG, H2SGNN), or N/A with a reason. Full suite once on one node; golden/mb0 only if `src/` outside `src/baselines/` changed (it should not). Report short: cell table, deviations, decisions for the author. Last line: `branch=baselines-h2gb-3 sha=<full head sha> card=docs/H2S-NN_baselines_h2gb_part3_card.md base=<pinned v3b sha>`.

## Rules and stop conditions
As part 2 (data, cluster, git, forbidden list, stop / not-stop), with the file list above.
