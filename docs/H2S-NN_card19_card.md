# Card 19 — Heterophily on H2GB's meta-path set (MLH, MDE; spec 1.31 option A)

Written 2026-11-23 by the model design session. The author chose option A: ℋ², MLH and MDE all use H2GB's meta-path rule. Draft: `docs/spec_v1.31_draft_edits.md`. Report: `docs/H2S-NN_card19_report.md`.

**This is analysis only.** Runs are in progress, so this card must not touch any code that training, models, baselines or the perturbation generator use. One new script and one test file. No file under `H2S-NN/src/`, `H2S-NN/configs/`, `H2S-NN/scripts/perturbation/` or `vendor/` changes. `v3b` and the freeze (`0b9873d`) do not move.

## Why
- Table 1 mixes two sets. ℋ² uses H2GB's set, and MLH/MDE use Hetero²Net's lists. Example: ACM MLH is 0.404 in the table, but the perturbation study starts at 0.262 (H2GB's set).
- **MLH needs no new computation.** Card 6's ℋ² records (`docs/heterophily_evidence/records/nc__<ds>__h2.json`) already store `calcHomophily`'s `h_edge` for every meta-path of H2GB's set, per label problem and fill seed. MLH = 1 − mean `h_edge` over the same rows.
- **MDE must be computed** on H2GB's set for: ACM, ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem, H-Pokec, PDNS. Freebase stays "–". DBLP, IMDB and IEEE-CIS-G have the same set as before and serve as gates.
- **LP MDE (appendix, author 2026-11-23)** is computed on H2GB's set for acm_pp (paper), pubmed (disease), ogbn-mag_pp (paper) and h-pokec_uu (user). dblp_ap (paper) and imdb_ma (movie) serve as gates. freebase_bb and lastfm stay "–" (features removed, F17).

## Base and git
- Branch `card19` from `v3b` `0b9873d0c19455c8df56fa152492188d8918b62d`. Copy `vendor/` (never symlink).
- Use `sbatch` only (lab cluster). Stage by explicit path and push `card19`. Never merge into `v3b`. The numbers carry the `card19` commit hash.

## Steps
1. **Script** `H2S-NN/scripts/heterophily/h2gb_set_measures.py`. It only imports from `src.analysis.heterophily` (as `het`) and from `scripts/heterophily_stats.py` (`load_nc`, the same symmetrisation `het.symmetrise_facts`). It does not modify them.
   - `--measure mlh --records DIR`: reads every `nc__<ds>__h2.json` and writes, per dataset:
     - MLH = 1 − mean `h_edge` over the rows of the first fill seed (the primary value), plus the value per fill seed where the record has several;
     - IMDB: the mean over genres;
     - the labeled-only value from the `labeled_only` rows where present;
     - the path names, and the count of skipped empty paths.
     - No graph is loaded.
   - `--measure mde --dataset X [--metapath-index I]`:
     - Load the graph as card 6 does. Enumerate `het.h2gb_metapaths(sym.edge_types, target)`. **Stop** if the names differ from the `h2gb_metapaths` list in that dataset's card 6 ℋ² record.
     - For each path, build `mini = het.labeled_only_metapath_data(sym, path, target, all-True)`. Skip the path if `het._mini_gp_empty(mini)` (as `calcHomophily` does). Otherwise call `het.mde_measure(mini, "tgt", [[("tgt", "mid"), ("mid", "tgt")]], x.float(), both=<dataset in HGB>)`.
     - MDE = mean over non-empty paths.
     - `--task lp --dataset <task> --measure mde`: as card 6's LP branch (`heterophily_stats.py` `run`, task `lp`). Load the training graph and symmetrise the same way. For each endpoint type that keeps features, enumerate `het.h2gb_metapaths(sym.edge_types, t)` with that type as the target, then use the same per-path mini-graph route. The task relation's training facts stay in the graph, as card 6 had them. An endpoint type without features is "–" with the card 6 reason.
   - `--metapath-index` computes one path per job, as card 6's H2c-2 does, for long paths (H-Pokec `has_friend`, OAG). A `--collect` call joins the parts.
   - Output: one JSON per (dataset, measure) in `docs/heterophily_evidence/h2gb_set/records/`, with job, node, commit, wall time and peak memory, as card 6 records them. Also write `docs/heterophily_evidence/h2gb_set/table.md`: ℋ² (copied from card 6), MLH 1.30 → 1.31, MDE 1.30 → 1.31, number of paths and of empty paths.
2. **Gates (stop on failure):**
   - (G1) MDE by the new route equals card 6's MDE records on DBLP, IMDB and IEEE-CIS-G, and on LP dblp_ap (paper) and imdb_ma (movie) (relative difference ≤ 1e-6). If an LP gate set differs from the card 6 `m2` list, report the two lists and stop.
   - (G2) MLH from the records: ACM 0.2621, DBLP 0.1278, IMDB 0.3247, Freebase 0.8464 (±1e-4). These are the Part J / card 6 values on H2GB's set.
   - (G3) `git diff 0b9873d card19 -- H2S-NN/src H2S-NN/configs H2S-NN/scripts/perturbation vendor` is empty.
3. **Runs:** MDE for ACM, ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem, H-Pokec and PDNS.
   - Before submitting, estimate the `G_P` sizes from card 6's `plan` and `hubs` records.
   - Use card 6's memory settings, and split by path where card 6 did.
   - Expected empty paths: OAG 30 of 46 each; no others.
   - **3b (LP):** MDE for acm_pp, pubmed, ogbn-mag_pp and h-pokec_uu, plus the gates dblp_ap and imdb_ma. Split h-pokec_uu by path as in NC.
4. **Read-only check of H2GB's MLH.** H2GB Table 2 prints MLH 0.8731 (ogbn-mag), 0.7718 (mag-year) and 0.3922 (H-Pokec). Our MLH on H2GB's ℋ² set is 0.844, 0.749 and 0.386. Find in H2GB's code or paper which paths or which code give its Table 2 MLH, with `file:line`. Report only; change nothing.
5. **Perturbation consistency (read-only, no regeneration):**
   - The generator's `compute_heterophily.h2gb_metapaths` lists the same paths as `het.h2gb_metapaths` on ACM and DBLP.
   - The original-graph h in card 7's manifest equals the step 1 MLH (0.2621 / 0.1278).

## Tests (only these; no full suite)
`H2S-NN/tests/test_card19_h2gb_set.py`:
- (a) Toy graph with two relations between the target type and itself (like cite/ref) plus one target–other relation. For each H2GB path, the mini-graph MDE equals `het.mde_reference` on that path's `G_P` built by hand.
- (b) MLH from a hand-made ℋ² record (two paths, two fill seeds) gives the expected value.

## Report
Include:
- the script and the tests;
- gates G1–G3 with their numbers;
- the table of step 1, and the LP table (task, endpoint type, paths, MDE 1.30 → 1.31);
- the per-path MDE of ACM;
- step 4 with `file:line`;
- step 5;
- jobs, wall time and peak memory.

Last line: `branch=card19 sha=<sha> card=docs/H2S-NN_card19_card.md base=0b9873d0c19455c8df56fa152492188d8918b62d`.

Estimate: about 2 h of agent work, plus the MDE jobs (H-Pokec and OAG dominate, card 6 scale).
