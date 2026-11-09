# H2S-NN — Part J report: evaluation scope (spec v1.7 §1, §6)

**Branch:** `part-j` from `6ac401f`, 13 commits `24fe8f7`…`ddd2217`, plus this report. Not pushed or merged. CPU only; no training.
**Evidence:** `docs/heterophily_evidence/`, `docs/lp_evidence/`, `docs/advisor_report_cumulative/`.
**Tags:** RUN, CODE, SRC, DERIVED, UNVERIFIED, PROPOSAL.

## 1. Summary
1. **J0, heterophily (RUN).** The gate passed on the first attempt: DBLP MLH = **0.872** (published 0.87). The same code reproduces ACM 0.596 (0.60) and IMDB 0.237 (0.24) from Hetero²Net Table II; its Table VII swaps those two. ℋ² values: DBLP 0.173, ACM 0.399. IMDB ℋ² is undefined for multilabel targets, and Freebase awaits a convention decision.
2. **J1.1–J1.3 (RUN).** The three derived link-prediction tasks are built and registered; every leak check passes on all five tasks. Validation does **not** depend on the training seed. Two loader findings: it ignores `link.dat.valid`, and a reverse-only leak goes undetected.
3. **J1.4 stopped (card rule).** The HeaRT adaptation needs choices C1–C4. **J1.5:** PubMed labels are unavailable, so nothing was computed.
4. **J2.** Cumulative advisor report, 21 pages; chapters 1–2 are embedded verbatim.
5. **Suite:** 400 tests. The only failures are 20 in `test_baselines`, all pre-existing (K5 had 24; 4 now pass, none is new). One bug of mine in J0.3 was found at J0.5 and fixed (§3).

## 2. Files changed
- **`src/data/link_prediction_dataset.py`:** 3 entries in `_LOCAL_HGB_LINK_DATASETS` (+16 lines). This is the only `src/` change.
- **`scripts/heterophily/compute_heterophily.py`:** calls H2GB `calcHomophily` and Hetero²Net `homophily.py` from the vendor clones. Own code only for `G_P` outside `calcHomophily`, a float64 cross-check, H2GB's seeded fill and its loader symmetrisation.
- **`scripts/lp/build_derived_lp.py`:** splits by fact and writes the HGB link-prediction files and `split_meta.json`.
- **Tests:** `tests/test_heterophily_metrics.py` (11) and `tests/test_lp_derived.py` (24).
- **Docs:** pre-registrations, J0.2/J0.4/J1.3/J1.4/J1.5 records, results, inventory, the cumulative report, and its build and render scripts.
- **Not committed:** the clones `vendor/H2GB` (`d3b49db`) and `vendor/Hetero2Net` (`4913394`), which went into the empty clone targets, and `data/lp_derived/` (gitignored).

## 3. Deviations
- **J-docs message** as the author updated it: "docs: spec v1.7, agent prompt (Parts J, K), …".
- **Reference code was cloned, not pip-installed**, so the environment is unchanged. HeaRT's paper and code were read from the scratchpad, outside the repo.
- **J0 bug (mine, fixed):**
  - H2GB's `calcHomophily` fills unlabeled labels **in place**. My wrapper let it fill the caller's tensor, so Freebase class counts included the random fill, and the symmetrised graph, which shares the tensor, showed every book as labeled.
  - **Fix:** the function now receives a clone. A test checks a reference held from before the call, and the old wrapper is a 4th mutation (caught).
  - The seed-0 values that are reported did not change.
- **J0 extras, flagged:** MLH is reported under both meta-path sets (H2GB's and Hetero²Net's). The Freebase symmetrised variant is reported as a flagged row.
- **J1.2 extra file `link.dat.valid`.** It fixes the 9 % validation facts, which also stay in `link.dat` (HGB format). The loader does not read it (Finding 1).
- **No J1 negative tests:** J1.4 stopped before generating any negatives.
- **No independent review:** the author's instruction for this run was "write the report and stop" (§9).

## 4. Verification evidence (RUN)
- **J0.3:** `python -m unittest tests.test_heterophily_metrics` → 11 OK.
  - Hand toy (P-A-P: 12 pairs, `h_edge` 2/3, `h_adj` 1/9, `h_node` 11/16; aggregates 7/24, 7/9).
  - A random graph, where ours equals the reference, and the fill equals the reference.
  - `j03_mutation_check.py` → **4/4 caught**; the file is byte-identical after restore.
- **J0.4:** `compute_heterophily.py --datasets dblp` → APA `h_edge` 0.872222, |G_P| 11,113 including 4,057 self-loops (`j04_gate.md`).
- **J1.2:** `tests.test_lp_derived` → 20 OK at J1.2, 24 OK at J1.3. A real rebuild gives 18/18 byte-identical files (`j12_rebuild_check.txt`).
- **J1.3:**
  - `j13_loader_check.py`: all five tasks load. Seeds 3 and 11 give identical splits. Mutations: both rows caught, r row caught, **r′ row only not caught**.
  - Affected modules: 79 OK.
- **J3:** `unittest discover` → `Ran 400 tests … FAILED (failures=11, errors=9)`, all in `test_baselines` (`docs/lp_evidence/j3_suite_output_raw.txt`).
  - Compared with K5's list of 24: **no new failure**, and 4 now pass.
  - LastFM, PubMed and the split-overlap test pass because those datasets were downloaded on 2026-11-09, after K5.
  - `test_adapters_can_be_called_twice_in_one_process` passes because my J0.2 clone makes `vendor/H2GB` importable.

## 5. Measurements
**Heterophily** (heterophily form; mean over H2GB's length-2 set; `results.md`, `results.json`):

| Dataset | MLH (H2GB set) | ℋ² | MLH, Hetero²Net set, homophily (ours / published) |
|---|---|---|---|
| DBLP | 0.128 | 0.173 | 0.872 / 0.87 |
| ACM (7 paths incl. cite/ref) | 0.262 | 0.399 | 0.596 / 0.60 |
| IMDB | 0.763 | not computed (multilabel) | 0.237 / 0.24 |
| Freebase as loaded (80 % random-filled) | 0.853 | 0.999 (fill seeds 0.9987–0.9993); labeled-only diagnostic `H_adj` 0.892 | — (empty set) |
| ogbn-mag, mag-year, IEEE-CIS-G, H-Pokec | cited from H2GB Table 2 | | |

**Derived link-prediction tasks** (facts → train / val / test):

| Task | Facts | Train / val / test | Loader validation instead |
|---|---|---|---|
| ACM P–A | 9,949 | 8,059 / 895 / 995 | 554 |
| DBLP A–P | 19,645 | 15,913 / 1,768 / 1,964 | 1,170 |
| IMDB M–A | 14,779 | 11,971 / 1,330 / 1,478 | 761 |

## 6. Self-review checklist
- **J-docs: exactly four files, hashes equal to the given ones:** PASS (`git show` and `md5` of the committed blobs).
- **J0.2 (a)–(d) read from code with `file:line`:** PASS. (d) is defined only for `H_edge`, so IMDB ℋ² was not computed, per the stop rule's intent.
- **J0.3 toy, random graph, mutation:** PASS (4/4).
- **J0.4 gate:** PASS (0.872).
- **No definition changed to fit:** PASS. Every convention traces to code.
- **J1 split tests:** PASS (disjoint in either orientation; 81/9/10; deterministic; test nodes present).
- **Loader leak checks and the mutation:** PASS.
- **Negatives tests:** NOT CHECKED (J1.4 stopped).
- **J1.5:** PASS (reported, nothing computed).
- **J2 renders, numbers traced, source folders unchanged (md5):** PASS.
- **Only `link_prediction_dataset.py` edited in `src/`; nothing pushed or merged; staged by explicit path:** PASS.

## 7. Assumptions imposed for theory and reality checks
- **No THEORY-CONSTRAINT touched:** no model code changed.
- **Reality checks:**
  - **The published value pins a convention.** DBLP 0.87 is reproduced only with self-pairs kept (0.799 without).
  - **Class imbalance:** IEEE-CIS-G (cited) has MLH 0.135 but ℋ² 0.985, which supports ℋ² as the primary measure. On ACM the meta-path set alone moves MLH by 0.14 (0.262 vs 0.404 in heterophily form).
  - **Freebase:** the reference convention is dominated by random labels (80 % unlabeled), giving ℋ² 0.999, against a labeled-only diagnostic of 0.89.
  - **HeaRT on heterogeneous link prediction:** its common-neighbour heuristic is identically zero on DBLP A–P and IMDB M–A (DERIVED from the relation table).

## 8. Open items
- **Loader Finding 1 (J1.3):** the evaluated split is about 84/6/10, not 81/9/10. PROPOSAL: read `link.dat.valid` when it exists (about 15 lines).
- **Loader Finding 2 (J1.3):** a reverse-only row breaks ρ and is undetected. PROPOSAL: raise when a reverse-typed relation exists but ρ(target) is `None`. The J1.2 file-level guard covers the derived tasks now.
- **No preset or task-type entries in `config.py`** for the derived tasks (out of scope). Configs must set `task_type` and `target_relation`.
- **ACM paper and author features** have equal widths (1,902); whether they share a vocabulary is UNVERIFIED (matters for C2).
- **Freebase link-prediction target relation:** the author's choice, from `inventory.md`.

## 9. Decisions needed from the author
1. **IMDB ℋ² multilabel convention:** label-set classes, per-label average, or a Jaccard variant (`j02_reference_code.md` (d)).
2. **Freebase heterophily:**
   - unlabeled books: the reference's random fill or labeled-only;
   - the graph as stored or symmetrised like H2GB's loaders.
   The advisor report shows "pending" until this is decided.
3. **ACM meta-path set** for the paper's MLH and ℋ²: H2GB's full set (spec §1 wording) or Hetero²Net's three paths.
4. **J1.4 choices C1–C4** (`j14_heart.md`; proposal there): RA on cross-type pairs, feature similarity, which validation set, duplicates.
5. **Apply the two loader proposals?** And should PubMed and LastFM get fixed validation files?
6. **The Freebase link-prediction target relation.**
7. **Run an independent review of Part J? Push `part-j`?**

## 10. J4: decisions of 2026-11-09 (spec v1.8 F6–F11)
**Commits:** `3c63df0` (J4-docs) … `b58ba27` (J4.6) on `part-j`, plus this section. Evidence: `docs/heterophily_evidence/j4*`, `docs/lp_evidence/j4*`.

**Summary**
1. **J4.1 heterophily conventions (RUN).**
   - IMDB per genre: MLH **0.325**, ℋ² **0.737** (fill seeds 0.737–0.739); labelled-only 0.312 / 0.718; exact-match reproduction 0.237; Jaccard appendix 0.470.
   - Freebase symmetrised: **0.846 / 0.995**; labelled-only 0.478 / 0.882.
   - ACM and DBLP are identical to J0.5.
2. **J4.2 loader (RUN).** The loader reads `link.dat.valid` and guards the reverse-only leak.
   - The derived tasks now evaluate **81.00/9.00/10.00**.
   - The PubMed and LastFM files reproduce the carve exactly (every loaded tensor identical).
3. **J4.3 `freebase_bb` (RUN).** 103,303 / 11,478 / 12,754 unordered pairs; 0 pairs shared; 0 held-out rows in the message graph in either orientation. Task-level `H_edge` 0.779 (labelled-only 0.323).
4. **J4.4 HeaRT negatives for all six tasks (RUN).** No pool was too small (minimum eligible 1,281 ≥ 250), and every property passes on all 12 real files.
   - **Finding:** HeaRT's "RA" is coded as CN(a,v)/deg(v).
5. **J4.5:** IMDB protocol compliant in the trainer and all 12 adapters. **J4.6:** chapter 3 updated, 21-page PDF. **Suite:** 455 tests; the same 20 `test_baselines` failures as J3 (identical list), none new.

**Files changed**
- `src/data/link_prediction_dataset.py`, the only `src/` file: `_split_validation` (file or carve), `_assert_reverse_partner_is_tied`, the `freebase_bb` entry with `exclude_self_loop_positives`.
- **Scripts:** `compute_heterophily.py` (per genre, diagnostics, Jaccard, task-level `H_edge`); `build_derived_lp.py` (`build_task_by_pair`); new `write_valid_files.py` and `build_heart_negatives.py`.
- **New tests:** `test_lp_valid_files` (10), `test_heart_negatives` (22), `test_imdb_protocol` (6), plus 10 in `test_lp_derived` and 7 in `test_heterophily_metrics`.

**Deviations**
- **Same-type pair heuristic (J4.4).** It follows HeaRT's **code**, CN(a,v)/deg(v), verified verbatim at HeaRT's pinned `torch_sparse`, rather than textbook RA. The RA variant was built with the same seed and shares only 36–54 % of each list (`j44_record.md`).
- **J4.4's record** was completed in a follow-up commit (`65b74e4`), because an append failed on console encoding.
- **Choices not written in the card, fixed in J4.0 before any value:**
  - IMDB fill seed is `100·s + g`;
  - Jaccard is computed on labelled-only pairs (0/0 otherwise);
  - `freebase_bb` positives are the stored rows (1–2 per pair);
  - the fill excludes nodes already chosen, because HeaRT's top-k can take zero-score nodes on rank ties.
- **J4.2 asymmetry.** The PubMed and LastFM files list every relation's carve ("no numeric change"), while the derived tasks' files list only the target. So about 10 % of each non-target relation stays out of the PubMed and LastFM message graphs; before J4, every task lost it, which was unrecorded.
- **The J0.3 mutation script** no longer applies, because the wrapper line changed. Its 4 mutations are carried over in `j41_mutation_check.py`.

**Verification (RUN).** Mutations: J4.1 7/7, J4.2 3/3, J4.3 2/2, J4.4 4/4, J4.5 2/2 caught; every file was restored byte-identical. Plus the pre-registered exactness checks (`j41_record.md`) and the real-data checks (`j42_*`, `j43_*`, `j44_real_checks_output.txt`).

**Checklist** (each item is backed by a record):
- every heterophily value carries its convention and source: **PASS**;
- no definition, split or negative scheme was changed to fit: **PASS**;
- only the allowed `src/` file edited, staging by explicit path, AGENTS/CLAUDE/v2_plan/`vendor` uncommitted: **PASS**;
- HeaRT parameters and L3 quoted from source before code: **PASS** (`32ccfbf`).

**Reality checks.**
- **Features encode the target relation:** ACM author = OR of their papers' features, 5,959/5,959. DBLP and IMDB features are identical for identical neighbour sets, against near zero at random.
- **Random-fill share:** DBLP 54 %, IMDB 33 %, LastFM and Freebase 7 %, ACM and PubMed 0.

**Open items.** The trainer does not yet read the HeaRT files (later LP card). `config.py` has no presets for `freebase_bb` or the derived tasks. YouTube and Amazon are not on disk, so the guard is untested on them.

**Decisions needed from the author**
1. **Same-type pair heuristic:** HeaRT as coded (primary, files in `heart/`) or textbook RA (`heart_variant_ra/`).
2. **Endpoint features encode held-out links** (ACM, DBLP, IMDB): recompute those features from training facts, drop them for link prediction, or keep them and state it.
3. **Non-target carve:** keep PubMed and LastFM as written, or list only the target in their files so every task keeps its full message graph. The latter changes PubMed and LastFM numerically.
4. **`freebase_bb` scoring** of two-row pairs (one positive or two), and Hits@K for the LP card (HeaRT: Hits@10).
5. **High random-fill share on DBLP and IMDB:** accept, as HeaRT's rule gives it, or revisit.

**Independent review** (`docs/lp_evidence/j_independent_review.md`, unedited; responses in `j_review_response.md`)
- **BLOCKER, acm_pa.** Features alone rank the true pair first in 895/895 validation and 995/995 test HeaRT lists, so no acm_pa LP number may be reported before decision 2.
- **Correction.** "Heuristics read only training facts" is **FAIL for acm_pa**: the C2 cosine reads author features that include held-out papers. The acm_pa files must be regenerated after decision 2.
- **Fixed.** The loader now raises when a held-out pair of an unordered-pair task stays in the message graph reversed (`_assert_no_reversed_held_out_pairs`; tests and mutation in `j_review_fix_output.txt`).
- **Recorded:** the freebase_bb task-level `H_edge` without self-loops is 0.825 (labelled-only 0.353).
- **Deviations.** The pre-registered behavioural test for training-only heuristics was not written. The mutation scripts temporarily edit `src/` files other than the loader; they are restored byte-identical and nothing is committed.
- **Further decisions:**
  - the self-loop convention for the task-level `H_edge`;
  - the 1:1 secondary negatives on unordered-pair tasks must exclude reversed positives (LP card).

## 11. J5: link-prediction features, validation files and negatives (spec v1.9 F12–F16)
**Commits:** `acb0950` (J5-docs) … `c8b3b05` (J5.6) on `part-j`, plus this section. Evidence: `docs/lp_evidence/j5*`, `docs/heterophily_evidence/j56_record.md`. Link prediction only: node-classification data, splits, labels and heterophily values are untouched (41 release files md5-identical, `j53_*`).

**Summary**
1. **J5.1 rules (RUN; candidates fixed in J5.0).**
   - **ACM author** = OR of its papers' features, 5,959/5,959.
   - **IMDB actor** = MEAN of its movies' 3,341 token columns, 6,124/6,124 (|diff| ≤ 2.2e-16).
   - **DBLP author:** affected (identical paper sets give identical features, 150/150), but **no pre-registered rule is exact** (best 491/4,057). The diagnosis points to a selected keyword set, not an OR.
   - Paper and movie features are their own; ACM subject, DBLP term and IMDB director are not on the target relation.
2. **J5.2 leak before (RUN).** Features-only scorer MRR:
   - acm_pa **1.000** (the reviewer's 895/895, 995/995, reused);
   - imdb_ma **0.91 / 0.92**;
   - dblp_ap 0.086 (inexact rule; Hits@10 0.26 against the training-graph control's 0.04).
3. **J5.3 fix (RUN), in `data/lp_derived/<task>/node.dat` only.**
   - ACM author and IMDB actor are **rebuilt** from training facts. Bitwise equal on every node to an independent node-by-node computation, the loader's float32 tensor included; 847 and 773 cold nodes get zero vectors.
   - DBLP author is **dropped** (learned embedding).
   - The scorer now **equals the training-graph control exactly** (difference 0.0000): acm_pa MRR 0.02, imdb_ma 0.04–0.06. dblp_ap is at the all-tie level.
4. **J5.4 (RUN).** PubMed and LastFM validation files list only the target. The target tensors are identical to before, and every non-target relation is complete: PubMed +13,409 facts (10,748 message edges, because self-inverse relations are tied), LastFM +3,895.
5. **J5.5 (RUN).** HeaRT regenerated for acm_pa, PubMed and LastFM (same script, seed 42).
   - J4.4 properties pass on all 12 files. dblp_ap, imdb_ma and freebase_bb are unchanged and reproduce md5-identical from today's inputs.
   - On the new acm_pa lists the scorer equals the control. The random-fill shares follow under Reality checks.
6. **J5.6.** The `freebase_bb` task-level `H_edge` is reported **without self-loops: 0.8254** (labelled-only 0.3529), from the stored review evidence, not recomputed. `results.json/md` are updated; chapter 3 does not show the value.

**Files changed**
- **No `src/` file.**
- **New scripts:** `lp_feature_rules.py` (rule search), `lp_feature_leak.py` (scorer and Ctrl-G), `rebuild_lp_features.py`.
- **Changed scripts:** `write_valid_files.py` (`--mode target`, default); `build_derived_lp.py` (docstring: the rebuild runs after the build).
- **New tests:** `test_lp_feature_rules` (4), `test_lp_feature_leak` (4), `test_lp_feature_rebuild` (5).
- **Changed tests:** `test_lp_valid_files`. Two J4.2 "file = carve" tests are replaced by target-only tests through the loader, and one toy test is added.

**Deviations**
- **DBLP author MISS against the pre-registered expectation** of an exact rule. A thresholded rule was not added after seeing the data, so the type is dropped as F12 says.
- **dblp_ap before-fix MRR < 0.1** meets the pre-registered "wrong if" for the scorer. Diagnosis: the rule is inexact and the scorer sees only 133/334 columns.
- **Two bugs found by my own checks and fixed before use.**
  - A rule could count as "exact" with unmatched columns; caught by a mutation.
  - The real-data validation-file tests read the file themselves, so a loader that ignored it would pass; they now go through the loader.
- **J5.6** writes the loop-free values at the probe's 4-decimal precision. The section added to `results.md` is not produced by `compute_heterophily.py`.

**Verification (RUN).** Mutations J5.1 4/4, J5.3 2/2 (a real-data copy: the scorer beats the control by 0.98 when rebuilt from all facts), J5.4 2/2. Every edited file was restored byte-identical. Full suite: 471 tests, the same 20 `test_baselines` failures as J4.7 (identical list), none new (`j57_suite_summary.txt`).

**Checklist**
- No rule, split or tolerance was changed to fit: **PASS**.
- Rebuilt features are bitwise the rule on training facts: **PASS**.
- The scorer is at the control after the fix: **PASS**.
- Node-classification data untouched: **PASS**.
- No `src/` edit; staging by explicit path; AGENTS, CLAUDE, `v2_plan` and `vendor` uncommitted: **PASS**.

**Reality checks.** The random-fill share per task (F16):
- DBLP 54 %, IMDB 33 %;
- **ACM 23 % (was 0 %)**;
- LastFM 7 %, Freebase 7 %, PubMed 0.

ACM's new share is exactly its cold authors: 421/895 and 461/995 positives whose author has no training fact. The leaked features had hidden them.

**Open items.**
- The trainer does not yet read the HeaRT files.
- The RA-variant files are stale for PubMed (old graph) and unused.
- The rebuild must be re-run whenever `build_derived_lp.py` is.

**Decisions needed from the author**
1. **DBLP author features:** keep them dropped (F12 as applied), or pre-register a thresholded or top-k rule as a new unit.
2. **ACM random fill 23 %** replaces "ACM 0" in F16: accept and report it.

**Independent review** (`docs/lp_evidence/j5_independent_review.md`, unedited; responses in `j5_review_response.md`)
- **BLOCKER, imdb_ma (reproduced).** The kept movie features contain actor-like counts: attributes 3, 4, 7 and 12, `actor_1/3/2_facebook_likes` and `cast_total_facebook_likes`. Matching those values against an actor's **training** movies gives test MRR 0.085–0.102 and Hits@10 0.16–0.26, against the control's 0.044 / 0.056.
  - This is the J5 stop condition for imdb_ma. **No imdb_ma LP number may be reported** until the author decides.
  - J5.1 missed it because it only tested whether a type aggregates the other endpoint's features. Summary item 1's "movie features are their own" is wrong for these columns.
- **Gate 2 is weaker than its name.** Ctrl-G is a training-graph control, not a feature-free one (pre-registered with reasons). Given the bitwise gate, equality with it is implied, and it cannot see a leak outside the rule's columns.
- **Corrections:**
  - the bitwise check shares the column map with the rebuild (only the per-node aggregation is independent; the reviewer's own reimplementation agrees on every node);
  - the ACM cosine before the fix was not measured;
  - about 47 % of acm_pa positives have an isolated author, which no method can rank.
- **Fixed:** the toy bitwise test now reads the loader's tensor through the loader.
- **Further decisions:**
  3. **IMDB movie columns:** drop the actor-like columns (and `director_facebook_likes`) or all numeric attributes in the LP files; drop movie features for LP; or keep them and state it.
  4. **A stronger leak gate** before LP numbers: a features-only scorer over all stored features of both endpoints.
  5. **A loader guard** against a stale `node.dat` after rebuilding a split.
  6. **C2 on IMDB** over the shared 3,341 columns.
