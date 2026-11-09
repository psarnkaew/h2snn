# Independent review: Part J + J4 (diff 6ac401f..HEAD)

## Verdict
The engineering is mostly sound. The derived splits, the `link.dat.valid` loader path, the PubMed/LastFM "no numeric change" files, the HeaRT re-implementation (I checked it line by line against HeaRT's own `create_heart_negatives.py` and `calc_ppr.py` in the scratchpad), the IMDB protocol tests and the heterophily conventions all do what the records say. The touched test modules pass when I re-run them. Every chapter-3 number traces to a committed output.

Two things would sink the link-prediction part of a paper:
1. **acm_pa can be solved from node features alone.** The rule "paper's feature support is contained in the author's feature support" ranks the true pair first in **895/895 validation and 995/995 test** HeaRT lists. The ACM HeaRT negatives were also selected with a feature heuristic that reads held-out facts. The agent disclosed the feature dependence as an author decision, but did not quantify it and still marked "heuristics read only training facts" PASS.
2. **The loader's leak checks cannot see a reciprocal leak on `freebase_bb`.** That task is leak-free only because the builder wrote correct files. "Loader leak checks pass" is no evidence for this property.

Neither is a coding error against the card's literal text. Both must be resolved or stated before any LP number on acm_pa or freebase_bb is reported.

## Findings

1. **BLOCKER: acm_pa's held-out links are readable from node features, and trivially so.**
   - Evidence: `docs/lp_evidence/j40_data_checks_output.txt` (each author's vector = OR of all own papers, 5,959/5,959, held-out papers included).
   - VERIFIED-RUN, my probe `scratchpad/review_j/acm_leak_probe.py` on `data/lp_derived/acm_pa/heart/`:
     - positives whose paper support lies inside the author's support: 1.0000 (valid and test);
     - HeaRT negatives with that property (1-in-10 sample): 0.0000;
     - positives ranked strictly first by that rule alone: **895/895 (valid), 995/995 (test)**, i.e. MRR = 1 with no model.
   - DBLP and IMDB show the same dependence (identical neighbour sets give identical features). I did not quantify it there; SUSPECTED similar.
   - Why it matters: any acm_pa LP result is uninterpretable until the features are recomputed from training facts or dropped. Report §10 lists this as decision 2 but understates it ("A model can read held-out links").

2. **MAJOR: the ACM HeaRT negatives use a heuristic that reads held-out facts.**
   - Code: `scripts/lp/build_heart_negatives.py:83` (`acm_pa … features=True`) and `:384-386` (cosine on author features). Since author features include held-out papers, C2 is not "computed on training facts only".
   - The J4.4 gate item "heuristics read only training facts" is marked PASS (`j44_record.md`, report §10). Yet `tests/test_heart_negatives.py:151-166` checks only the graph and the L3 matrix, never C2.
   - The J4.0 pre-registration already showed the leak (it was the evidence for "one vocabulary"). The ACM files are therefore not training-only, and they will need regenerating if decision 2 changes the features. Not stated. VERIFIED-CODE.

3. **MAJOR: the loader's leak checks cannot detect a reciprocal leak on `freebase_bb`.**
   - Cause: ρ(rel_0) is `None`.
     - `_assert_no_message_passing_leak` (`link_prediction_dataset.py:762-780`) compares stored orientation only.
     - `held_out_fact_leaks` (`fact_layer.py:216-249`) checks the reverse only through ρ.
     - `_split_validation` with a file removes only the listed orientation for a relation without a partner (`:432-441`).
   - VERIFIED-RUN, `scratchpad/review_j/reciprocal_probe.py` on the repo's own `write_freebase_like` fixture. Both cases load with no error and leave 1 held-out row reversed in the message graph:
     - (A) the reverse row of a one-row test pair is put back into `link.dat`;
     - (B) `link.dat.valid` lists only one orientation of a two-row pair.
   - The card's J4.2(a), "removed from the message graph in both orientations", is not enforced by the loader for such relations.
   - The real files are clean (the separate script `j43_loader_check_output.txt` shows "reversed 0"). But leak freedom rests only on `build_task_by_pair` and the file-level test. The stop condition "the pair split cannot guarantee leak freedom" is met by the builder, not by the loader.
   - Fix to propose: a dataset flag such as `split_unit: unordered_pair` that makes the loader check and remove held-out pairs in both orientations.

4. **MINOR: the loader's own negatives for `freebase_bb` can be reversed positives.**
   - `_sample_random_tail_negatives` (`:578-604`) excludes known tails by stored orientation only. For a pair stored only as (w, u), the pair (u, w) can be drawn as a validation negative. The same likely holds for `_sample_hgb_eval_negatives` (not read).
   - The HeaRT files do mask both orientations (`partners()`, `:355-367`). But until the LP card switches the trainer to those files, the running pipeline contradicts the unordered-pair semantics. VERIFIED-CODE (sampler); the actual occurrence count was not measured.

5. **MINOR: the reverse-partner guard may wrongly reject datasets with several same-type relations.**
   - `_assert_reverse_partner_is_tied` (`:444-463`) treats any other same-type relation as "reverse-typed" when the target is same-type. Amazon (`target_relation_ids (0, 1)`, both product→product) would raise whenever ρ(rel_0) is `None`, although a co-purchase relation is not a stored reverse of co-view.
   - This could break previously loadable configs. The data is not on disk; disclosed only as "untested". SUSPECTED.

6. **MINOR: task-level `H_edge` for `freebase_bb` includes the 15,036 self-loops, which are never task positives.**
   - Code: `compute_heterophily.py:643` uses `to_undirected` with loops kept. Loops always agree.
   - VERIFIED-RUN (`scratchpad/review_j/fb_loops_probe.py`): labelled-only diagnostic 0.3229 with loops (1,335 labelled loops among 15,707 edges), **0.3529 without**. The primary 0.7794 is affected the same way.
   - The record mentions the loops in its diagnosis but does not report a loop-free value.

7. **MINOR (scientific): the primary Freebase values mostly measure label coverage, not heterophily.**
   - About 80 % of books are random-filled with uniform labels while the labelled class distribution is skewed (class 1 = 49 %). So ℋ² 0.995, MLH 0.846 and task `H_edge` 0.779 are largely set by the fill. Labelled-only: 0.882 / 0.478 / 0.323.
   - IMDB per genre: filled movies get each genre with p = 0.5, against prevalences of 24–55 %.
   - These are author-mandated conventions (spec v1.8) and disclosed. Still, chapter 3's single table places Freebase 0.995 beside fully labelled datasets as if comparable; a reviewer will call it an artefact.

8. **MINOR: the J4.4 tests are weaker than pre-registered.**
   - `lp_evidence/j4_preregistration.md` §J4.4 says the full-graph mutation test "checks … that negatives equal those from the training-only graph".
   - The actual test (`tests/test_heart_negatives.py:151-166`) only checks edge membership of the heuristic graph and the L3 matrix. A mutation that builds the graph correctly but scores on another one would pass. The cosine heuristic is untested for training-only use (see finding 2).

9. **MINOR: inconsistent non-target carve (disclosed).**
   - PubMed and LastFM drop about 10 % of every non-target relation from the message graph; the derived tasks drop none (`j42_record.md`). Cross-task comparisons of heuristic hardness and model scores are confounded.
   - Properly recorded as decision 3. Listed here because the HeaRT files for PubMed and LastFM were generated on the reduced graph and would change if the decision changes.

10. **NIT (bookkeeping and wording).**
    - `j41_record.md` says `test_heterophily_metrics` has 17 tests (11 + 6); the file now has 18 test methods (report §10 says +7).
    - In `results.md` the column "Reproduction … (ours / published)" holds Freebase 0.1542, which has no published counterpart.
    - Chapter 3, "as the benchmark that released IMDB does": HGB defines per-genre classification, not a per-genre heterophily measure.
    - Chapter 3, "359 … handled as described for Freebase": the Freebase note also covers symmetrisation, so the reference is ambiguous.

11. **NIT: PubMed training positives include its target's self-loops.** HeaRT drops loops when reading (`hard_utils.py:110-111`). VERIFIED-RUN: `link.dat.valid` and `link.dat.test` have 0 loops, so only training is affected. This predates J4 and is outside its scope.

12. **NIT: the mutation scripts temporarily edit `src/` files beyond the allowed one.** `j45_mutation_check.py` edits `src/utils/misc.py` and `src/baselines/metrics.py`; `j43_mutation_check.py` edits `link_prediction_dataset.py`. All are restored byte-identical with an md5 assert and nothing was committed, so this is acceptable. It should be stated as a deviation from "only `link_prediction_dataset.py` may change".

**Checked and found correct** (VERIFIED-CODE unless noted):
- `_split_validation`: partner flip, self-inverse (min, max) handling, and the raise on a missing, inconsistent or one-orientation file.
- `build_task` and `build_task_by_pair`: pair-level assignment, test rows removed in both orientations, loops in no part.
- L3: matches Kovács Eq. (1) on both sides (path x–U(D)–V(S)–y, degrees k_U = k_D, k_V = k_S).
- HeaRT "RA as coded" = CN(a, v)/deg(v): torch_sparse `mul` with a (1, N) tensor is column-wise (HeaRT `:69-75`), so the agent's finding holds.
- Faithful to HeaRT, with the carded changes:
  - `rank_score_matrix`, the `min` combination, `max_greater_zero` and the top-k path;
  - the PPR push matches `calc_ppr.py:76-108`;
  - masked candidates can never enter the top-k (combined rank min_h(max_h + 2) is above any zero-score rank);
  - the C4 pool excludes the head, so no duplicates.
- The J4.4 record's counts reproduce from `j44_build_output.txt`: fill shares 54.1/53.5 %, 32.9/32.7 %, 7.2/6.6 %, 7.1/7.5 %; minimum pools 3,810, 4,439, 1,459, 39,911.

## Checklist
| Unit | Gate | Status | Evidence |
|---|---|---|---|
| J-docs | exactly 4 files | PASS | `git show --stat 24fe8f7`: 4 files |
| J0.1 | pre-registration before code | PASS | `5106a1c` precedes `aba26c7`; file never edited later |
| J0.2 | (a)–(d) from code | NOT CHECKED in depth | `j02_reference_code.md` read only in the diff |
| J0.3 | toy, random graph, mutation | PASS | tests re-run OK; mutations carried into `j41_mutation_check.py` (7/7) |
| J0.4 | DBLP MLH 0.87 ± 0.01 | PASS | `j41_run_log.txt`: 0.872222 |
| J0.5 / J4.1 | values with convention and source | PASS (scientific caveats: findings 6, 7) | `results.md` matches `j41_run_log.txt` |
| J1.1 | inventory | PASS | `inventory.md` |
| J1.2 / J1 tests | split tests | PASS | `tests.test_lp_derived`: 34 OK (my run) |
| J1.3 | load and leak checks | PASS, but blind to reciprocal leaks (finding 3) | `j13_*`, `j42_loader_check_output.txt` |
| J1.4 | stop and list choices | PASS | `6c3f30e`, `j14_heart.md` |
| J1.5 | report, compute nothing | PASS | `j15_pubmed_labels.md` |
| J2 / J4.6 | PDF renders; numbers traced | PASS (render not re-run) | PDF has 21 page objects; every 0.xxx value in `chapter3_2026_11_09.html` traced to `results.md` or the README |
| J3 | report | PASS | `H2S-NN_partJ_report.md` |
| J4-docs | exactly 3 files | PASS | `git show --stat 3c63df0` |
| J4.0 | pre-registration before code | PASS | `32ccfbf` precedes `f96453a`; files never edited later |
| J4.1 | toy; mutations "skip fill" and "exact match" | PASS | `j41_mutation_output.txt` 7/7; tests re-run OK |
| J4.2 | carve == file; 81/9/10; reverse-only raises; mutations | PASS, but both-orientation removal not enforced when ρ is `None` (finding 3) | `tests.test_lp_valid_files` OK (my run); `j42_*` |
| J4.3 | pair-disjoint; no loops; directed-split mutation; loader checks | PASS on files and FAIL as a loader property | `j43_*`; my probe (finding 3) |
| J4.4 | J1 negative tests; training-only heuristics; no duplicates; same seed | PASS except training-only: **FAIL for acm_pa (C2)** | `j44_real_checks_output.txt`; findings 1–2 |
| J4.5 | IMDB test and adapter inventory | PASS | `tests.test_imdb_protocol` OK (my run); `j45_mutation_output.txt` 2/2 |
| J4.7 | §10; full suite | PASS | `j47_suite_output_raw.txt`: 455 tests; 20 failures, identical to J3's list (diffed by me) |
| Scope | only `link_prediction_dataset.py` changed in `src/` | PASS | diffstat (see NIT 12) |

## What I could not check
- That the numbers in DBLP's and IMDB's features are actually exploitable. I only quantified ACM.
- HeaRT's paper text (only its code was available) and Kovács et al. 2019 (I relied on the quoted Eq. (1); the formula as quoted is implemented correctly).
- H2GB `calcHomophily` and Hetero²Net internals beyond what the diff and records cite; vendor clones not opened.
- PDF visual rendering; the Amazon and YouTube behaviour of the new guard (data not on disk); the full suite, which I did not re-run (I ran only `test_lp_valid_files`, `test_heart_negatives`, `test_imdb_protocol`, `test_heterophily_metrics` (56 OK) and `test_lp_derived` (34 OK)).
- Whether PubMed's 200-d features are themselves derived from links (they feed C2).

Probe scripts, none of which touched the repository: `C:\Users\Acer\AppData\Local\Temp\claude\c7cc368768f7bbeb\2da21964-8839-4f87-8015-970c3f72d432\scratchpad\review_j\acm_leak_probe.py`, `...\reciprocal_probe.py`, `...\fb_loops_probe.py`.
