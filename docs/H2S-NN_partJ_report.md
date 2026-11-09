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
