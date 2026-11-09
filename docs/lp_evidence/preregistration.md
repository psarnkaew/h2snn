# J1 pre-registration: derived link-prediction tasks (J1.2, J1.3)
Written 2026-11-09 on `part-j` after J1.1 (`1588f58`), before any J1 code. J1.4 (hard negatives) gets its own addendum after HeaRT has been read.

## Changes
**J1.2: new files only.**
- `H2S-NN/scripts/lp/build_derived_lp.py`.
- `H2S-NN/tests/test_lp_derived.py`.
- The outputs, written to `H2S-NN/data/lp_derived/{acm_pa, dblp_ap, imdb_ma}/`. This folder is gitignored (`git check-ignore -v`: `H2S-NN/.gitignore:47:/data/`).

**J1.3: the only `src/` change.**
- Three entries in `_LOCAL_HGB_LINK_DATASETS` (`src/data/link_prediction_dataset.py:28`).
- Its consumers: `link_prediction_dataset.py:330`, `:767`, `:795`, plus a comment at `config.py:266`.
- No config preset is added (`config.py` is out of scope). A derived task is named with `data.task_type: link_prediction` and an explicit `data.target_relation` (`config.py:483-521`, `:680-689`).

## Split design (J1.2)
**Target relation r, in its stored orientation, and its exact reverse r′ = ρ(r)** (from J1.1):

| Task | r | r′ |
|---|---|---|
| ACM P–A | 2 (paper→author), 9,949 facts | 3 |
| DBLP A–P | 0 (author→paper), 19,645 facts | 3 |
| IMDB M–A | 2 (movie→actor), 14,779 facts | 3 |

**Procedure.**
- The facts are the distinct `(u, v)` pairs of r, sorted.
- They are permuted with `numpy.random.default_rng(seed).permutation`, **seed 42**, recorded together with the numpy version.
- Counts: `n_test = round(0.10 n)`, `n_val = round(0.09 n)`, train = the rest.
  - ACM: 995 / 895 / 8,059.
  - DBLP: 1,964 / 1,768 / 15,913.
  - IMDB: 1,478 / 1,330 / 11,971.
- The written files are the fixed split; their md5 values are recorded in `split_meta.json`.

**Files per task.**
- `node.dat` and `info.dat`: byte copies of the HGB node-classification release (ids and features unchanged).
- `link.dat`: every row of the release's `link.dat` in its original order and bytes, **except** the rows of r whose pair is a test fact and the rows of r′ whose reversed pair is a test fact.
- `link.dat.test`: the release's r rows of the test facts (one orientation, r).
- `link.dat.valid`: the release's r rows of the validation facts.
  - These facts **stay in `link.dat`**, because the HGB format keeps train and validation together in `link.dat`, and the card says "link.dat holds everything else".
  - The current loader does not read this file. It carves its own validation from `link.dat` (`:366-370`). J1.3 reports that.

## Tests and expected results (`tests/test_lp_derived.py`, unittest)
**Split, on a synthetic HGB-format graph and, if built, on each real task.**
- The three parts are disjoint by fact in either orientation.
- Proportions 81/9/10, each within ±0.5 percentage points.
- The same seed gives byte-identical files.
- Every node of a test fact exists in `node.dat`.
- No test fact is left in `link.dat` in either orientation (r or r′).

**Loader (J1.3).**
- `build_link_prediction_dataset` runs its existing leak checks (`_assert_no_leakage`, `_assert_no_message_passing_leak`, `assert_no_held_out_facts`) without error on every derived task.
- **Mutation 1:** put one test fact back into `link.dat` as an r row. Expected: caught by the loader (train/test overlap).

**Expected:** all pass, and mutation 1 is caught.

## Results that would show the work wrong
- An overlap between parts, a proportion outside its band, a non-deterministic file, or a test node missing from `node.dat`.
- A leak check that fails on a clean derived task, or mutation 1 not caught.
- A derived `link.dat` that differs from the release in anything other than the removed test rows (checked by counting rows: release rows − 2 × n_test).

## Rule 6(b): routes by which a held-out fact can reach training (this unit's analogue of the information-route audit)
1. **An r row in `link.dat`.** Covered by the split test and by the loader (mutation 1).
2. **An r′ row in `link.dat`, i.e. the reverse orientation.** Covered by the split test.
   - Expected blind spot of the loader, to be checked and reported, not fixed: ρ is exact set equality (`fact_layer.py:90-122`). A single reverse row breaks ρ(r) = r′, the two relations are then not tied, and `assert_no_held_out_facts` checks (v, ρ(r), u) only when ρ(r) is defined (`:242-245`).
   - **Mutation 2:** put back only the r′ row of one test fact. Expected: **not** caught by the loader, but caught by the split test.
3. **A path cell through such a row.** The loader checks every path group (`:752-761`), which reduces this route to routes 1 and 2.
4. **Negatives or heuristics computed on a graph containing test facts** (J1.4). Its own test and mutation are pre-registered in the J1.4 addendum.

Reachability (rule 6(a)) does not apply: nothing is trained.
