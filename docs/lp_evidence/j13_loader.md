# J1.3 Loader registration, leak checks and the seed question (2026-11-09)

## Change (the only `src/` edit of Part J)
- Three entries in `_LOCAL_HGB_LINK_DATASETS` (`src/data/link_prediction_dataset.py`):
  - `acm_pa` → `lp_derived/acm_pa`, types paper, author, subject, term;
  - `dblp_ap` → `lp_derived/dblp_ap`, types author, paper, term, venue;
  - `imdb_ma` → `lp_derived/imdb_ma`, types movie, director, actor, keyword.
- **No config preset** (`config.py` is out of scope). A config names a task with `data.task_type: link_prediction` and `data.target_relation`:
  - `[paper, rel_2, author]`, `[author, rel_0, paper]`, `[movie, rel_2, actor]`.

## Evidence (VERIFIED-RUN)
**Load smoke and leak checks:** `python ../docs/lp_evidence/j13_loader_check.py <scratch>` → `j13_loader_check_output.txt`.
- `build_link_prediction_dataset` loads each task with every existing leak check passing (`_assert_no_message_passing_leak`, `_assert_no_leakage`, `assert_no_held_out_facts`).

| Task | train / val / test positives | val / test negatives | time |
|---|---|---|---|
| acm_pa | 8,400 / 554 / 995 | 554 / 995 | 32 s |
| dblp_ap | 16,511 / 1,170 / 1,964 | 1,170 / 1,964 | 48 s |
| imdb_ma | 12,540 / 761 / 1,478 | 761 / 1,478 | 28 s |
| lastfm | 66,966 / 7,301 / 18,567 | 7,301 / 18,567 | 6 s |
| pubmed | 32,167 / 1,942 / 8,528 | 1,942 / 8,528 | 27 s |

**Unit tests** (`tests/test_lp_derived.py`, `SyntheticLoaderTests`, 4 tests):
- the test set equals `link.dat.test`;
- the split does not depend on the training seed;
- the two leak mutations are caught (the message must name a leak or an overlap).

**Affected modules:** `test_lp_derived`, `test_paths_and_facts`, `test_task_heads` → `Ran 79 tests … OK` (`j13_affected_tests.txt`).

## The seed question: the validation split does **not** depend on the training seed
- **Code (VERIFIED-CODE):**
  - The validation seed is `config.data.acm_split_seed + relation_id · 9973` (`link_prediction_dataset.py:369`).
  - Validation and test negatives use `acm_split_seed + 101/202 + rel_id` (`:926`, `:941`).
  - `acm_split_seed` defaults to 42 (`config.py:333`). No H2S-NN runner or search space sets it: `grep` over `scripts/` and `src/` finds it only in the config, the loaders and `src/baselines/bundles.py:384`.
- **Run:** loading with training seeds 3 and 11 gives identical train/val/test positives and val/test negatives on **all five tasks**, PubMed and LastFM included.
- **Conclusion:** the spec's "one split for all 10 training seeds" holds as long as `acm_split_seed` stays fixed. No fix is needed for the seed question.

## Finding 1: the loader does not use J1.2's validation split
The loader ignores `link.dat.valid`. It carves its own validation from `link.dat` (which holds J1.2's train + validation facts) by head: `_split_edge_index_hgb_by_head`, ratio `link_val_ratio` 0.09, and the first edge of every head always goes to train (`:285-325`, `:366-370`).

| Task | Loader validation | J1.2 `link.dat.valid` | In both | Loader validation, share of facts |
|---|---|---|---|---|
| acm_pa | 554 | 895 | 57 | 5.6 % |
| dblp_ap | 1,170 | 1,768 | 122 | 6.0 % |
| imdb_ma | 761 | 1,330 | 79 | 5.1 % |

The test split is J1.2's. The evaluated split is therefore about 84/6/10, not the spec's 81/9/10.

**PROPOSAL (smallest fix, not applied):**
- In `_load_local_hgb_link_data`, when `dataset_dir / "link.dat.valid"` exists, read it with `_read_local_hgb_link_file` and take its pairs as the validation facts of that relation (ρ-partner flipped, as at `:376-378`). The rest of that relation's `link.dat` pairs become train, instead of calling `_split_edge_index_hgb_by_head`. About 15 lines, plus one test.
- PubMed and LastFM are unchanged (no such file), unless the author also wants fixed validation files for them.

## Finding 2: a test fact present only as its reverse-relation row is not caught by the loader
This is route 2 of the pre-registration, confirmed. On a copy of `acm_pa`:

| Mutation | Result |
|---|---|
| Both rows (as HGB stores a fact) | **caught**: "Held-out edges leak into the message-passing graph" |
| The r row alone | **caught**: same message |
| The r′ row alone | **not caught** |

- **Why:** ρ is exact set equality (`fact_layer.py:90-122`). One extra reverse row breaks ρ(rel_2) = rel_3; both become `None` (logged ρ). `assert_no_held_out_facts` checks (v, ρ(r), u) only when ρ(r) is defined (`fact_layer.py:242-245`). `_assert_no_message_passing_leak` checks only the target relation id (`:950-954`).
- **Further consequence (DERIVED):** with ρ broken, `rel_3` is split independently of `rel_2` (`:357-378`), so the reverses of `rel_2`'s validation facts also stay in the message graph.
- **Guard in place:** the J1.2 file-level check `test_fact_leaks` catches it (`= 1`). The test `test_a_reverse_only_leak_is_caught_at_file_level` covers it, and the three real tasks are clean.
- **PROPOSAL (not applied; loader logic is out of scope):** when a relation with the reversed endpoint types exists but ρ(r) is `None` for the target relation, the loader should raise. Alternatively, `assert_no_held_out_facts` could check (v, ·, u) against every relation with reversed endpoint types.
