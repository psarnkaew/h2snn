# Card 3 Part B — stated deviations (B1)

Vendor pins as in `b0_preconditions.md` (SeHGNN `e92bd37`, PSHGCN `f87406c`, H2SGNN `93c33fa`).

## D1 — generic HGB branch completes missing reverses (author decision 2026-11-18)

**What.** `plan_relations` (`H2S-NN/src/baselines/adapters/_hgb_generic.py`) adds the transpose of every stored
relation whose reverse is not stored:
- a cross-type relation `s -> d` gets a reverse key `d -> s`;
- a same-type key is merged with its transpose (A + Aᵀ), then gets `set_diag`, as ACM's `PP` (PSHGCN
  `hgb/processing.py:61-67`).

**Where it applies.** Freebase only. All 36 HGB relations are stored in one direction, and the vendor code has no
Freebase branch for PSHGCN or H2SGNN.

**Where it is a no-op.** Where every reverse is stored: ACM, IMDB, AMiner and H2GB's `ToUndirected` graphs.

**Precedent in the sources:**
- SeHGNN's own Freebase branch: `hgb/utils.py:450` symmetrises the same-type key (`adjs['00'].to_symmetric()`), and
  `:461-469` adds `adj.t()` under the reversed name for every cross-type key.
- PSHGCN's `hgb-lp` processing: `hgb-lp/scripts/data_loader.py:44` calls `gen_transpose_links`, and `:514-517` stores
  the transpose of every relation. Its `hgb` NC processing has no Freebase branch.
- The prompt cited `hgb/utils.py:446,459-467`. At the pinned SeHGNN, line 446 is a diagnostic `print`. The lines
  above are the ones checked on 2026-11-18.

**Tests** (`H2S-NN/tests/test_freebase_baselines.py`):
- The ACM-branch equivalence is kept for PSHGCN and H2SGNN (`EquivalenceTests`).
- `RelationPlanTests.test_stored_reverses_leave_the_plan_unchanged` checks the no-op case.
- `RelationPlanTests.test_one_direction_toy_graph_gets_both_directions_and_a_symmetric_same_type_key` uses 2
  relations stored one way. The operators must hold both directions and a symmetric same-type key with self-loops.
- Mutation: with the transpose dropped, the toy test fails; with the real function, it passes
  (`b1_reverse_mutation.py`, `.txt`).

## D2 — sparse identity instead of the dense `torch.eye` for featureless types

SeHGNN `hgb/utils.py:241-250` and PSHGCN `hgb/processing.py:13-21` build a dense identity for each featureless type.
On Freebase this is about 37 GB. SeHGNN's Freebase branch never reads it, so `_freebase_inputs` skips it. The
PSHGCN and H2SGNN generic branches use `sparse_identity` instead. Values are the same; only the storage format
changes.

## D3 — PSHGCN on the 6 non-MAG H2GB NC datasets and on large LP

The vendor code has an ogbn-mag decoupled scheme only. It is made generic as follows:
- Target letter `P`.
- `expander` over the dataset's keys, with K = 2.
- No label-propagation part.
- On NC, training goes through the shared `train_index_minibatch`. On large LP, it goes through the shared
  `fit_large_lp`, whose optimiser is the shared single AdamW.

The H-Pokec override from the HGB search space does not apply to the decoupled search space.

## D4 — SeHGNN Freebase published lr / weight decay (3e-5) are outside the shared grids

There is no `configs/baselines/sehgnn/node_classification/freebase.yaml`: the published-value reachability test
would fail. The value stays in the report as an item for the author.
