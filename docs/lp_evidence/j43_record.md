# J4.3 Freebase link prediction `freebase_bb` (spec v1.8 F10): record (2026-11-09)
Pre-registration: `j4_preregistration.md` §J4.3 (`lp_evidence`) and the task-level `H_edge` section (`heterophily_evidence`).

## Build (VERIFIED-RUN, `j43_build_output.txt`; `data/lp_derived/freebase_bb/split_meta.json`)
**Builder:** `scripts/lp/build_derived_lp.py` → `build_task_by_pair` (task `freebase_bb`: relation 0, unit `unordered_pair`, seed 42).
- It refuses a relation that has a stored reverse relation; such a relation is split by fact instead.

| | Unordered pairs | Stored rows |
|---|---|---|
| train | 103,303 (81.00 %) | 152,021 |
| valid | 11,478 (9.00 %) | 16,894 |
| test | 12,754 (10.00 %) | 18,723 |
| self-loops (in `link.dat`, in no part) | — | 15,036 |

- The pair counts equal the pre-registered ones.
- `link.dat` = the release minus the 18,723 test rows (both orientations of every test pair).
- `link.dat.valid` lists the validation pairs' stored rows, which stay in `link.dat` (HGB format).
- **output md5:** `link.dat` `c68908fb…`, `link.dat.test` `3fc5363d…`, `link.dat.valid` `b420348c…`. Node and info files are byte copies of the release (md5 equal to the source).

## Loader (VERIFIED-CODE `src/data/link_prediction_dataset.py`)
- `freebase_bb` is registered with 8 node types from `info.dat` (book … business).
- **`exclude_self_loop_positives`:** training positives drop self-loops, which stay in the message graph as stored facts.
  - They also remain in the known-positive set, so no self-loop can be a negative.
- ρ(rel_0) = `None`, and no other book→book relation exists, so J4.2's guard does not apply. The split by pair is what prevents the reciprocal leak.

## Checks (VERIFIED-RUN)
**Real load** (`j43_loader_check.py` → `j43_loader_check_output.txt`), with every existing leak check passing:
- train / val / test = 152,021 / 16,894 / 18,723 rows = 103,303 / 11,478 / 12,754 pairs, with 0 self-loops among positives;
- 0 pairs shared between parts;
- 0 held-out rows in the message graph, as stored or reversed;
- all 15,036 self-loops are in the message graph;
- message graph 834,089 edges = 869,706 tied facts (J1.1) − 35,617 held-out rows.

**Tests:** `tests.test_lp_derived` with `TaskLevelHEdgeTests` → **35 OK**.
- 7 new synthetic Freebase-like tests:
  - pairs disjoint;
  - reciprocal leak 0;
  - every stored row of a held-out pair is in its file;
  - self-loops in no part;
  - 97/11/12 by pair;
  - byte-identical rebuild;
  - loader leak checks with no self-loop positives.
- 3 real-task tests (`RealFreebasePairTests`).

**Mutations** (`j43_mutation_check.py` → `j43_mutation_output.txt`), **2/2 caught**:
- "split by directed fact" (the card's mutation), caught by the reciprocal-leak test;
- "self-loops kept among training positives".

## Task-level `H_edge` book–book (`../heterophily_evidence/freebase_bb_task_h_edge.json`, `j43_task_h_edge_log.txt`)
- **Graph:** relation 0 symmetrised as H2GB's loaders (`to_undirected`, self-loops kept), 270,106 edges.
- **Value:** Hetero²Net `label_homophily` on `calcHomophily`-filled labels, cross-checked with own code to 1e-6. Hand test: `TaskLevelHEdgeTests`.

| Convention | `H_edge` (heterophily) |
|---|---|
| **Fill (primary), seed 0** | **0.7794** (seeds 0–4: 0.7794–0.7829) |
| Labelled-only diagnostic (15,707 edges) | 0.3229 |

**Against the pre-registration:**
- The labelled-only value is lower, as expected.
- The filled value is below the rough 1 − 1/7 ≈ 0.86, which was explicitly not a sharp prediction. Diagnosis: the 15,036 self-loops (5.6 % of edges) always agree, and labelled–labelled edges agree 68 % of the time. Only edges touching a filled book agree at about chance.
