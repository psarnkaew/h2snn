# Task card — Baselines on H2GB, part 2: the remaining adapters after the author's decisions (card 1b)

**Design authority:** `docs/H2S-NN_baselines_h2gb_card.md` (part 1: scheme table, rules, Slim mode) as amended here; `docs/H2S-NN_baselines_h2gb_report.md` (part 1 report, `6ba6dec`) and `docs/baselines_h2gb_evidence/`; decisions log rows "Baselines on H2GB: S1 sampler depth", "Baselines on H2GB review" and "Baselines on H2GB: S3 PSHGCN, inputs B" (2026-11-16). `docs/working_preferences.md` binds (incl. **Slim preparation** and **follow the source's code exactly**). `H2S-NN/AGENTS.md`/`CLAUDE.md` bind.
**Branch:** `baselines-h2gb` (now equal to `v3b` `6ba6dec`), then `git merge --no-ff` of the pinned `design-docs` SHA. **Report:** `docs/H2S-NN_baselines_h2gb_part2_report.md`; evidence `docs/baselines_h2gb_evidence/` (prefix `p2_`).
**Runs in parallel** with `dry-run` (card 2). **Files this card may change:** `src/baselines/**`, `configs/baselines/**`, tests of baselines (`tests/test_baselines*.py`, `tests/test_partP_h2gformer.py`), the evidence folder, the report. Anything else (H2S-NN `src/` outside `src/baselines/`, `src/data/`, `scripts/`, `configs/experiments/`) is not edited: if a fix needs it, it is a stop item listed for the design session.

## Author decisions (2026-11-16) that amend part 1
| # | Decision |
|---|---|
| S1 | Class A and class C sample with **H2GB's YAML fan-out lists unchanged** (their length is the depth, independent of the tuned layers; H-Pokec per-type lists as H2GB). Where a tuned layer count exceeds H2GB's depth − 1 the sampled model is as H2GB runs it, not full-batch-exact; stated |
| S5 | HAN on sampled subgraphs exactly as H2GB runs it (semantic attention averaged over the sampled subgraph); stated. No equivalence test for HAN |
| S2 | HINormer on H2GB = **H2GB's own HINormer** (`vendor/H2GB/H2GB/network/hinormer_model.py`, H2GB's sampler for it in `sampler/custom_sampler.py`, H2GB's configs `configs/<dataset>/<dataset>-HINormer.yaml`); class A. A dataset without an H2GB HINormer config uses H2GB's config rule of the nearest dataset family only if H2GB's code defines one; otherwise report it (stop item for that dataset only) |
| S3 | **PSHGCN (b):** their ogbn-mag model (`vendor/PSHGCN/ogbn-mag/model.py:57-66`) hard-codes MAG's input widths (type `P` 128, other types 256, `lin_residue` 128). The adapter builds the same model with the input width of each node type and the target type taken from the dataset; propagation, coefficients, label part, loss and optimiser unchanged. Runs on every H2GB NC dataset (H-Pokec: K = 1, hidden ≤ 64 as decided) |
| S4 | H2SGNN follows S1 (D1 second branch: H2GB's sampler) |
| C4-a / inputs B | Each class-B method takes **its source's own inputs where its source defines them**, H2GB's features otherwise; stated. Hetero²Net on ogbn-mag/mag-year: PyG MetaPath2Vec embeddings (`mag_metapath2vec_emb.zip`, path in the part 1 report; author downloads). SeHGNN on ogbn-mag/mag-year: the embeddings its published MAG run uses (read `SeHGNN/ogbn/` README and code): if a file can be downloaded, write URL, file name, sha256 rule and target path (author downloads; the cell waits, not a stop); **if the embeddings must be trained** (e.g. by another repository's script), do not train them: record it and leave the two cells "pending author (embeddings need training)" |
| C4-b | SeHGNN on OAG and PDNS merges parallel relations between the same type pair (as in part 1) |
| C4-c | ChebNetII on datasets without one feature width (IEEE-CIS, H-Pokec, PDNS): per-type input projection as its HGB adapter (`TypeProjectedInput`). Because the projection is linear, the Chebyshev bases are precomputed per type block (`T_k(L) X_t`) and the projection is applied after; this equals projecting first (test) |

## Units (in order; Slim tests only; a unit that stops is reported and the next independent unit starts)

### E0 — Preconditions (no code)
Branch check and merge. Re-read part 1's scheme table rows for every method below against the decisions above; record per method: class, sampler/batching code (file:line), layer parameter, what changes. Read H2GB's HINormer code and configs; read SeHGNN's MAG input source (decision C4-a). A file:line that contradicts a decision above is a stop.

### E1 — Class A: GCN, GAT, HAN, H2G-former, HINormer (NC)
Each trains through `src/baselines/minibatch.py` with H2GB's fan-out lists (S1). H2G-former as part 1's C3 (D-P3, D-P6, D-P9). HINormer per S2. Move the GCN full-fan-out equivalence from the evidence folder into `tests/`, at **depth = propagation steps + 1** (float64, `atol 1e-10`, negative control fan-out 1); the HAN part of that module is removed (S5).
**Tests (Slim):** per adapter one step finite and moves; H2G-former label-route test **and its mutation**; GCN equivalence as above.

### E2 — Class B: PSHGCN, SeHGNN (remaining cells), ChebNetII (remaining cells), Hetero²Net (NC)
PSHGCN per S3 with its own ogbn-mag batching and label-propagation handling as part 1 C4 states. SeHGNN on ogbn-mag/mag-year per C4-a, OAG/PDNS per C4-b. ChebNetII on IEEE-CIS, H-Pokec, PDNS per C4-c. Hetero²Net with its own `NeighborLoader` (part 1 scheme table); its MAG cells read the downloaded file (sha256 recorded at first use, a different file refused) and wait while it is absent.
**Tests (Slim):** per adapter one step finite and moves; **PSHGCN equivalence:** with MAG's widths (128/256) the adapter's model and their `PSHGCN` class, same state dict, give the same output on a synthetic MAG-typed graph (float64, exact); **ChebNetII:** per-block precomputation then projection equals projection then propagation (float64, `atol 1e-10`).

### E3 — Class C: NSD, CSNN, H2SGNN (NC)
D1 second branch with S1's lists; their homogeneous conversion as on HGB. **Tests (Slim):** per adapter one step finite and moves.

### E4 — Large LP wiring
Every method whose source has link prediction: class A/C on `src/baselines/large_lp_minibatch.py` (message graph, S1's lists); decoupled methods (PSHGCN; H2SGNN only if node-wise) precompute per node on the message graph, then batch links. Decoder, negatives, budgets as the local LP tasks and spec §6. Methods without LP in their source keep their reason. **Tests (Slim):** per adapter one LP step finite and moves; the held-out-fact test of part 1 covers every wired adapter's loader (parametrised), with its mutation.

### E5 — Suite and report
Registry: every H2GB NC and large-LP baseline cell is runnable, waiting for a named file, or N/A with a reason. Full suite once on one node; golden and mb0 compares only if `src/` outside `src/baselines/` changed (it should not). Report short: cell table (method × dataset: runnable / waiting / N/A + reason), scheme table deltas, tests and commits, deviations, decisions for the author. Last line: `branch=baselines-h2gb sha=<full head sha> card=docs/H2S-NN_baselines_h2gb_part2_card.md base=<pinned v3b sha>`.

## Rules and stop conditions
As part 1 (data, cluster, git, forbidden list), plus the file restriction above. **Stop (the item, report, continue):** a decision above contradicts the vendor code; a fix needs a file outside this card's list; the PSHGCN or ChebNetII equivalence fails; a held-out fact reaches a sampled subgraph; any existing (HGB, local LP, GC, full-batch) output changes. **Not stop conditions:** a missing author-downloaded file; embeddings that need training (cell pending); memory/time; any loss value. No training run on real data, no metric (the dry run of card 2 covers real data).
