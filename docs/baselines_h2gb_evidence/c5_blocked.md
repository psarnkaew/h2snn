# C5 — class C (NSD, CSNN, H2SGNN): not started (blocked by C0 stop S1)

D1 applied per method (VERIFIED-CODE, `c0_preconditions.md`):
| Method | Node-wise on precomputed features? | D1 branch | Blocked by |
|---|---|---|---|
| H2SGNN | No: every forward mixes target-target meta-path adjacencies with learned weights and propagates K times (`H2SGNN/model.py:67-86, 90-140`) | second: H2GB's `hgt` sampler, the dataset's H2GB GCN fan-out, depth = its layers (K) | S1 (depth); also its adjacencies come from its own preprocessing of HGB files (`processing.py:125-158`) and would be rebuilt per sampled subgraph |
| NSD | No (Laplacian from the whole `edge_index` given, degree-normalised, `laplacian_builders.py:42, 70, 142-152`) | second | S1: with depth = layers, boundary nodes' degrees are incomplete (as GCN, C1 diagnostic) |
| CSNN | No (`CSNNconv.py:74-78, 90`) | second | S1 (same) |

No code and no test in this unit; the registry cells say why (`src/baselines/registry.py`, `H2GB_NC_ADAPTER_PENDING`).
