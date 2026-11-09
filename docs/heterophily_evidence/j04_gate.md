# J0.4 validation gate: DBLP MLH (2026-11-09)

**Pre-registered** (`preregistration.md` §2): Hetero²Net's DBLP MLH, homophily form, meta-path APA, reproduced within ±0.01 of 0.87, i.e. in [0.86, 0.88].

**Command** (from `H2S-NN/`, `h2snn` environment, CPU):
`python scripts/heterophily/compute_heterophily.py --datasets dblp --out-json ../docs/heterophily_evidence/j04_dblp_gate.json`
Output: `j04_gate_run.txt` and `j04_dblp_gate.json`.

**Result (VERIFIED-RUN): PASS on the first attempt.**
- The H2GB meta-path set and the Hetero²Net set are both {author → paper → author}.
- `G_P` has 11,113 ordered pairs, including 4,057 self-loops (one per author; every author has a paper).
- **`h_edge` = 0.872222 (MLH, homophily form). Published: 0.87. Inside [0.86, 0.88].**
- The value comes from H2GB `calcHomophily`. It was cross-checked against our own float64 code and Hetero²Net's `label_homophily` (tolerance 1e-6, both agree).
- Also reported, not gated: `h_node` 0.929465, `h_adj` 0.826764. MLH heterophily form = 0.127778; ℋ² = 0.173236.

**The gate also discriminates the self-loop convention** (DERIVED from the run's counts):
- 0.872222 × 11,113 = 9,693 pairs agree, and the 4,057 self-loops always agree.
- Without self-loops: (9,693 − 4,057) / (11,113 − 4,057) = 5,636 / 7,056 = **0.799**, which is **outside** the gate.
- So the published 0.87 corresponds to `G_P` with self-loops kept, which is how H2GB builds it (`calcHomophily.py:146-150`). Hetero²Net's own `G_P` builder is not in its repository (J0.2), so this agreement is the evidence for its convention.

**Dataset version check:** PyG `HGBDataset` DBLP. The link file has 239,566 rows and the node file 26,128 nodes; both equal Hetero²Net Table II (26,128 nodes, 239,566 edges).

The J0.5 numbers may now be computed (card: "only values that passed J0.4").
