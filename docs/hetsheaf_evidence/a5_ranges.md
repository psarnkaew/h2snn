# Card 3 A5.2: HetSheaf fitting ranges on a 24 GB GPU (OOM ladder; no metric)

Ladder (decision 1, as card 2 D6.3): from A4's point (`sheaf` general/attention_type_ensemble, `d` 5, `layers` 8,
`hidden_channels` 64, dropout / input_dropout 0.5, lr 1e-3, weight_decay 1e-4, LP decoder distmult) one grid step at a
time, axis order `hidden_channels` 32 -> 16 -> 8, `d` 5 -> 4 -> 3 -> 2, `layers` 8 -> ... -> 2; stop at the first point that
fits (one epoch + validation, status OK). `hidden_channels` 64 and 32 are A4's records (`a4_cells/`, OOM), so the ladder
starts at 16. Script `a5_dryrun.py --ladder`, job script `a5_job.sh`, records `a5_ladder/`, attempt table
`a5_ladder_table.md` (from `a5_table.py`). All runs at `1867681`, RTX 3090 on g3n01 (24 GB, one job per GPU).

Full range of the stepped axes: `hidden_channels` {8, 16, 32, 64}, `d` {2, 3, 4, 5}, `layers` {2, ..., 8}.
"Fitting range" = the largest fitting point and every point no larger on any stepped axis (the other axes at A4's
point); assumes memory does not increase when an axis decreases (not measured point by point).

| Cell | Axes stepped | Largest fitting point | Fitting range | Source |
|---|---|---|---|---|
| NC acm | hidden_channels | h 8, d 5, layers 8 | hidden_channels <= 8; d, layers full | 17118 |
| NC dblp | hidden_channels | h 32, d 5, layers 8 | hidden_channels <= 32; d, layers full | A4 16943 |
| NC freebase | hidden_channels | h 8, d 5, layers 8 | hidden_channels <= 8; d, layers full | 17119 |
| LP acm_pa | hidden_channels | h 8, d 5, layers 8 | hidden_channels <= 8; d, layers full | 17120 |
| LP dblp_ap | hidden_channels | h 32, d 5, layers 8 | hidden_channels <= 32; d, layers full | A4 16974 |
| LP freebase_bb | hidden_channels | h 8, d 5, layers 8 | hidden_channels <= 8; d, layers full | 17121 |
| LP pubmed | hidden_channels | h 16, d 5, layers 8 | hidden_channels <= 16; d, layers full | 17122 |
| LP ogbn-mag P-P | hidden_channels, d, layers | none | **N/A for HetSheaf**: OOM on a 24 GB GPU at the smallest grid point (`hidden_channels` 8, `d` 2, `layers` 2) under the shared large-LP sampler (author A5-1a, 2026-11-17; range rule 1 kept at 24 GB although tuning runs on TSUBAME) | 17123 |
| LP H-Pokec U-U | hidden_channels | not established (h 16, d 5, layers 8: vendor finite assertion, not OOM) | **hidden_channels <= 16; d, layers full**; a non-finite run is a failed trial (author A5-2a, 2026-11-17); central-point status: A6.1 | 17124 |

Stop items (reported, not fixed):
- **ogbn-mag P-P, OOM at the ladder's bottom.** Sites `sheaf_models.py:1236-1238` (the sheaf learner's per-edge input:
  `x_row`, `x_col` = `index_select` of the node features by edge endpoints, `h_cat` = their concatenation, width
  2 x `d` x `hidden_channels`). At h 8, d 2, layers 2: 15.81 GiB allocated, tried to allocate 9.65 GiB more (23.68 GiB
  GPU). The same 15.81 + 9.65 GiB at every `layers` value 8..2. Request / (bytes per edge of the failing tensor) gives
  about 5.7e7-8.1e7 edges per sampled step subgraph at every point (DERIVED, not measured); the design session's 3.4 GiB
  estimate for d 2, h 8 does not hold on this sampled subgraph.
- **H-Pokec U-U, `AssertionError` at `transductive/disc_models.py:571`** (their `assert torch.all(torch.isfinite(x))`,
  commented "To detect the numerical instabilities of SVD") after 4,070 s at h 16, d 5, layers 8, peak GPU 12.13 GiB;
  not an OOM, so the ladder did not step past it (one run, seed 0; not rerun). Whether the cell fits at h 16 is not
  established.

## A6 update (2026-11-17, author decisions A5-1a, A5-2a, A5-3; card 3 A6)
- ogbn-mag P-P: **N/A for HetSheaf** (reason in the table; A5-1a). The A5 stop item is closed by this decision.
- H-Pokec U-U: range `hidden_channels` <= 16 (`d`, `layers` full); a non-finite run is a failed trial (A5-2a).
  Central-point status run (A6.1): `a6_pokec_uu_central.txt`.
- Categoricals at the adapter's default (A5.3) stay as recorded (A5-3).
