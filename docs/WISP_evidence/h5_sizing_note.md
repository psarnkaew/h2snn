# H5 sizing decisions (run-size rule of the 2026-11-08 correction)
Written 2026-11-08 23:32:24, from timing only, before the S2 probe's result was available.

- S1: n = 1000 targets as pre-registered; runs took 19-51 s. No change.
- S1': n = 500 targets as pre-registered (8 authors, 8 cpapers each, 4,000 decoys); the timed
  full/seed-42 run took 59.2 s. No change.
- S2: the probe at n = 600 (full, seed 42) logged about 0.7 s per epoch, about 80 s per run, over
  the ~1 minute budget. Decision: n = 400 targets for every S2 run (per-target structure, r, ℓ_max
  and criteria unchanged). The n = 600 probe finishes and is reported as a probe
  (key s2_probe_n600), not as one of the 3 seeds.

## Outcome (added after the runs)
- The S2 probe took 125.1 s at n = 600; the three n = 400 runs took 158.2 / 139.2 / 141.5 s.
  Reducing n did not reduce the time: the cost is per-epoch overhead (two evaluation forwards,
  checkpoint writes, the logged diagnostics every 10 epochs), not graph size. The ~1 minute
  budget was therefore missed for S2; no further resizing was done.
- S1: 19.5-51.4 s per run. S1': 54.7-64.4 s per run.
