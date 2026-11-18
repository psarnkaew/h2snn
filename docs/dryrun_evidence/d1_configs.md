# Card 2 D1 — missing configurations

17 new `H2S-NN/configs/experiments/<dataset>_h2snn_v3b.yaml`, written by `d1_write_configs.py` from the existing files
(same keys; only the dataset's data keys change): NC freebase and the 8 H2GB NC datasets (from `dblp_h2snn_v3b.yaml`);
LP dblp_ap, imdb_ma, freebase_bb, pubmed, lastfm (from `acm_pa_h2snn_v3b.yaml`, target relation = the dataset's preset);
GC proteins, enzymes, nci1 (from `mutag_h2snn_v3b.yaml`). The two large LP tasks are out of D1-D4 (no entry point).

- `r` and `l_max` (design-session review): named explicitly, r = 8, l_max = 4, inside the tuned ranges {8,16,32,64} /
  {2,3,4,5} (as acm_pa / mutag already do); no calibration table, no `WISP_WALKS_PER_ANCHOR` entry. Validation did not
  force r = 0-or-calibrated: an explicit r resolves (no stop item).
- H2GB NC `data.batch_size: 32` = H2GB's own value in its GNN configs (`vendor/H2GB/configs/*/*-GCN.yaml`,
  `train.batch_size 32`). **Finding (decision for the author):** the budget table tunes B in {32, 256, 1024} on H2GB NC
  (`MINIBATCH_BATCH_SIZES`, `config.py:91, 631`; the SeHGNN and ChebNetII adapters tune it), but H2S-NN's sweep has no
  `batch_size` axis (`scripts/run_h2snn_wandb_tuning.py:323-344`, `KNOWN_PARAMETER_NAMES`), so the YAML's B is the B of
  every H2S-NN trial. Not changed here (a search-space change).
- Existing YAMLs unchanged: acm (r 42), dblp (r 65), imdb (r unset = its calibrated entry) carry the withdrawn E11 r,
  outside {8,16,32,64}; the sweep and the dry run override r.

Test `tests/test_experiment_configs.py` (Slim, text only): each of the 22 H2S-NN cells resolves through
`TrainConfig.from_dict` to its task, path source wisp, r >= 1 (new files: r and l_max in the tuned ranges), H2GB NC in
mini-batch with B in the range, and passes `reject_never_tuned_settings`. Job 16795 step 7 (g2tn01): 2 tests OK
(`d1_tests.txt`, with `test_notebook_configs`).
