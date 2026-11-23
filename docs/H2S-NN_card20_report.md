# Card 20 report — H2G-former: MetaPath2Vec channel in the B1a validation loader

Card `docs/H2S-NN_card20_card.md` (design-docs c22d0d4), base `origin/v3b` `0b9873d`, branch `card20`. Work commit
`6726f57`.

## Step 1: which way, and the diff

**The fallback.** The call order is kept, and the embedding is copied onto the subset view's node stores.
- **Why not the preferred order:** moving `configure_from_yaml` earlier would change the cfg the model sees. `hgt_loader` calls H2GB's `set_cfg(cfg)` before and after it builds a loader (`src/baselines/minibatch.py`, `hgt_loader`). That resets the cfg to H2GB's defaults.
- **So,** if the yaml cfg were configured before the four loaders, it would be wiped before `build_gtmodel`.

**The diff** is in `src/baselines/adapters/_h2gformer_h2gb.py` only; `minibatch.py` is unchanged.
- **New function `share_metapath_with_subset_view(subset_loader, data)`.** It takes the view the subset loader samples from (`subset_loader.loader.data`: H2GB's `LoaderWrapper` → PyG `HGTLoader`). Unless the view is `data` itself, it sets the view's `pestat_Hetero_Metapath` to the same tensor as `data`'s, for every node type.
- **In `fit`:** one call, right after `attach_h2gb_metapath`, inside the same `if METAPATH in stack` branch.
- The function is added to `__all__`.

## Step 2: the training path

No training input changes.
- **Loaders:** the train, val and test loaders, and the subset ids, are built exactly as before.
- **RNG:** the new line draws no random numbers. `torch.manual_seed(seed)` still comes right before `build_gtmodel`.
- **Model and cfg:** unchanged, because the cfg order is unchanged.
- **What the change touches:** the view's node stores, which only the B1a subset loader reads.

## Tests (job 18602, CPU, `--mem 16G --time 1:00:00`)

**The test.** `tests/test_card20_h2gformer_subset_metapath.py` uses card 17's cache fixture: ogbn-mag on a MAG-typed synthetic graph, with H2GB's `preprocess_Metapath` stubbed to write a known embedding.
- The real `fit` runs with a plan of one training step and one epoch, so the B1a subset is evaluated once.
- It asserts that a subset batch carries `pestat_Hetero_Metapath` on every node type a training batch carries it on.
- The base run used a second checkout at `0b9873d` (detached worktree `/mnt/data/peeradon/h2snn-card20-base`), with the test file copied in for the run and removed after.

| code | result |
|---|---|
| fix (card20, `src` = this diff) | OK (1 test, 51 s) |
| base `0b9873d` | ERROR: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x8 and 16x16)`; the missing 8 columns are `dim_pe` = `metapath_dim_pe(16)` |

**Training identity:** first training-step loss base = 1.46321439743042, fix = 1.46321439743042, equal = yes.

Log: `docs/card20_evidence/slurm-18602-test.log`. The memory peak was not recorded: sacct could not reach the accounting database.

## Step 3: the other `val_subset_loader` callers (read only)

| caller | writes a node / edge attribute to `data` after the call that its model reads |
|---|---|
| `src/baselines/adapters/h2sgnn.py:493` | no (the model is built before the call; then the training loop) |
| `src/baselines/adapters/hetsheaf.py:786` | no (straight into the training loop) |
| `src/baselines/adapters/hgb_han.py:454` | no (`configure_from_yaml` reads `data` only, then `build_hetero_gnn`) |
| `src/baselines/adapters/rgcn.py:282` | no (`configure_from_yaml` reads `data` only, then `build_h2gb_network`) |
| `src/baselines/h2gb_published.py:304` | no (`resolve_published` → `configure_from_yaml`, read only) |
| `src/baselines/minibatch.py:555` | no (straight into the training loop) |

None has the defect. Nothing else was changed.

branch=card20 sha=6726f579ee79c4e0b50bdd5fb46269625b6b6580 card=docs/H2S-NN_card20_card.md base=0b9873d0c19455c8df56fa152492188d8918b62d
