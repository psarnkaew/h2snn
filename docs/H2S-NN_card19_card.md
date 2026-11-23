# Card 19 — H2G-former: MetaPath2Vec channel missing from the B1a validation loader (H2GB mini-batch)

Written 2026-11-23 by the model design session on the ops report of the same day (wave 1, job 9523125, code `0b9873d`). Report: `docs/H2S-NN_card19_report.md`.

## Why (VERIFIED-CODE, `0b9873d`)
- H2G-former seed 0 on mag-year, oag-chem, ogbn-mag and PDNS crashed at the first validation (end of epoch 5) with `mat1 and mat2 shapes cannot be multiplied` in `pre_gt_dict` (`vendor/H2GB/H2GB/network/gt_model.py:201`). Training batches pass.
- Cause, in `src/baselines/adapters/_h2gformer_h2gb.py` `fit`:
  - line 194 builds the B1a subset loader. `val_subset_loader` (`src/baselines/minibatch.py:224-241`) makes `view = copy(data)`, a shallow copy that has **its own node stores**;
  - line 208, later, `attach_h2gb_metapath` writes `pestat_Hetero_Metapath` into `data`'s node stores only;
  - so batches from the subset loader have no MetaPath2Vec channel. The train, val and test loaders use `data` itself and see it.
- The width missing in each crash equals `dim_pe` exactly (`metapath_dim_pe`, `_h2gformer_encoders.py:168`): PDNS 32 (64 − 32), oag-chem 128 (256 − 128), mag-year and ogbn-mag 128 (512 − 384).
- Not affected: stacks without MetaPath2Vec (IEEE-CIS-G, OAG-ENG, OAG-CS: `Hetero_Raw+Hetero_Label`), which finished. The label channel is safe on the view: `prepare_label_channel` keeps the same training mask, and H2GB's label encoder on a "val" batch reads only `train_mask` (`hetero_label_encoder.py:43-50`), so validation labels never enter.
- Scope: every H2G-former cell with `Hetero_Metapath` on the H2GB mini-batch path, published and tuned.

## Fix (eval input only; training unchanged)
1. In `_h2gformer_h2gb.fit`, make the B1a subset loader see the same node features as the other loaders. Preferred: attach the MetaPath2Vec embedding to `data` **before** `val_subset_loader` is called. That means H2GB's cfg is configured first, as `attach_h2gb_metapath` requires.
   - The split loaders and the subset ids must still be built from the original masks, before `prepare_label_channel`, as now.
   - Check that moving `configure_from_yaml` earlier does not change the cfg the loaders or the model see. `hgt_loader` sets H2GB's cfg from its arguments. If it does change it, keep the order and attach `pestat_Hetero_Metapath` to the view's node stores instead. State which way you chose and why.
2. The training path must be unchanged:
   - same batches, same RNG draws (`_Seed0` restores the RNG state; `torch.manual_seed(seed)` still comes before `build_gtmodel`), same model;
   - say in the report whether any training input changes. Expected: none.
3. Check the other callers of `val_subset_loader`: `h2sgnn.py:493`, `hetsheaf.py:786`, `hgb_han.py:454`, `rgcn.py:282`, `h2gb_published.py:304`, `minibatch.py:555`.
   - For each, does the caller write any node or edge attribute to `data` after the call that its model reads? List the answers. Read only, change nothing unless one does.
   - If one does, stop and report.

## Tests (only these)
- **One CPU test.** Reuse card 17's cache-test fixture (`tests/test_card17_ieee_cis.py`, H2GB's `preprocess_Metapath` stubbed to write a known embedding) on the same Metapath dataset.
  - Run `fit` with a plan of one training step and one evaluation epoch, so the B1a subset is evaluated once.
  - It must pass on the fix and fail on `0b9873d` with the shape error. Show both.
  - Also assert that a subset-loader batch has `pestat_Hetero_Metapath` on every node type that a train batch has it on.
- **Training identity.** In the same job, record the fixture's first training-step loss on `0b9873d` and on the fix; they must be equal. Run the old commit from a second checkout or `git worktree`.
- No GPU smoke, no full suite. Ops runs the full suite once at review, and the real check is ops' rerun of seed 0.

## Preconditions
- Worktree `/mnt/data/peeradon/h2snn-card19`, branch `card19` from `origin/v3b` `0b9873d0c19455c8df56fa152492188d8918b62d`; `git merge` the design-docs SHA given in the prompt.
- Copy (do not symlink) the git-ignored `vendor/`; `data` → `/mnt/data/peeradon/h2snn/H2S-NN/data`.

## Rules
- `sbatch` only on the lab, CPU job `--mem 16G --time 1:00:00`.
- `git diff origin/v3b...card19 -- H2S-NN/src` touches only `src/baselines/adapters/_h2gformer_h2gb.py`, plus `src/baselines/minibatch.py` only if step 1's fallback needs it there.
- Stage by explicit path; one commit for code + test, one for the report; push `card19`. Never merge into `v3b`; never stash, reset or rebase.
- Stop if the training identity fails, or if another caller in step 3 has the same defect.

## Report (`docs/H2S-NN_card19_report.md`, short)
- The diff and which way step 1 went.
- The test result on the fix and on `0b9873d`, with job ids.
- The training-identity line.
- The step 3 table: caller, writes after the call (yes/no).
- Last line `branch=card19 sha=<head> card=docs/H2S-NN_card19_card.md base=0b9873d0c19455c8df56fa152492188d8918b62d`.

## Estimate
About 2 h, including the queue.
