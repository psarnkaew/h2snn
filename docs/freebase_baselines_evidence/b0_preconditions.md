# Card 3 Part B, B0: preconditions (2026-11-18)

Branch `freebase-baselines` from `origin/v3b` = `a8828a8c002e17714cbd588d147b8c9f128fca1f`; merged design-docs
`149729031f74dd8262a2de2b79fa963d91825636` (`--no-ff`, `250fc39`; docs and `H2S-NN/paper/` only). An earlier local branch
created from the wrong base (v3b `d11810c`, never pushed) was renamed `freebase-baselines-wrongbase-7b58f01` (kept).

Author decision 2026-11-18 (B2 amendment): no real-data run on the lab cluster; B2 = full suite once.

## Vendor commits (`vendor/VENDORS.tsv`)
SeHGNN `e92bd37`, PSHGCN `f87406c`, H2SGNN `93c33fa`.

## F-1: each source's dataset branch (file:line)
| Source | Branch | Lines |
|---|---|---|
| SeHGNN | **own Freebase branch** | target `'0'`, types `0..7`: `hgb/main.py:43-46`; propagation device / cache dir: `:54-62`; sparse meta-adjacency propagation `hg_propagate_sparse_pyg(..., prop_feats=True)` and `raw_feats['0'] = SparseTensor.eye`: `:80-124`; `data_size` = node counts: `:138-139`; label propagation (meta-adjacencies reused when `num_label_hops <= num_hops`, `left_keys` filter, `remove_diag`): `:237-300`; model per-source-type projection on SparseTensor features: `hgb/model.py:198-202`, Freebase task head `:157-170`, `:229-234`; relations named `f'{dst}{src}'`: `hgb/utils.py:273-288`; `00` symmetrised: `:428-447`; reverse of each cross-type relation added as `adj.t()`: `:461-467` |
| SeHGNN | (not used) | dense `torch.eye` per featureless type before the branch: `hgb/utils.py:241-250` (Freebase: ~37 GB for the 8 types; the Freebase branch never reads it) |
| PSHGCN | **no Freebase branch**; generic = ACM/IMDB/AMiner pattern | features (eye if featureless) `hgb/processing.py:13-21`; one `SparseTensor` per stored relation `:43-50`; DBLP reorder `:51-52`; even/odd reorder `:53-59`; ACM `PP` merge + `set_diag` `:61-67`; row-normalise `D^-1 A` `:69-77`; per-dataset keys DBLP `:80-86`, ACM `:87-95`, IMDB `:96-102`, AMiner `:103-107`; `lis` chaining `i[-1]==j[0]` `:109-119`; `lis_t` `:120-128` |
| H2SGNN | **no Freebase branch**; generic = ACM/IMDB/AMiner pattern | features `processing.py:17-25`; relations `:47-54`; reorder `:55-63`; ACM `PP` merge `:65-71`; row-normalise `:73-81`; per-dataset keys with `target` / `target_max` ACM `:85-95`, IMDB `:96-104`, AMiner `:105-111`, DBLP (meta-paths) `:112-123`; DBLP meta-path products `:128-144`; generic target–X–target products filtered to the target block `:145-158` |

## Freebase schema (HGB `info.dat`)
8 node types (target 0 BOOK), 36 relations, **each stored in one direction only**, no two relations on the same
ordered type pair. Author decision (this session, 2026-11-18, question "Freebase dir"): the generic branch **completes
missing reverses**: the transpose of each relation whose reverse is not stored is added (cross-type: a reverse key;
same-type: merged into the key, which gets `set_diag` as ACM's `PP`). A no-op where reverses are stored (ACM, IMDB,
H2GB's `ToUndirected` graphs), so the ACM-branch equivalence holds. As SeHGNN's Freebase branch (`hgb/utils.py:446,
461-467`) and PSHGCN's `hgb-lp` processing.

## Batching (card Part B, part 1 classes; prompt: one of the two shared loops of `minibatch.py`)
- PSHGCN, 6 non-MAG H2GB NC: class B, their decoupled scheme (per-monomial propagation of the inputs once, batches of
  target rows), generic relation list, no label part -> `minibatch.train_index_minibatch`.
- H2SGNN, 8 H2GB NC: class C, not node-wise (`H2SGNN/model.py:67-86, 90-140`, part 1 C0) -> H2GB's `hgt` sampler with
  the dataset's GCN fan-out (D1 second branch, S4 = S1) -> `minibatch.train_minibatch_node_classification`.
- Large LP: the shared `large_lp_training.fit_large_lp` (not edited). PSHGCN decoupled: per-node table on the message
  graph, rows read by `n_id`; H2SGNN: sampled subgraph (`h2gb_sampler`).
