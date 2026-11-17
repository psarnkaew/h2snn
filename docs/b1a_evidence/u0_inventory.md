# Card 1f U0 — H2GB NC training paths (inventory, no code)

Base: `baselines-h2gb-6` after merging v3b `a8828a8`. Wired cells come from `registry.H2GB_NC_WIRED_CELLS`. Paths are given under `H2S-NN/src/baselines/`.

| Method | Entry (H2GB NC branch) | Training call | Loop |
|---|---|---|---|
| GCN, GAT | `adapters/hgb_gnn.py:192` → `_fit_h2gb` :304 | `minibatch.fit_homogeneous_on_hgt` (:386) → `train_minibatch_node_classification` (:417) | A (HGTLoader) |
| NSD | `adapters/nsd.py:327` → `_fit_h2gb` :466 | `fit_homogeneous_on_hgt` | C (HGTLoader) |
| CSNN | `adapters/csnn.py:275` → `_fit_h2gb` :394 | `fit_homogeneous_on_hgt` | C (HGTLoader) |
| HAN | `adapters/hgb_han.py:193` → `fit_h2gb_han` | `train_minibatch_node_classification` (`hgb_han.py:459`) | A |
| H2G-former | `adapters/h2gformer.py:267` → `_h2gformer_h2gb.fit` | `train_minibatch_node_classification` (`_h2gformer_h2gb.py:157`) | A |
| HetSheaf | `adapters/hetsheaf.py:561` → `fit_h2gb_node_classification` :763 | `train_minibatch_node_classification` (:785) | C |
| HiNormer | `adapters/hinormer.py:196` → `_fit_h2gb` :280 | `train_index_minibatch` (:305) | B (row batches) |
| SeHGNN | `adapters/sehgnn.py:218` → `_fit_h2gb` :362 | `train_index_minibatch` (:389) | B |
| ChebNetII | `adapters/chebnetii.py:164` → `_fit_h2gb` :231 | `train_index_minibatch` (:284) | B |
| PSHGCN (ogbn-mag, mag-year) | `adapters/pshgcn.py:205` → `_pshgcn_mag.fit` | `train_index_minibatch` (`_pshgcn_mag.py:215`) | B |
| **Hetero2Net** | `adapters/hetero2net.py:250` → `_fit_h2gb` :519 | **own loop** (:556-571): vendor `train_mini_batch`, then at each `is_eval_epoch` its own `evaluate(val_loader)` and `evaluate(test_loader)` over NeighborLoaders | **STOP ITEM** |
| H2SGNN | not wired on H2GB NC (`LARGE_LP_ADAPTER_PENDING` / card 3 Part B) | — | — |

## Stop item: Hetero2Net

Hetero2Net on the 8 H2GB NC datasets uses its own epoch loop, which follows the vendor's scheme (NeighborLoader, `train_mini_batch`). It does not use either shared loop. So U2 does not reach it: it keeps evaluating the whole validation and test splits at every evaluation epoch. Per the card, no change was made. Options for the design session:
- (a) Apply the U2 rule inside `hetero2net.py::_fit_h2gb`: a NeighborLoader over the subset ids, no test pass during training, best weights restored, then the whole-split evaluation. This touches an adapter outside the card's list of callers. About 1 to 2 h, plus one test.
- (b) Leave it on whole-split validation and state this as a protocol difference in the table notes.
