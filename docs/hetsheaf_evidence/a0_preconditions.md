# A0 — Preconditions (card 3 Part A, HetSheaf)

Branch `hetsheaf` from `v3b` `2ef9cac14ae03136efb07ad0c71233697ae138b1` (checked = `origin/v3b`), `git merge --no-ff
93092977c5045c5c9b851d412c81306312d52b23` (clean; `origin/design-docs` contains it). Worktree `/mnt/data/peeradon/h2snn-hetsheaf`.
Vendor clones copied (rsync, not symlinked) from `/mnt/data/peeradon/h2snn/H2S-NN/vendor`; each clone's `HEAD` equals its
`VENDORS.tsv` commit (15 clones); the only working-tree changes are those of the source worktree (H2GB: the vendor patch
`ieee_cis_dataset.py`, `pokec_dataset.py` and the author's `examples/use_dataset.py`; HGB: `CLONED_COMMIT.txt`).
`H2S-NN/data` is a symlink to the main worktree's data (never staged). Jobs: allocation 16741 (g2n01) via `srun --overlap`.

## Readings (VERIFIED-CODE, clone `vendor/Heterogeneous-Sheaf-Neural-Networks` at `6964e1e`; paths under `hetsheaf/`)
Part P P0 (a)–(e) re-read against the clone: every point holds. File:line per item:

| Item | File:line | Reading |
|---|---|---|
| Model arguments | `models/sheaf_gnn/config.py:42-91` | `SheafModelArguments`: `d`, `layers`, `hidden_channels`, `input_dropout`, `dropout`, `initial_dropout` (never read by the forward), `output_dim` (only `lin2`, commented out), `orth=householder`, `normalised=True`, `edge_weights=True`, `add_lp/add_hp=True` |
| Encoder base | `models/sheaf_gnn/sheaf_base.py:10-65` (`:22-28`) | `final_d = d + add_hp + add_lp`; `hidden_dim = hidden_channels · final_d`; `edge_index`, `graph_size` fixed at construction (`:18`, `:30`); `update_edge_index` `:67-76` |
| Encoder construction / forward (diag) | `models/sheaf_gnn/transductive/disc_models.py:80-162` / `:164-225` | `lin1(input_dim→hidden_dim)` after input dropout, ELU; reshape to `[N·final_d, C]`; per layer: learner → `laplacian_builder` → left/right weights → sparse product → ELU → `x0 = (1+tanh ε)x0 − x`; returns `(x [N, hidden_dim], maps)`; `lin2` commented out (`:223`) |
| Bundle / general | `disc_models.py:233-415` / `:417-587` | same structure; builders `NormConnectionLaplacianBuilder` (O(d), `orthogonal.py:10` `torch_householder_orgqr`) / `GeneralLaplacianBuilder` (`laplacian_builders.py:294`, `:425`) |
| Learner choice | `models/sheaf_gnn/utils.py:28-93`, `config.py:99-120`, `disc_models.py:38-52` | string → class; homogeneous `local_concat`, `attention`; heterogeneous `type_concat`, `type_ensemble`, `edge_type_concat`, `node_type_concat`, `node_type`, `edge_type`, `types_only`; seven `attention_*` |
| Learners use type ids | `models/sheaf_gnn/sheaf_models.py:236-316` (TE), `:296-299` | `F.one_hot(node_type, num_node_types)`, `F.one_hot(edge_type, num_edge_types)`: contiguous int64 ids |
| Map family × learner lists (their sweeps) | `sweeps/{nc/acm,dblp,imdb; lp/hgb_*,last_fm,…; gc/*}/diag_sheaf.yaml` (e.g. `sweeps/nc/acm/diag_sheaf.yaml:11-21`) | diag: 8 learners (`local_concat` + the 7 heterogeneous); bundle and general: 16 (the 8 + `attention` + 7 `attention_*`) in 16 of 18 files; the two 8-learner bundle/general files are `lp/as733`, `nc/routenet_traffic` (not our datasets) |
| Shared sweep ranges | `sweeps/nc/acm/diag_sheaf.yaml:24-57` | `d` int 2–5, `layers` 2–8, `hidden_channels` {8,16,32,64}, `input_dropout` U[0, 0.9], `dropout` U[0, 0.9], `initial_dropout`, `output_dim` (no effect), lr / wd categorical, 500 epochs, bayes, `run_cap 150` (H-3 holds) |
| `to_homogeneous` type ids | `datasets/hgb.py:63-82` | `dataset[0].coalesce()`, then `data.to_homogeneous()`; `num_node_types`, `num_edge_types`, `graph_size`, `in_channels = data.num_features` |
| Transform chain | `datasets/hgb.py:51-60`; `datasets/utils.py:11-28` (`RemoveSelfLoops`), `:31-88` (`SubsetTrainSplit`, unseeded unless `seed` given, `:44-45`) | `Constant(node_types=None)`, `ToUndirected`, `SubsetTrainSplit(0.1)`, `RemoveSelfLoops`, `NormalizeFeatures`, `RemoveDuplicatedEdges` (H-2 holds; `SubsetTrainSplit` dropped) |
| NC head | `node_classification/node_classifier.py:39, 66-71`; `sheaf_node_classifier.py:37-46, 49-65` | `Linear(model.hidden_dim, C)` on the encoder output; CE, IMDB `multilabel_soft_margin_loss`; mask = split ∧ labelled |
| NC optimiser | `node_classifier.py:178-192` | AdamW + `CosineAnnealingLR(T_max=1000)` (dropped, H-2) |
| LP head (theirs; not used, H-2) | `link_prediction/sheaf_link_predictor.py:26-27, 35-53` | `Linear(2·hidden, 1)` on `[h_u ‖ h_v]`, BCE |
| GC poolings | `graph_classification/graph_classifier.py:43-54`; `pooling.py:7-40` (`AttentionPooling`, heads 2) | mean / max / add (PyG globals), attention |
| GC SheafPool | `graph_classification/sheaf_graph_classifier.py:33-142` (`LateTokenSheafReadoutTransport`), `:161-192` (decoder input `channel_dim` for `late_token`, `hidden_dim` otherwise) | default `pooling=late_token` (`configs/sheaf_config_gc.yaml:23`, `scripts/run_sheaf_gc.py:54`); GC uses the inductive models (`configs/sheaf_config_gc.yaml:4`, `models/sheaf_gnn/inductive/disc_models.py:18-60`) |
| GC optimiser | `graph_classifier.py:155-161` | AdamW 1e-3 / 5e-4 hard-coded + cosine (replaced by the shared axes, H-2) |

No file:line contradicts H-2 or H-3.

## H-1 check (VERIFIED-RUN)
- `a0_env_check.txt` (job 16741): in `h2snn-b` before any install, missing: `strenum`, **`lightning`** (the unified package),
  **`omegaconf`**, `hydra`. Present: `pytorch_lightning` 2.6.5, `torchmetrics` 1.8.2, `torch_householder` 1.0.1,
  `torch_scatter`, `torch_sparse`, torch 2.4.1+cu124, PyG 2.8.0, DGL 2.4.0+cu124.
- `a0_install_strenum.txt`: `pip install --no-deps strenum==0.4.15` (their `uv.lock:1255-1256`). `pip freeze` diff: only
  `StrEnum==0.4.15` added; torch / PyG / DGL unchanged; `pip check` unchanged (the accepted DGL line).
- `a0_gate_after_strenum.txt` (step 118): `v3b_golden` compared 1365 tensors, **differing: none**; `mb0` 139 tensors, **differing: none**.
- `a0_import_probe.txt` (`a0_import_probe.py`, CPU):
  1. `import hetsheaf.models.sheaf_gnn` **fails**: `ModuleNotFoundError: No module named 'lightning'`. Python runs
     `hetsheaf/__init__.py:4-10`, which imports `core`, `datasets` (`hgb.py:9, 11`: `lightning`, `pytorch_lightning`),
     `link_prediction`, `node_classification`, `graph_classification` (all Lightning modules); `models/__init__.py:4`
     imports `gnn_baselines` (`HAN.py`: `omegaconf`).
  2. With the package `__init__` chain bypassed (`hetsheaf`, `hetsheaf.models`, `hetsheaf.graph_classification` registered
     as empty packages whose `__path__` is the vendor folder; no vendor file changed, nothing else installed), the model code
     imports and runs on torch 2.4.1: forward + backward finite for diag (`local_concat`, `type_concat`, `type_ensemble`,
     `types_only`), bundle and general (`type_concat`, `attention_type_concat`) on a synthetic 2-type, 3-relation graph after
     `to_homogeneous`; mean / max / add / attention GC poolings run.
  3. SheafPool's module `sheaf_graph_classifier.py:10, 14` imports `lightning.pytorch.utilities.types` and
     `graph_classifier.py:7, 11, 24` (`lightning.pytorch`, base class `L.LightningModule`): **fails** as is. With `lightning.pytorch` aliased to the installed
     `pytorch_lightning` 2.6.5 (probe only) it imports and runs.

So the model code does **not** need torch > 2.4.1 (H-1's stop condition is not met), but H-1's premise "the only missing
package is `strenum`" is false for importing it: see stop items.

## Stop items (design session / author)
| # | Item | Facts | Options (recommendation first) |
|---|---|---|---|
| S-A0-1 | Import route (H-1 premise) | `import hetsheaf.models.sheaf_gnn` needs `lightning` + `omegaconf` through the package `__init__` files; SheafPool's own module imports `lightning` | (a) **Recommended:** the adapter imports the model code with the package `__init__` chain bypassed (empty package modules over the vendor folders, no vendor edit, no install; probe 2 above), and imports SheafPool's module with `lightning.pytorch` mapped to the installed `pytorch_lightning` 2.6.5 (their lock: `lightning` 2.6.0, `pytorch-lightning` 2.6.1; only type names and the `LightningModule` base class of `GraphClassifier` are touched, the readout class is a plain `nn.Module`); stated. (b) Install `lightning==2.6.0` and `omegaconf==2.3.0` with `--no-deps` (their `uv.lock:437-438, 746-747`) and import normally; golden/mb0 again. (c) Separate env (H-1 fallback). |
| S-A0-2 | Requirements-lock line for `strenum` | `tests/test_check_env.py:63-71` asserts the lock has exactly 28 pins; adding `strenum==0.4.15` makes 29 and fails that existing test (`a0_gate_lock.txt`, step 121; with the line, `check_env.py` reported 0 differences, `a0_check_env.txt`); the test file is outside Part A's list. The line is therefore **not** committed (installed only; `check_env.py` does not check unlocked packages). | (a) **Recommended:** allow the one-line change in `tests/test_check_env.py` (28 → 29, plus `pins["strenum"] == "0.4.15"`) together with the lock line. (b) Record `strenum` only in the report/evidence. |

A1–A4 all import the model code, so none is independent of S-A0-1.
