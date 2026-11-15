# B7 — meta-path items on H2GB (decisions log 2026-11-13, Meta-paths on H2GB, M-path 3 and 4)

## 1. Hetero²Net's MAG features vs the B2 loader's (M-path 3)

| Item | Hetero²Net's own MAG branch (`vendor/Hetero2Net` `4913394`, `hetero2net/dataset.py:56-63`) | B2 loader, ogbn-mag / mag-year (H2GB `MAGDataset`, `src/data/h2gb_nc.py`) |
|---|---|---|
| Source | PyG `OGB_MAG(root, preprocess='metapath2vec', transform=T.ToUndirected())` | H2GB's processed `NC/mag/{ogbn-mag,mag-year}/processed/data.pt` (built by H2GB from OGB's `PygNodePropPredDataset`, `mag_dataset.py:91-177`) |
| Node types | paper, author, institution, field_of_study | the same four (`h2gb_nc_counts.md`) |
| Paper features | OGB's `node-feat/paper` (128), read by PyG (`torch_geometric/datasets/ogb_mag.py:128-131`) | OGB's `x_dict['paper']` (128); **bit-identical** to OGB's own processed array on disk (`torch.equal` = True, job 16741 step 43, `b7_mag_features.txt`) |
| Author / institution / field features | PyG's pretrained **MetaPath2Vec** embeddings (128), downloaded from `https://data.pyg.org/datasets/mag_metapath2vec_emb.zip` (`ogb_mag.py:51-53, 119, 150-153`); **not on disk** (`NC/mag` holds OGB's own download and H2GB's files only) | H2GB's row-normalised averages of paper features: author over `writes`, field over `has_topic`, institution over the authors' features (`mag_dataset.py:144-162`), 128 each |
| Featureless types | none (all four have `x`) | none |
| Edges | OGB's four relations + `ToUndirected` (reverses added, `cites` symmetrised) | the same relations, `cites` symmetrised at processing time, reverses stored (`mag_dataset.py:94-115`); 42,182,144 edges |
| Split | OGB's time split (`ogb_mag.py:167-176`) | OGB's split via `get_idx_split` (ogbn-mag); random 50/25/25 (mag-year) |
| Meta-paths (metapath-drop) | `MAG` list P–A–I–A–P, A–P–F–P–A (`dataset.py:58-63`; decisions log M-path 2) | — (a property of the method, not the loader) |

**Decision for the author (B7, M-path 3): which features the Hetero²Net adapter uses on ogbn-mag and mag-year** (the adapter is not changed by this card):

| Option | What | Consequence |
|---|---|---|
| **(a) their MetaPath2Vec features** (recommended) | Paper features as everywhere (identical array); author, institution, field from PyG's `mag_metapath2vec_emb.zip`, which the author downloads once (no download in this project's code) and places under `NC/mag` | Spec §6: "feature preprocessing follows each method's original code, as part of the method"; their MAG branch is the code they ran. The other methods keep H2GB's averaged features; only the three non-target feature tables differ. |
| (b) the B2 loader's features | H2GB's averaged features for every method, Hetero²Net included | One feature table for every method; a stated deviation from their MAG branch (`dataset.py:56-57`). No download. |

Recommendation (a), because the spec's rule makes feature preprocessing part of each method; (b) if a download is not
wanted. Either way paper features and the graph are the same.

## 2. SeHGNN and PSHGCN on H-Pokec: meta-path counts and memory, before any run (M-path 4)

Enumeration by each method's own code on the H-Pokec schema (`src/baselines/metapath_counts.py`; tests
`tests/test_metapath_counts.py` on a schema with hand-counted keys): SeHGNN's `hg_propagate_feat_dgl` and
`hg_propagate_sparse_pyg` (`vendor/SeHGNN/hgb/utils.py:52-140`, `e92bd37`) on a one-node-per-type schema graph, and
PSHGCN's term block (`vendor/PSHGCN/hgb/processing.py:109-129`, `f87406c`) read from its file and executed. Schema and
counts from the B2 record: 16 node types (user 1,632,803; 15 club types, largest `relationships` 22,067, `life_style`
21,579, `music` 20,080), 31 relations (friendship symmetric; 15 user→club and their reverses); N_all = 1,731,977.
No feature was propagated and no single propagation was timed. Job 16741 step 42 (`b7_job.sh` → `b7_job.txt`,
`b7_metapath_cost.json`).

| SeHGNN `num_hops` / `num_label_hops` (its range {2, 3, 4}) | feature keys at the user type | label keys | stored propagated features (float32) | label features |
|---|---|---|---|---|
| 2 | 48 | 17 | 1,213.6 GiB | 0.21 GiB |
| 3 | 319 | 48 | 10,877.7 GiB | 0.58 GiB |
| 4 | **1,055** (expected ≈ 1,050 from the schema, decisions log) | 319 | 29,685.4 GiB | 3.88 GiB |

Feature memory = Σ over keys of N_user × width of the key's source type × 4 B, where user features keep their width 65
and a featureless type enters SeHGNN as `torch.eye(N_type)` (`hgb/utils.py:246-247`), so a key ending at a club type
is `N_user × N_club` (e.g. `relationships`: 1,632,803 × 22,067 × 4 B = 134 GiB for one key). SeHGNN propagates and keeps
these features dense before training (`main.py:61-76`, DBLP/ACM/IMDB branch). Its Freebase branch, the only one whose
types are all featureless in its code, propagates sparse meta-path adjacencies instead (`main.py:77-143`); how an H-Pokec adapter
would treat featureless types is not in their code (adapter pending, B8).

| PSHGCN `K` (its NC range {1, 3, 5}) | terms per pass (1 + Σ_k \|lis[k]\|) | activations per forward, hidden 16 / 64 / 256 (its range up to 256) |
|---|---|---|
| 1 | 32 | 6.6 / 26.4 / 105.7 GiB |
| 2 | 303 | 62.6 / 250.2 / 1,001.0 GiB |
| 3 | 1,039 | 214.5 / 858.1 / 3,432.3 GiB |
| 4 | 5,840 | 1,205.8 / 4,823.1 / 19,292.3 GiB |
| 5 | 21,681 | 4,476.4 / 17,905.7 / 71,622.8 GiB |

Activations = terms × 2 passes (g and gᵀ) × N_all × hidden × 4 B (`hgb/model.py:45-100` keeps one tensor per term
and pass for autograd). Cluster memory for comparison: host RAM 64,210 MB on g2n01, g2tn01, g3n01 and 225,171 MB on
g3an01 (`sinfo`); GPUs 2080 Ti 11 GB, TITAN RTX 24 GB, 3090 24 GB, A6000 48 GB (`H2S-NN/CLAUDE.md`, cluster section).
`docs/lab-gpu-cluster.html` was not read (the read was refused by the session's permission policy); `sinfo` and
CLAUDE.md give the numbers used.

**Reading (search ranges unchanged, as the decision says):** on H-Pokec only PSHGCN at K = 1 with hidden ≤ 64 fits a
GPU (26.4 GiB on the A6000) and only K = 1 fits a host; every SeHGNN setting exceeds every node by one to four orders
of magnitude. This is reported for the author's plan of the baseline cells (B8 lists the adapters as pending).
