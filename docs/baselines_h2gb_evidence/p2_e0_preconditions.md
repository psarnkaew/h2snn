# Part 2, E0 — preconditions against the author's decisions (card `H2S-NN_baselines_h2gb_part2_card.md`)

Branch check (VERIFIED-RUN, login node, git only): `origin/v3b` = `6ba6dec00369ff590b341ce3771da90528fcc1cd`, `origin/design-docs`
= `726ec849d2a644b50e4a0ff22d5a37c753ab7075`; `baselines-h2gb` was at `6ba6dec`; `git merge --no-ff 726ec84` clean.
Read: the part-2 card, the decisions-log rows of 2026-11-16 from "Baselines on H2GB: S1 sampler depth" on, part 1's card,
report and `c0_preconditions.md`; vendor files below (VERIFIED-CODE, quoted lines opened by me).

## Per method: what the decisions change
| Method | Class / decision | Sampler / batching code | Layer parameter | Change in part 2 | E0 result |
|---|---|---|---|---|---|
| GCN, GAT | A, S1 | H2GB `get_HGTloader` with the YAML list unchanged (`custom_sampler.py:323-341`; `src/baselines/minibatch.py`) | HGB `num_layers` (+1 convs) | adapter on `hgt` batches; equivalence test at depth = convs + 1 | agrees |
| HAN | A, S1 + S5 | as GCN; H2GB's HAN = PyG `HANConv` on raw relations (`hetero_gnn_model.py:127-129`) | `gnn.layers_mp` | H2GB's HAN on `hgt` batches, stated | agrees |
| H2G-former | A, S1 | H2GB `GTModel` on `hgt` batches; head reads the first `batch_size` targets (`hetero_node.py:29-38`) | `gt.layers` | existing adapter on batches | agrees |
| **HINormer** | A, S2 | H2GB's `hin` loader (`custom_sampler.py:602-690`) | `gnn.layers_mp` | — | **stop (H1, below)** |
| **PSHGCN** | B, S3 (b) | `PSHGCN/ogbn-mag` (`main.py:36-40, 91-95`) | `--K` | — | **stop (P1, below)** |
| SeHGNN | B, C4-a/b | part 1 adapter | `--num-hops` | OAG/PDNS: merge (C4-b, wired); MAG: ComplEx embeddings must be trained (below) | agrees; MAG cells pending author |
| ChebNetII | B, C4-c | part 1 adapter | `--K` | per-type projection after per-block bases (IEEE-CIS, H-Pokec, PDNS) | agrees |
| Hetero²Net | B, inputs B | `Hetero2Net/main.py:147-175` NeighborLoader | `num_layers=2` | NeighborLoader path; MAG reads `mag_metapath2vec_emb.zip` | agrees |
| NSD, CSNN | C, S1 | H2GB `hgt` with the dataset's GCN YAML list | `layers` | adapter on `hgt` batches | agrees |
| **H2SGNN** | C, S4 | — | `--K` | — | **stop (S4b, below)** |

## Stops (decision vs vendor code)
**H1 — HINormer (S2: "H2GB's own HINormer, H2GB's sampler and configs; class A").** H2GB's `HINormerModel` is not HINormer
and not mini-batched:
- its transformer is commented out: `h = gh[seqs]` (`H2GB/network/hinormer_model.py:274`), `# h = self.GTLayers[layer](h, rh=r)`
  (`:277`), `output = self.prediction(h[:, 0, :])` (`:289`): the output is a linear map of the **ego node** after
  `layers_mp` full-graph GCN layers (`:264-271`, on `self.edge_index` of the whole `dataset[0]`, `:210-211`); the sequences
  are read but only position 0 is used;
- its feature encoder is commented out (`:221-222`) although the configs ask for `node_encoder_name: Hetero_Raw`;
- its loader yields the whole split as one item (`TensorDataset(features.unsqueeze(0), label.unsqueeze(0),
  seq.unsqueeze(0))`, `DataLoader(..., batch_size=None)`, `custom_sampler.py:678-685`): one full-graph step per pass;
- it caches the sequences at a hard-coded absolute path (`/nobackup/users/junhong/Data/<name>.pt`, `:618-646`);
- H2GB has HINormer configs for ogbn-mag, mag-year, oag-eng and PDNS only; no "nearest family" rule exists in its code.
Running "H2GB's own HINormer" therefore means a full-batch GCN + linear head under the name HINormer. Decision needed.

**P1 — PSHGCN (S3 (b): "only the input widths and the target type from the dataset; propagation, coefficients, label part,
loss and optimiser unchanged").** More than the widths is MAG-specific in their ogbn-mag code:
- the polynomial coefficients are enumerated from MAG's relation list `keys = ["AI","AP","PP","PF","IA","PA","FP"]`
  (`ogbn-mag/processing.py:14-15`, `expander`); another dataset needs its own list;
- the label part uses the fixed keys `['PPPP', 'PAAP', 'PP', 'PFFP']` (`main.py:126`) and removes each key's self-effect
  with diagonals written by `sparse_tools` for those MAG keys (`processing.py:133-151`, `main.py:118-123`);
  `sparse_tools` (a separate C++ package) is not installed in `h2snn-b`, and installing it is outside this card's files;
- the model itself is reachable without an edit (`propagate`, `utils.py:111-133`, and `PSHGCN.forward`, `model.py:96-120`,
  read the target as letter `P` and each type by its letter; only `embedings_x` and `lin_residue` carry widths).
So: on ogbn-mag/mag-year the decision is reachable except the label part (needs `sparse_tools`); on the other datasets the
relation list and label keys must be chosen (a design decision). Stop item for PSHGCN (all cells).

**S4b — H2SGNN (S4: "follows S1").** Its operators come from its own preprocessing, which has one hand-written branch per
HGB dataset: links paired by HGB's file order (`H2SGNN/processing.py:46-60`), ACM merges P–P with its reverse and drops
`IP`, DBLP reorders links and fixes meta-paths (`:60-121`), then target–X–target products (`:125-158`). No branch or rule
exists for the H2GB datasets (OAG has parallel relations, IEEE-CIS 12 types). Running it on `hgt` batches needs a relation
dictionary per dataset (a design decision). Stop item for H2SGNN (NC and LP).

## SeHGNN on ogbn-mag / mag-year (decision C4-a, inputs B)
`SeHGNN/ogbn/Readme.md`: the published MAG run uses ComplEx embeddings for author / field / institution, **generated by
training** (`dglke` under `dgl<1.0`, `../data/complex_nars`: `convert_to_triplets.py`, `embed_train.py --model ComplEx ...
--max_step 500000`, `split_node_emb.py`); no download is offered. Per the decision the two cells stay **"pending author
(embeddings need training)"**; nothing is trained here. (Their alternative, random-uniform inputs, is the README's
"without extra embeddings" command.)
