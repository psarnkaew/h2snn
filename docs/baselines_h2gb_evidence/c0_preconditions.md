# C0 — preconditions: the mini-batch scheme of each baseline, from the vendor code

Card `docs/H2S-NN_baselines_h2gb_card.md` C0. Branch `baselines-h2gb` from `v3b` `4bfa5dea2da5ad8f96a6e6d181c0cfc58b05e5cd`
with `design-docs` `4d207013b0e73249fda10c0bb984a51460f76a12` merged `--no-ff` (merge commit `724c117`, no conflict);
`origin/v3b` and `origin/design-docs` equalled the pinned SHAs after `git fetch`.

Read: the card, `docs/working_preferences.md`, the M3b report and `docs/partM3b_evidence/{b7_metapaths,b8_budget_and_cells,
pdns_protocol}.md`, the decisions-log rows "Part M3b review", "Baselines on H2GB card" (2026-11-16) and D-P3/D-P6/D-P9,
`H2S-NN/AGENTS.md`, `H2S-NN/CLAUDE.md`, `src/baselines/{registry,budget,training,bundles,heads,encoders}.py`,
`src/baselines/adapters/hgb_gnn.py`, `src/data/{h2gb_nc,large_lp}.py`, `src/trainers/minibatch_lp.py`. The vendor files below
were read by me (lines quoted) or, where marked (ag), first located by a read-only search agent and then re-opened by me at
the decisive line. Vendor commits as `vendor/VENDORS.tsv`.

## Probe (VERIFIED-RUN): H2GB's `hgt` sampler as H2GB's code runs it
`c0_probe.py` → `c0_probe.txt` (job 16741 step 59, g2n01, commit `724c117`, 21.5 s, peak RSS 0.74 GiB; synthetic graph and
H2GB's YAML only). Each YAML merged into H2GB's cfg (`set_cfg`, `merge_from_file`, as `H2GB/main.py`), `sample_sizes`
computed with H2GB's own expression (`H2GB/sampler/custom_sampler.py:323-326`). PyG's HGTLoader runs
`num_hops = max(len(list))` iterations (`torch_geometric/sampler/hgt_sampler.py:48`, PyG 2.8.0).

| dataset | GCN / GAT / HAN: fan-out × iterations; model depth key | H2Gformer: fan-out × iterations; `gt.layers` |
|---|---|---|
| ogbn-mag, mag-year | 512 × **6**; `gnn.layers_mp` 4 | 512 × 3; 3 |
| oag-cs, oag-eng, oag-chem | 64 × **6**; `layers_mp` 3 | 64 × **6**; 3 |
| ieee-cis-g | 64 × **6**; `layers_mp` 3 | 64 × **6**; 3 |
| h-pokec | `neighbor_sizes_dict`: user [2048, 2048, 512, 512] (**4**), 15 club types [128, 128, 128] (**3**); `layers_mp` 3 | same dict; 3 |
| pdns | 256 × **4**; `layers_mp` 3 | 256 × 4; 3 |

Also (same run): with fan-out ≥ every node, HGTLoader's node set after d iterations is exactly the d-hop in-neighbour ball
and its edges are the induced subgraph (P3, d = 1, 2, 3); HGTLoader accepts per-type lists of unequal length (P2).
H2GB also benchmarks HINormer itself (`configs/{ogbn-mag,mag-year,oag-eng,PDNS}/*-HINormer.yaml`, sampler `hin`).

## Per method (class, source scheme, depth parameter, adapter today, what changes)

| Method | Class (card) | Source's scheme (file:line) | Depth / hop parameter | Adapter today on HGB (file:line) | Agrees with the scheme table? |
|---|---|---|---|---|---|
| GCN | A | H2GB `hgt`: `custom_sampler.py:323-341` (HGTLoader, `num_samples` = the YAML list, iterations = list length) | HGB GCN: `num_layers + 1` GraphConvs (`HGB/NC/benchmark/methods/GNN/GNN.py:74-79`); `norm="both"` = source out-degree × destination in-degree of the graph passed (`dgl/nn/pytorch/conv/graphconv.py:291, 427-430, 462-464`) | full batch, fixed `self.g` (`GNN.py:68, 88-89`; `hgb_gnn.py:192-218`) | **No** (S1) |
| GAT | A | as GCN | `num_layers + 1` GATConvs (`GNN.py:32-43`) | as GCN (`GNN.py:24, 52-54`) | **No** (S1) |
| HAN | A | H2GB: PyG `HANConv` on the raw relations (`H2GB/network/hetero_gnn_model.py:127-129`, `metadata = dataset[0].metadata()` l.66) under `hgt` | `gnn.layers_mp` | HGB's DGL HAN with meta-paths (`hgb_han.py:73, 186-191`; `HGB/NC/benchmark/methods/HAN/model_hetero.py`) | **No** (S1); see also S5 |
| H2G-former | A | H2GB `GTModel` (`H2GB/network/gt_model.py:62, 187`) on `hgt` batches; head reads the first `batch_size` targets (`H2GB/head/hetero_node.py:29-38`) | `gt.layers` (+ `layers_pre_gt`/`layers_post_gt`) | full batch, `batch_size = num_nodes` (`h2gformer.py:289, 538-542`) | **No** (S1) |
| SeHGNN | B | `SeHGNN/ogbn/main.py:85-96` (`hg_propagate` once before training), `:170-174` (DataLoader of target indices), `ogbn/utils.py:138-139` (rows of the stored features); multi-stage "enhance" `main.py:141-171, 410`; label propagation `main.py:179-283` | `--num-hops` (`main.py:418`), `--num-label-hops` (`:422`) | already batches over HGB features (`sehgnn.py:83, 245, 272-302`, hgb model) | Yes. Note: the ogbn data path exists for ogbn-mag only (`ogbn/utils.py:334-372`), with **random-uniform or ComplEx embeddings** for author/field/institution (`:345-354`), not H2GB's averaged features (out-of-scope finding F3) |
| PSHGCN | B | `PSHGCN/ogbn-mag/processing.py:153-159` (propagation once), `main.py:36-40, 91-95` (batches), `ogbn-mag/utils.py:36-38`; `main.py:78-87` are the **enhance loaders**, label propagation is `main.py:96-116` | `--K` (`main.py:198`) | full batch, hgb model propagates inside the forward (`pshgcn.py:220, 233-235`; `PSHGCN/hgb/model.py:45-110`) | **No** (S3) |
| ChebNetII | B | `ChebNetII/ogb/processing.py:90-105` (bases once), `ogb/large_train.py:54, 75-83, 125` (contiguous slices), `ogb/model.py:27-43` | `--K` (`large_train.py:29`) | full batch, `TypeProjectedInput` + `ChebnetII_prop` over the whole graph (`chebnetii.py:78-99, 150, 185-187`) | Yes |
| Hetero²Net | B | `Hetero2Net/main.py:64` (`mini_batch = dataset in ['MAG', 'RCDD']`), `:147-175` NeighborLoader, `--num_neighbors [10, 10]`, `--batch_size 1024` (`:39-40`); MAG data `hetero2net/dataset.py:54-63` | `HeteroGNN(num_layers=2)` default (models.py:32-33; not passed by main.py) | full batch (`hetero2net.py:240-293`) | Yes |
| HINormer | B | sequences built once by BFS (`HINormer/run.py:100-127`), **but every forward first runs `num_gnns` GCN / relation layers on the whole graph `self.g`** and only then gathers the sequences (`HINormer/model.py:204-208`); training is one forward over all training targets (`run.py:155`) | `--len-seq` (`run.py:242`), `num_gnns` | full batch (`hinormer.py:161-197`) | **No** (S2) |
| NSD | C | none (no loader in the source); Laplacian from the `edge_index` fixed at construction, degree-normalised (`neural-sheaf-diffusion/models/laplacian_builders.py:42, 70, 142-152`) | `layers` (`models/sheaf_base.py:29, 42`) | full batch on the homogeneous view (`nsd.py:186, 281-317`) | Yes (class C) → D1 second branch → `hgt` (S1) |
| CSNN | C | none (`vendor/csnn/utils/data_utils.py:61`: the whole graph as one batch); normalisation from the graph's degrees (`models/CSNNconv.py:74-78, 90`) | `layers` (`coopshv_model.py:32`) | full batch (`csnn.py:234-264`) | Yes (class C) → `hgt` (S1) |
| H2SGNN | C | **not node-wise**: every forward mixes target-target meta-path adjacencies with learned weights and propagates K times (`H2SGNN/model.py:67-86, 90-140`; adjacencies `processing.py:125-158`) | `--K` (`main.py:88`) | full batch (`h2sgnn.py:94, 265-267`) | D1 → second branch → `hgt` (S1, S4) |

## Contradictions with the scheme table (card C0: "a file:line that contradicts the scheme table is a stop")

**S1 — sampler depth (classes A and C).** The table sets "sampler depth = the method's number of layers (tuned value; H2GB's
per-hop fan-out repeated)". H2GB's code does not tie the depth to the layers: the number of HGTLoader iterations is the
length of the YAML list (`custom_sampler.py:323-331`), 6 for GCN/GAT/HAN on MAG, OAG and IEEE-CIS (4 or 3 layers), 4 on PDNS
(3 layers), 6 for H2G-former on OAG/IEEE (3 layers); on H-Pokec the per-type lists differ in length (user 4, clubs 3) and the
user fan-out differs by hop (2048, 2048, 512, 512), so "per-hop fan-out repeated" has no single value to repeat. The table's
depth rule is therefore not H2GB's scheme and is not stated as a deviation with file:line. It also matters for exactness:
with depth = number of propagation steps L, nodes at hop L have incomplete degrees in the induced subgraph, and HGB's GCN
(and NSD/CSNN) normalise by the degrees of both endpoints, so full-fan-out equivalence cannot hold at depth L (it can at
L + 1; tested in C1). **Decision for the author** (C0 report): (a) H2GB's own list unchanged (its depth, whatever the tuned
layers); (b) the card's rule (depth = propagation steps), with H-Pokec defined separately; (c) depth = propagation steps + 1.

**S2 — HINormer is not "batches of targets".** Its forward runs `num_gnns` full-graph GCN layers before the per-node
sequences are read (`HINormer/model.py:204-208`); batching targets leaves a full-graph computation in every step, and
batch-local computation needs a change to the model (or the full graph each step). H2GB benchmarks HINormer with its own
`hin` sampler (`custom_sampler.py:602-660`, configs for 4 of the 8 datasets), which would make it class A. Stop for HINormer.

**S3 — PSHGCN's own large-graph scheme cannot be reached without editing its model.** The ogbn-mag code that batches
precomputed features (`PSHGCN/ogbn-mag/main.py:36-40, 91-95`) builds its model with MAG's node type `P` and input widths
128/256 hard-coded (`PSHGCN/ogbn-mag/model.py:57-66`; H2GB's MAG has 128-dim features on every type, OAG 768); the HGB model
propagates inside its forward (`PSHGCN/hgb/model.py:45-110`). The table's line reference is also off: `main.py:78-87` are the
enhance (pseudo-label) loaders; label propagation is `main.py:96-116`. Stop for PSHGCN (NC and the decoupled LP branch of D2).

**S4 — H2SGNN is not node-wise on precomputed features.** D1's first branch does not apply (VERIFIED-CODE above); D1's second
branch sends it to H2GB's `hgt` sampler with depth = its layers (K), so it inherits S1. Its propagation runs over target-target
meta-path adjacencies built by its own preprocessing from HGB files (`processing.py:125-158`), which a sampled subgraph would
have to rebuild per batch; D2's "decoupled" LP branch does not apply to it either. Follows S1.

**S5 (expected C1 result, not a C0 stop) — HAN's semantic attention is a mean over every node of the graph it is given**:
HGB's `model_hetero.py:28` (`w = self.project(z).mean(0)`) and PyG's `HANConv` group function (`han_conv.py:25-30`,
`torch.tanh(k_lin(out)).mean(1)` over destination nodes). On a sampled subgraph that mean runs over the sampled nodes, so a
target's output depends on the other nodes of its subgraph: neither full-fan-out equivalence nor batch independence can hold
without changing the model. Tested in C1 (stop condition there if it fails).

**Notes (no stop).** SeHGNN's HGB model ends in `BatchNorm1d(track_running_stats=False)` (`SeHGNN/hgb/model.py:155`), a batch
statistic in eval; its ogbn model (the class-B scheme) uses running statistics (`ogbn/model.py:182, 198`). Hetero²Net uses
per-type BatchNorm (train-mode batch statistics; eval uses running statistics). H2GB evaluates at most `val.iter_per_epoch` ×
`batch_size` shuffled targets per evaluation (`custom_sampler.py:20-55`, configs), not the whole split; the protocol's whole-split
evaluation (MB3) is a stated difference.

## What can proceed (the card: "a unit that stops is reported and the next independent unit starts")
- C1: the shared infrastructure (batch plan, YAML fan-out through H2GB's code, whole-split evaluation, selection) does not depend
  on S1; the sampler takes the fan-out and the depth as explicit inputs. The equivalence tests run at depth = propagation steps
  (the card's rule) and report what they show.
- C2 (PDNS), C7 (housekeeping): independent.
- C3, C5 (NSD, CSNN, H2SGNN) and the message-passing part of C6 need S1 decided; HAN additionally S5.
- C4: SeHGNN, ChebNetII, Hetero²Net do not use the `hgt` sampler; PSHGCN (S3) and HINormer (S2) stop.
