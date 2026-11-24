# OOM layer 1: theoretical GPU-memory screen of the baseline cells at the smallest grid point

Read-only analysis. Code: `/mnt/data/peeradon/h2snn-runs/f1fc158409ad7d122ee67c42c897208282b9fd65/H2S-NN` (paths below
are relative to it). No job was submitted.

**Device.** H100 94 GB = 95,830 MiB = **93.58 GiB total** (as recorded in the docs); host 96 GB = 89.4 GiB. All
numbers below are GiB (2^30 B), float32 = 4 B, int64 index = 8 B.

**Smallest grid point.** `src/baselines/campaign.py:179-191` (`smallest_grid_point`): numeric list -> min,
distribution -> `min`, non-numeric list (incl. booleans) -> **first value**. So `add_lp`/`add_hp` (NSD) = True,
`conformal` (CSNN) = True, `mask_lp` (Hetero2Net) = True, `residual` (SeHGNN) = True, `metapath_pe` (H2G-former,
where the axis exists) = True, `sheaf` (HetSheaf) = `diag/local_concat`, `sheaf_model` (NSD) = `DiagSheaf`,
`gnn_type` (CSNN) = `GCN`. `batch_size` on H2GB NC = 32 (`src/configs/config.py:109`). lr = 1e-4, weight decay = 0,
dropout = 0 everywhere.

**Cell set.** The four 0b9873d list files and `e52b025/cells_tuning_early.txt` are **not on the lab filesystem**
(`/mnt/data/peeradon/h2snn-out/` has only the 0b9873d dir, without those files; no e52b025 dir). They were read
(read-only `grep`) from the TSUBAME work dir, where `t4_auto.py` keeps them. In scope after removing the deferred
datasets (mag-year, oag-chem, freebase_bb, ogbn-mag_pp): **88 cells**. MAGNN has no line in these lists (its H2GB NC
and large-LP cells are `probe` cells in `scripts/tsubame/make_cell_list.py:38, 299-302`); see the MAGNN note.
NC acm/dblp/imdb lines (h2gformer x3, hetsheaf imdb, early csnn/hetsheaf acm/dblp) are outside the requested datasets.

**Graph sizes** (records `nc__*__h2.json`, `lp__*__mde.json`; feature widths from
`/mnt/data/peeradon/h2snn-card19/docs/partM3b_evidence/b2_counts_*.json`, `b5_pdns_check.json`):

| dataset | V | E (stored, incl. reverses) | features |
|---|---|---|---|
| freebase (HGB) | 180,098 (music 82,351; book 40,402) | 1,531,157 | none (all 8 types featureless) |
| ogbn-mag | 1,939,743 | 42,182,144 | all types 128 |
| oag-cs | 1,112,691 | 27,537,448 | all 768 |
| oag-eng | 929,315 | 12,346,854 | all 768 |
| ieee-cis-g | 153,880 | 2,873,472 | transaction 4,785; 11 types featureless (max 8,499 nodes) |
| h-pokec | 1,731,977 (user 1,632,803) | 51,774,836 (friend 44.6M) | user 65; 15 types featureless (max 22,067) |
| pdns | 1,173,558 | 76,797,104 | domain 10, ip 2 |
| h-pokec_uu (message graph) | 1,731,977 | 43,300,090 | as h-pokec |
| pubmed / acm_pp / dblp_ap / imdb_ma / lastfm | 63k / 11k / 26k / 21k / 21k | 358k / 546k / 232k / 81k / 206k | HGB |

**Mini-batch shapes** (`src/baselines/minibatch.py`, H2GB YAMLs `vendor/H2GB/configs/<ds>/<ds>-GCN.yaml`): H2GB NC =
H2GB `hgt` sampler, B = 32 seeds, per type per iteration: ogbn-mag 512 x 6, oag/IEEE 64 x 6, PDNS 256 x 4, Pokec
user [2048, 2048, 512, 512] + 128 x 3 for the 15 other types. Sampled subgraph <= 12.3k nodes (ogbn-mag), ~10.9k
(Pokec), <= 4.6k (IEEE), ~2k (PDNS), ~1.5k (OAG). Large LP (`src/baselines/large_lp_training.py:49-60, 171-179`): one
step = 8,192 positives + 8,192 negatives -> up to 32,768 unique endpoints as hgt seeds, plus the Pokec budget
(~5.1k users + 5.8k others) -> ~40k-node subgraph.

Factor convention: training peak ~= persistent tensors + saved activations x ~2 (forward + gradient) + optimizer
(AdamW 2 x params; params are small everywhere here).

## (a) Table

Verdict: CERTAIN_OOM (arithmetic > 93.58 GiB), LIKELY_FITS (< ~30 GiB), UNCERTAIN (probe). Host column flags a
host-RAM risk against 89.4 GiB.

| model | dataset | task | batch mode | dominant term | est. GiB (GPU) | verdict | evidence (file:line) |
|---|---|---|---|---|---|---|---|
| chebnetii | freebase | NC | full batch (HGB) | per-type proj. + K=5 bases, N x 64 | < 2 | LIKELY_FITS | `adapters/chebnetii.py:136-146`; 24 GB dry run OK 1.32 GiB at max |
| csnn | freebase | NC | full batch | E x 2 x hc x d per-edge maps | ~2-5 | LIKELY_FITS | `adapters/csnn.py:194-208`; 24 GB: OK 19.6 GiB at max with d 3 |
| h2sgnn | freebase | NC | full batch | book-X-book meta-path operators (<= 40,402^2 each) | < 27 | LIKELY_FITS | `adapters/h2sgnn.py:241-281`, `_hgb_generic.py:116-133`; CPU tests of the same inputs peak 26.9 GiB host |
| hetero2net | freebase | NC | full batch | dense `torch.eye` for all 8 types on GPU: sum n_t^2 x 4 B = 34.46 GiB | ~36-40 | UNCERTAIN (expected fit) | `vendor/Hetero2Net/hetero2net/dataset.py:48-53`, `adapters/hetero2net.py:264` |
| hetsheaf | freebase | NC | full batch | E x 2 x (hc*d=16) learner input | < 2 | LIKELY_FITS | `adapters/hetsheaf.py:529-551` |
| hinormer | freebase | NC | full batch | GCN N x 64 x 3 layers + seq attention 40k x 2 x 20^2 | < 2 | LIKELY_FITS | `adapters/hinormer.py:197-268`, `vendor/HINormer/model.py:198-215` |
| nsd | freebase | NC | full batch | E_dir(~2.1M) x 2 x hc(32) x final_d(4) x 4 B = ~1.9 GiB / layer | ~3-6 | LIKELY_FITS | `adapters/nsd.py:244-260, 163-197`; 24 GB OOM at d 4 (6.97 GiB alloc) scales to ~1.9 GiB here |
| pshgcn | freebase | NC | full batch | K=1: ~#keys x N x emb 16 (spmm) | < 3 | LIKELY_FITS | `adapters/pshgcn.py:189-202`, `_hgb_generic.py:89-113` |
| **h2gformer** | **freebase** | NC | full batch | eye copies: `data.to(device)` 34.46 + `pristine_x` clone 34.46 + first-forward clone of music 25.26 = **94.18 GiB live** | **>= 94.2** | **CERTAIN_OOM (thin margin)** | `adapters/h2gformer.py:558-568, 295-302` |
| chebnetii | ogbn-mag | NC | index batches (bases precomputed on host) | host: (K+1) x n_paper x 128 x 4 = 2.1 GiB | < 1 | LIKELY_FITS | `adapters/chebnetii.py:265-273`, `_chebnetii_h2gb.py:58-75` |
| csnn | ogbn-mag | NC | hgt B=32 (512x6) | subgraph <= 12.3k nodes | < 3 | LIKELY_FITS | `minibatch.py:524-570`, `adapters/csnn.py:395-405` |
| h2sgnn | ogbn-mag | NC | hgt B=32 | P-X-P operators on <= 3.1k papers (<= 0.2 GiB each) | < 3 | LIKELY_FITS | `adapters/h2sgnn.py:457-512` |
| hetsheaf | ogbn-mag | NC | hgt B=32 | subgraph; chain on whole graph on host | < 3 | LIKELY_FITS | `adapters/hetsheaf.py:22-30, 666, 763` |
| nsd | ogbn-mag | NC | hgt B=32 | subgraph edges x 2 x 32 x 4 | < 3 | LIKELY_FITS | `adapters/nsd.py:467-477`, `minibatch.py:524-570` |
| pshgcn | ogbn-mag | NC | index batches (their MAG code) | host: K=2 propagation 8.1 GiB + labels 1 GiB/key | < 1 | LIKELY_FITS | `adapters/_pshgcn_mag.py:135-173` |
| sehgnn | ogbn-mag | NC | index batches | host: hg_propagate num_hops 2: 8.1 GiB | < 1 | LIKELY_FITS | `adapters/_sehgnn_h2gb.py:91-120`, `vendor/SeHGNN/ogbn/utils.py:49-80` |
| chebnetii | oag-cs | NC | index batches | host: bases 6 x 546,704 x 768 x 4 = 9.4 GiB (+ ~10 GiB transient) | < 1 | LIKELY_FITS | `_chebnetii_h2gb.py:58-75` |
| csnn | oag-cs | NC | hgt B=32 (64x6) | subgraph ~1.5k nodes | < 1 | LIKELY_FITS | `minibatch.py:524-570` |
| h2sgnn | oag-cs | NC | hgt B=32 | subgraph | < 1 | LIKELY_FITS | `adapters/h2sgnn.py:470-512` |
| hetero2net | oag-cs | NC | NeighborLoader [10,10], B=32 | subgraph, 768-d x | < 1 | LIKELY_FITS | `_hetero2net_h2gb.py:35, 109-121` |
| hetsheaf | oag-cs | NC | hgt B=32 | subgraph | < 1 | LIKELY_FITS | `adapters/hetsheaf.py:763` |
| hinormer | oag-cs | NC | index batches, full-graph GCN every step | X 3.2 + DGL graph ~1.3 + 3 x N x 64 activations ~3 | ~8 | LIKELY_FITS | `adapters/hinormer.py:287-317`, `vendor/HINormer/model.py:198-215` |
| nsd | oag-cs | NC | hgt B=32 | subgraph | < 1 | LIKELY_FITS | `minibatch.py:524-570` |
| pshgcn | oag-cs | NC | index batches (decoupled) | host: K=2 propagation peak 22.0 GiB (target store 17.2) | < 1 | LIKELY_FITS | `_hgb_generic.py:321-350`, `vendor/PSHGCN/ogbn-mag/utils.py:107-132` |
| sehgnn | oag-cs | NC | index batches | host: num_hops 2 peak 22.0 GiB | < 1 | LIKELY_FITS | `_sehgnn_h2gb.py:105-120` |
| (same 9 models) | oag-eng | NC | as oag-cs | as oag-cs; host pshgcn/sehgnn 16.4 GiB, chebnetii bases 6.4 GiB; hinormer ~6 GiB | < 8 | LIKELY_FITS (x9) | as oag-cs |
| chebnetii | ieee-cis-g | NC | full-graph bases per forward (`ForwardBases`) | X 2.6 + K=5 x N x 64 | ~4 | LIKELY_FITS | `adapters/chebnetii.py:242-255`, `_chebnetii_h2gb.py:144-190` |
| csnn, h2sgnn, hetsheaf, nsd | ieee-cis-g | NC | hgt B=32 (64x6) | subgraph <= 4.6k nodes (one-hot rows <= 384 x 8,499) | < 2 | LIKELY_FITS (x4) | `minibatch.py:499-512, 524-570` |
| hetero2net | ieee-cis-g | NC | NeighborLoader [10,10] | host eye 0.27 GiB | < 2 | LIKELY_FITS | `_hetero2net_h2gb.py:60-70` |
| hinormer | ieee-cis-g | NC | full-graph GCN every step | X 2.6 GiB + small graph | ~4 | LIKELY_FITS | `adapters/hinormer.py:287-317` |
| pshgcn | ieee-cis-g | NC | index batches | host: 32.5 GiB (11 card types x 144k x 4,785 back to transaction) | < 1 | LIKELY_FITS (host ~35-45) | `_hgb_generic.py:321-350` |
| sehgnn | ieee-cis-g | NC | index batches | host: 32.5 GiB, 23 keys, 60,236 floats per row | < 1 | LIKELY_FITS (host ~35-45) | `_sehgnn_h2gb.py:105-120` |
| (all 9) | pdns | NC | hgt B=32 / index batches / full-graph GCN (hinormer) | widths 10/2; hinormer: DGL graph 76.8M edges ~3.5 GiB | < 6 | LIKELY_FITS (x9) | `adapters/chebnetii.py:256-264` (projected bases), `adapters/hinormer.py:287-317` |
| chebnetii | h-pokec | NC | full-graph bases per forward | Laplacian ~52M nnz x 20 B ~1 GiB (+ transpose) + 6 x N x 64 x 4 x ~2 | ~8-10 | LIKELY_FITS | `_chebnetii_h2gb.py:164-190` |
| csnn, nsd, gat, h2sgnn, hetsheaf | h-pokec | NC | hgt B=32, Pokec dict | subgraph ~10.9k nodes; h2sgnn U-X-U blocks <= 5.2k^2 x 20 B x 16 paths <= 8.5 GiB worst case | < 10 | LIKELY_FITS (x5) | `adapters/h2sgnn.py:457-467`, `_hgb_generic.py:116-133` |
| hetero2net | h-pokec | NC | NeighborLoader [10,10], B=32 | host eye 5.7 GiB; batch rows only | < 3 | LIKELY_FITS | `_hetero2net_h2gb.py:60-70` |
| hinormer | h-pokec | NC | full-graph GCN every step | 3 x N x 64 x ~4 tensors ~5 + DGL graph ~2.5 | ~8-10 | LIKELY_FITS | `adapters/hinormer.py:287-317` |
| pshgcn | h-pokec | NC | index batches (decoupled, K fixed 2) | **host**: 1.63M users x 8,850 floats x 4 = 53.8 GiB stored + graph copies | < 2 | UNCERTAIN (host ~60-65 of 89 GiB) | `adapters/pshgcn.py:181-188, 411-436`, `_hgb_generic.py:303-350` |
| csnn, gat, gcn, nsd, hetsheaf | h-pokec_uu | LP | hgt, ~32k seeds per step | ~40k-node subgraph, ~0.5-1M edges | < 6 | LIKELY_FITS (x5) | `large_lp_training.py:49-60, 212-224`, `adapters/nsd.py:200-211` |
| han | h-pokec_uu | LP | hgt (force), H2GB HAN | relation-level HANConv on the subgraph (no meta-path graph) | < 6 | LIKELY_FITS | `adapters/hgb_han.py:407-428` |
| rgcn | h-pokec_uu | LP | hgt (Pokec RGCN yaml) | hidden 40, bases 4, subgraph | < 4 | LIKELY_FITS | `adapters/rgcn.py:383-413` |
| h2gformer | h-pokec_uu | LP | hgt (force), RAW encoder | SparseNodeTransformer (edge attention), hidden 64 | < 8 | LIKELY_FITS | `adapters/h2gformer.py:570-595`, `vendor/H2GB/configs/Pokec/Pokec-H2Gformer.yaml:68` |
| h2sgnn | h-pokec_uu | LP | hgt, ~32k seeds | user-X-user meta-path products over ~37k users; hub interest values make blocks of (deg x 2.3 %)^2 | 5-40 (wide) | UNCERTAIN | `adapters/h2sgnn.py:515-530, 457-467`, `_hgb_generic.py:116-133` |
| hetero2net | h-pokec_uu | LP | LinkNeighborLoader [10,10] from 16,384 links | 2-hop of ~32k endpoints on the friend graph (up to ~1M users) + eye rows of sampled interest nodes (<= 5.7 GiB) | 10-30 (wide) | UNCERTAIN | `adapters/hetero2net.py:616-644`, `_hetero2net_h2gb.py:35, 60-70` |
| pshgcn | h-pokec_uu | LP | rows of the decoupled features | **host** 53.8 GiB (as NC, message graph); GPU ~2 GiB per step (32k rows x 8,850 floats) | ~3 | UNCERTAIN (host) | `adapters/pshgcn.py:447-461` |
| gat | acm_pp, pubmed | LP | full batch (HGB LP) | E x heads x hidden attention at heads 4, hidden 64 | < 12 | LIKELY_FITS (x2) | `adapters/hgb_gnn.py:127-140`; 24 GB: OOM only at max |
| h2gformer | acm_pp, dblp_ap, imdb_ma, lastfm | LP | full batch | MetaPath2Vec + GTModel; 24 GB OOM needed one more 5.19 GiB at ~19 GiB | ~25 | LIKELY_FITS (x4) | `adapters/h2gformer.py:209-237`; dry run cells.csv (5.19 GiB alloc) |
| hetsheaf | acm_pp, dblp_ap, imdb_ma, lastfm, pubmed | LP | full batch | diag, d 2, layers 2, hc 8: E x 2 x 16 floats | < 2 | LIKELY_FITS (x5) | `adapters/hetsheaf.py:529-551`; lab fit check OOM only at max (layers 8, d 5, hc 64) |
| nsd | acm_pp, dblp_ap, lastfm, pubmed | LP | full batch | E x 2 x 32 x 4 floats, 1 layer | < 6 | LIKELY_FITS (x4) | `adapters/nsd.py:244-264`; 24 GB: pubmed OK at d 4 (21.7 GiB) |
| pshgcn | pubmed | LP | full batch | K=1 monomials (16 keys) | < 4 | LIKELY_FITS | `adapters/pshgcn.py:156-173`; 24 GB: OK at K 2 (10.9 GiB) |

Counts (88 cells): **CERTAIN_OOM 1, UNCERTAIN 5, LIKELY_FITS 82.**

## (b) Per-model notes and formulas

**H2G-former, Freebase NC (CERTAIN_OOM, thin margin).** `_build_hetero_data` gives every featureless type
`torch.eye(n_t)` and returns `data.to(device)` (`h2gformer.py:558-568`): sum_t n_t^2 x 4 B = (40,402^2 + 19,427^2 +
82,351^2 + 1,025^2 + 17,641^2 + 9,368^2 + 2,731^2 + 7,153^2) x 4 B = **34.46 GiB** (music alone 25.26 GiB, the tensor
the 24 GB dry run failed on). `pristine_x = {nt: x.detach().clone()}` (`:295-298`) keeps a second copy on the GPU
(68.92 GiB). The first `forward()` runs `data[nt].x = value.clone()` (`:302`): the clone is allocated before the old
tensor is released, so at the music step 34.46 + 34.46 + 25.26 = **94.18 GiB** are live > 93.58 GiB device total.
No grid axis touches these tensors (hidden, heads, layers, `metapath_pe` all act after them), so the smallest point
does not help. Margin is 0.6 GiB; the steady state after the first forward would be ~69 GiB. A 1-epoch probe is
cheap and would confirm it. (Adapter-level cause: a sparse identity or an embedding table instead of a dense eye
would remove it; not a model property.)

**Hetero²Net, Freebase NC (UNCERTAIN, expected fit).** Their FreeBase branch, `torch.eye` per type
(`vendor/Hetero2Net/hetero2net/dataset.py:48-53`), moved once by `data.to(device, "x", "y")` (`hetero2net.py:264`):
34.46 GiB, no second copy (`train_full_batch` moves only `edge_index`, `utils.py:131-151`). Projections
`Linear(n_t, 16)` keep x as their saved input (no extra copy); activations N x 16 are MB-scale. Estimate 36-40 GiB.
Host: 35.3 GiB peak measured at the 40G retry (dry-run report `d6_ranges.md:39`).

**PSHGCN (generic decoupled branch: oag-cs/eng, ieee, pdns, h-pokec, h-pokec_uu) and SeHGNN (ogbn-mag, oag,
ieee, pdns).** Both precompute on the host (`_hgb_generic.py:321-350` -> `vendor/PSHGCN/ogbn-mag/utils.py:107-132`;
`_sehgnn_h2gb.py:105-120` -> `vendor/SeHGNN/ogbn/utils.py:49-80`). Two hops, mean over each (src, dst) type pair:
hop 1 key on dst of width w(src) for every pair; hop 2 only into the target: for every pair (s, target), every hop-1
key on s. Stored per target = N_target x (w_t + sum hop-1 widths into t + sum hop-2 widths) x 4 B. Featureless types:
random 256-wide inputs (PSHGCN `RANDOM_INPUT_WIDTH`; SeHGNN `embed_size` min 256). Results (script in
`calc.py` beside this file):

| dataset | host peak (raw + hop1 + hop2) | target store | keys | floats per target row |
|---|---|---|---|---|
| ogbn-mag (SeHGNN; PSHGCN MAG code) | 8.1 | 5.3 | 10 | 1,920 |
| oag-cs | 22.0 | 17.2 | 11 | 8,448 |
| oag-eng | 16.4 | 11.7 | 11 | 8,448 |
| ieee-cis-g | 32.5 | 32.4 | 23 | 60,236 |
| pdns | 0.2 | 0.1 | 6 | 44 |
| h-pokec (PSHGCN only; SeHGNN is N/A) | **54.0** | **53.8** | 48 | 8,850 |

GPU holds only B rows of each key (B = 32; large LP: ~32k endpoint rows x 8,850 floats = 1.1 GiB per step) and the
per-key MLPs. So GPU always fits; **H-Pokec (NC and U-U) is a host-RAM question**: 53.8 GiB of stored features plus
the DGL heterograph, PyG graph, `torch.unique` copies in `plan_relations` and process baseline -> ~60-65 GiB of 89.4
GiB. Not certain either way -> probe. Note: PSHGCN's H-Pokec restriction "K = 1, hidden <= 64"
(`search_grids.py: DATASET_OVERRIDES`, spec 1.23) is **not reached** by this code path: `pshgcn.py:181-188` returns
the generic space before `space.update(dataset_override(...))` at `:202`, and the decoupled branch fixes K = 2
(`_hgb_generic.DECOUPLED_K`). Worth a line to the model session.

**ChebNetII.** MAG/OAG: bases T_0..T_K on the host, kept for target rows only: (K+1) x n_target x F x 4 B (K = 5:
ogbn-mag 2.1, oag-cs 9.4, oag-eng 6.4 GiB) plus ~3 full N x F tensors during the recursion
(`_chebnetii_h2gb.py:58-75`). PDNS: per-type block bases, widths 11/3 (`:122-140`). IEEE and H-Pokec: `ForwardBases`
(`:164-190`), the whole graph every forward on the GPU: sparse Laplacian (nnz ~ E, 20 B each, plus its transpose for
backward) + (K+1) x N x hidden_dim(64) x 4 B activations x ~2 -> ~4 GiB (IEEE, incl. 2.6 GiB dense transaction
features), ~8-10 GiB (H-Pokec). Fits; but every evaluation batch of 32 recomputes the whole-graph bases (time risk,
not memory).

**HINormer (H2GB).** `_fit_h2gb` (`hinormer.py:287-317`) keeps all features and the DGL graph on the GPU and runs
`num_gnns` (min 3) full-graph GCN + RE layers per step (`vendor/HINormer/model.py:198-215`): X + DGL graph (~16-48 B
per edge with formats) + 3 x ~4 x N x 64 x 4 B. oag-cs ~8, oag-eng ~6, h-pokec ~8-10, pdns ~5, ieee ~4 GiB. Sequence
attention: B x heads(2) x len_seq(20)^2, negligible. Time risk: `build_node_sequences_for` loops in Python over every
target (1.63M users on H-Pokec, 546k papers on oag-cs).

**Class C on the hgt sampler (NSD, CSNN, H2SGNN, HetSheaf, and GAT/GCN class A).** `fit_homogeneous_on_hgt`
(`minibatch.py:524-570`) / H2SGNN `batch_forward` (`h2sgnn.py:457-467`) / HetSheaf rebinding (`hetsheaf.py:763`):
memory scales with the sampled subgraph, not the graph. Featureless types become dense one-hot rows of the sampled
ids only (`minibatch.py:499-512`): rows x n_type, at most 384 x 22,067 x 4 B = 34 MB per type here. NSD per layer:
E_sub x 2 x hidden_channels x final_d x 4 B (final_d = d + add_lp + add_hp = 4 at the smallest point). H2SGNN: one
sparse target-X-target operator per meta-path, nnz <= n_target_in_batch^2 (`_hgb_generic.py:116-133`); bounded at
B = 32 for NC (<= 8.5 GiB worst case on H-Pokec), but on H-Pokec U-U the ~37k users of a step and Pokec's
hub interest values (users listing the same value) make the U-X-U blocks large: a value listed by D users contributes
about (0.023 D)^2 nnz x 20 B (D = 300k -> 0.9 GiB from one value), summed over the sampled hub values of 15 types;
spspmm also materialises the product before the target crop. Wide range -> probe.

**HetSheaf (full batch HGB and the small LP tasks).** Smallest point: `diag/local_concat`, d 2, layers 2,
hidden_channels 8. The lab fit-check OOMs (acm_pp, dblp_ap, imdb_ma, lastfm, pubmed) were at the largest point
(layers 8, d 5, hc 64, general/attention maps): the per-edge learner input E x 2 x hc x d drops by ~20x and the d^2
maps by ~6x, and depth by 4x.

**Hetero²Net on H-Pokec U-U.** `Sampler("link", [10, 10])` (`hetero2net.py:625`): `LinkNeighborLoader` from the
step's 16,384 links samples 10 neighbours per relation per hop for ~32k endpoints: hop 1 ~ 32k x (10 friends + listed
values) users/values, hop 2 ~ up to ~1M distinct users (of 1.63M). Each sampled interest node carries its dense eye
row (`_hetero2net_h2gb.py:69`, <= 5.7 GiB if every value of the 15 types is sampled). Activations ~N_sub x 64 x
(2 channels x 2 layers x ~3). Estimate 10-30 GiB, wide -> probe.

**HAN, R-GCN, H2G-former on H-Pokec U-U.** H2GB's HAN is relation-level `HANConv` (no meta-path graph); R-GCN typed
linear on subgraph edges; H2G-former `SparseNodeTransformer` (edge attention). All subgraph-bound.

**MAGNN (not in these lists).** Its H2GB NC / large-LP cells are `probe` cells (`make_cell_list.py:38, 299-302`) and
were left out of the b2 lists ("MAGNN (host RAM)", `t4_auto.py`). `build_large_lp` (`magnn.py:663-700`) enumerates
t-X-t instances inside each sampled subgraph (`schema_lists`, host): instances ~ sum over sampled X of
deg_sub(X)^2, the same hub problem as H2SGNN on U-U but stored per instance (x path length) -> a host-RAM risk on
H-Pokec. Not estimated further.

## (c) Probe list (1 epoch, `--dry_run --n_trials 1 --grid_point min`)

The probe line the launcher builds (`make_cell_list.py:299-302`) fixes every axis at `smallest_grid_point`; the
values that point resolves to:

| # | cell | smallest-grid values | what to watch |
|---|---|---|---|
| 1 | h2gformer / freebase / NC (CERTAIN_OOM, confirm) | lr 1e-4, weight_decay 0, hidden_dim 64, dropout 0, attn_dropout 0, gt_layers 2, attn_heads 4, metapath_pe True (if the axis exists) | CUDA OOM in the first forward at ~93.6 GiB (clone of `music`) |
| 2 | hetero2net / freebase / NC | lr 1e-4, weight_decay 0, hidden 16, dropout 0, alpha 0.0, beta 0.2, p 0.6, r 0.3, mask_lp True | GPU peak ~36-40 GiB; host ~35 GiB |
| 3 | pshgcn / h-pokec / NC | lr 1e-4, weight_decay 0, emb_dim 128, hidden_x 256, layers_x 2, dropout 0, input_drop 0, batch_size 32 (K fixed 2 in code) | **host** MaxRSS (expect 60-65 GiB of 89.4) |
| 4 | pshgcn / h-pokec_uu / LP | lr 1e-4, weight_decay 0, emb_dim 128, hidden_x 256, layers_x 2, embed_dim 16, dropout 0, input_drop 0, decoder_type dot | **host** MaxRSS; GPU ~3 GiB |
| 5 | h2sgnn / h-pokec_uu / LP | lr 1e-4, weight_decay 0, hidden 16, dropout 0, emb_dim 64, embed_dim 16, input_drop 0, K 2, poly GPR_GPR, alpha 0.1, alpha_list 0.1, temp_lr 1e-3, temp_wd 0.01, beta_lr 1e-5, beta_wd 0.0, decoder_type dot | GPU and host peak of the U-X-U operators over ~37k users |
| 6 | hetero2net / h-pokec_uu / LP | lr 1e-4, weight_decay 0, hidden_channels 64, embed_dim 16, dropout 0, num_layers 2, alpha 0.0, decoder_type dot | GPU peak of a [10, 10] link batch of 16,384 links |

Optional (LIKELY_FITS but > 1/3 of the device or a host spike): sehgnn and pshgcn on ieee-cis-g (host ~35-45 GiB).

## Not determined

* Exact homogeneous edge counts after HGB's symmetrisation for Freebase (used ~2.1M directed); exact nnz of the
  H2SGNN meta-path operators (Freebase NC bounded by the 26.9 GiB CPU test peak; H-Pokec U-U not bounded).
* Real induced-edge counts of hgt subgraphs (estimates assume ~0.5-1M edges at ~40k nodes for U-U).
* Whether `freebase` has the H2G-former `metapath_pe` axis (`has_walk_schema`); it does not change verdict 1.
* The 1,214 GiB figure quoted for SeHGNN on H-Pokec (Part M3b B7) does not match this code's `hg_propagate` at
  num_hops 2 with embed_size 256 (54 GiB by the same arithmetic as PSHGCN); SeHGNN H-Pokec is N/A in the registry and
  not in scope, but the quoted number may come from a different construction.
