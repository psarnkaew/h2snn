# Baselines on the H2GB datasets, part 2 (card 1b) — report (slim mode)

Card `docs/H2S-NN_baselines_h2gb_part2_card.md`; branch `baselines-h2gb` at `v3b` `6ba6dec00369ff590b341ce3771da90528fcc1cd`,
then `git merge --no-ff 726ec849d2a644b50e4a0ff22d5a37c753ab7075` (clean). All jobs: SLURM allocation 16741 on g2n01 via
`srun --overlap`, env `h2snn-b`; scripts and logs in `docs/baselines_h2gb_evidence/` (prefix `p2_`). No training run on
real data, no metric. Part 2 changed no `src/` file outside `src/baselines/`, so the bit-exact compares were not required.

**Verdict.** 58 more H2GB node-classification cells are runnable: GCN, GAT, HAN, NSD, CSNN (all 8 datasets), H2G-former
(7), SeHGNN (5), ChebNetII (6). Three decisions contradict the vendor code (E0 stops: HINormer, PSHGCN, H2SGNN), and two
cells are not runnable under H2GB's own code or the C4-c rule. The large-LP training path is written and tested for
GCN/GAT, but claiming those cells regenerates the lane notebooks, which are outside this card's files (stop item). No
large-LP cell is runnable yet.

## Units
| Unit | Commits | Jobs (16741 step) | State |
|---|---|---|---|
| E0 preconditions | `b3fa508` | — | done; stops H1, P1, S4b; SeHGNN MAG embeddings need training (`p2_e0_preconditions.md`) |
| E1 class A | `c0b43b7` (rename only, see deviations), `fde69e2` | 85-87 tests, 88-89 mutation, 90 registry | done; H2G-former mag-year stop item |
| E2 class B | `7a50b9c` | 91 | partly: SeHGNN, ChebNetII done; PSHGCN (P1) stop; Hetero²Net not wired |
| E3 class C | `fd06ade` | 92 | NSD, CSNN done; H2SGNN (S4b) stop |
| E4 large LP | `fb93276` | 93-94 tests, 95 mutation | infrastructure + GCN/GAT path; registry wiring blocked (notebooks) |
| E5 suite + report | `de461cf` (prep), this commit | suite after 95 | done |

## Cell table — H2GB node classification
R = runnable; W = waiting for the author (named file or training); S = stop item (reason below); — = N/A.

| Method | ogbn-mag | mag-year | oag-cs | oag-eng | oag-chem | ieee-cis-g | h-pokec | pdns |
|---|---|---|---|---|---|---|---|---|
| GCN, GAT | R | R | R | R | R | R | R | R |
| HAN (S5) | R | R | R | R | R | R | R | R |
| H2G-former | R | S (a) | R | R | R | R | R | R |
| HINormer | S (H1) | S | S | S | S | S | S | S |
| SeHGNN | W (b) | W (b) | R | R | R | R | — (author) | R |
| PSHGCN | S (P1) | S | S | S | S | S | S | S |
| ChebNetII | R | R | R | R | R | S (c) | S (c) | R |
| Hetero²Net | W/S (d) | W/S (d) | S (d) | S (d) | S (d) | S (d) | S (d) | S (d) |
| NSD, CSNN | R | R | R | R | R | R | R | R |
| H2SGNN | S (S4b) | S | S | S | S | S | S | S |

(a) H2GB's mag-year H2Gformer config enables MetaPath2Vec, but `preprocess_Metapath` has no schema for the name
`mag-year` (`encoding_generator.py:187-436`; `mag_dataset.py:67`): H2GB's own code cannot run it.
(b) SeHGNN's published MAG inputs are ComplEx embeddings trained with `dglke` under `dgl<1.0` (`SeHGNN/ogbn/Readme.md`); no
download exists; not trained here (decision C4-a).
(c) C4-c's per-block precomputation `T_k(L)[:, block] X_t` is a dense N × N_type matrix per k for a featureless (identity)
type: IEEE-CIS-G and H-Pokec have such types.
(d) Not wired: its metapath-drop needs a metapath list per dataset, which its code has for HGB, MAG and RCDD only
(`hetero2net/dataset.py:20-67`); its label channel `y_emb` on sampled batches needs a route test; ogbn-mag/mag-year also
wait for `mag_metapath2vec_emb.zip` (part 1 report, `c4_hetero2net_download.md`).

## Cell table — large LP
| Method | ogbn-mag P–P | H-Pokec U–U |
|---|---|---|
| GCN, GAT | S: code path and tests in place (`large_lp_training.py`, `hgb_gnn._fit_large_lp`); claiming the cell regenerates `notebooks/baselines/` (outside card 1b's files) | S: H2GB's per-node-type fan-out (user [2048, 2048, 512, 512], clubs [128 × 3]) has no per-relation form for PyG's link sampler |
| NSD, CSNN, HAN, H2G-former | not wired in this card | not wired |
| PSHGCN / H2SGNN / Hetero²Net | S (P1) / S (S4b) / not wired | same |
| ChebNetII, SeHGNN, HINormer | — (no LP in source) | — |

## E0 stops (decision vs vendor code; `p2_e0_preconditions.md`)
- **H1 HINormer (S2).** H2GB's `HINormerModel` has its transformer and feature encoder commented out
  (`hinormer_model.py:221-222, 274-289`): a linear head on the ego node after full-graph GCN layers. Its `hin` loader gives
  the whole split as one step (`custom_sampler.py:678-685`) and caches at `/nobackup/users/junhong/Data` (`:618-646`).
  Configs exist for 4 of 8 datasets.
- **P1 PSHGCN (S3 (b)).** Besides the widths, its ogbn-mag code enumerates the polynomial terms from MAG's relation list
  (`ogbn-mag/processing.py:14-15`) and fixes MAG's label keys (`main.py:126`). Their self-effect removal needs
  `sparse_tools` diagonals (`processing.py:133-151`), and `sparse_tools` is not installed. Its model and `propagate` would run unchanged
  (target as letter `P`, `model.py:96-120`, `utils.py:111-133`).
- **S4b H2SGNN (S4).** Its operators come from one hand-written branch per HGB dataset (`H2SGNN/processing.py:46-121`); no
  rule exists for the H2GB graphs.

## Scheme deltas (part 1 table → part 2)
- Class A/C: H2GB's YAML lists unchanged (S1), via H2GB's `get_HGTloader`. Class A: GCN/GAT are HGB's models on HGB's
  reduction of each batch; HAN and H2G-former are H2GB's own `HeteroGNNModel`/`GTModel` from H2GB's YAML (`assert_cfg`
  as H2GB's `load_cfg`). Class C (NSD, CSNN) uses the dataset's GCN list. Featureless types get one-hot rows, i.e.
  HGB's identity encoding.
- H2G-former: H2GB's own `preprocess_Metapath` (its `test()` only prints; the embedding after 5 fixed epochs is used), cached
  outside `data/`; label channel training-only, set after the loaders are built; true labels passed to the loop.
- ChebNetII on PDNS: per-type projection after per-block bases (C4-c).
- Large LP: steps of 8,192 positives with per-epoch HGB tail negatives; link-level sampling per step on the message graph;
  each evaluation endpoint computed once by node-level sampling; HGB's 2-hop and random test sets; 40/40.

## Tests (slim) and mutations
| Check | Job | Result |
|---|---|---|
| GCN full-fan-out equivalence at depth = steps + 1 (moved into `tests/`, HAN removed) + negative control | 87 | OK |
| One step finite and moves: GCN, GAT, HAN, H2G-former | 87 | OK |
| H2G-former label route (permuting val/test labels changes no output) | 87 | OK |
| Mutation: label channel reads all labels | 89 | caught (max \|diff\| 0.75). Step 88 ran the unmutated file (NFS lag) and passed; not counted |
| ChebNetII: per-block bases then projection = projection then propagation (float64, 1e-10); one step | 91 | OK |
| NSD, CSNN one step | 92 | OK |
| Held-out facts over every large-LP adapter's loaders (step and evaluation), one LP step for GCN, GAT | 94 | OK |
| Mutation: full graph instead of the message graph | 95 | caught (10 / 27 / 27 held-out facts) |
| PSHGCN equivalence | — | not run (P1 stop) |

## E5 — full suite
`scripts/slurm/run_tests.sbatch` at `de461cf` on g2n01 (job 16741, the step after 95; 2080 Ti visible, the CUDA test ran): **Ran 696 tests, OK** (`p2_e5_suite_g2n01_16741.txt`) = part 1's 684 + 12 new (`test_baselines_h2gb_classa` 5, GCN equivalence moved into `tests/` 1, ChebNetII 2, `test_baselines_h2gb_classc` 2, large LP 2). Existing tests pass unchanged; `test_baselines.py` has only its cell counts updated (runnable 141 → 193, NC 51 → 103).

## Deviations
- **E1 split over two commits.** `c0b43b7` holds only the file rename: a failed `git add` (a path already moved) left the
  rest unstaged, and its message describes the whole unit. `fde69e2` carries the content. No amend, as the rules require.
- **`tests/test_h2gb_nc.py`** (outside card 1b's files) was changed in E2 and restored to `6ba6dec` in `de461cf`; the
  SeHGNN MAG reason is worded so the existing rule accepts it.
- The shared loop takes the true labels separately (`labels=`), because H2G-former's label channel rewrites `y` in the graph.
- Binary datasets: H2GB's `create_model` uses one logit and BCE (`model_builder.py:22-23`); here 2 logits and the shared
  cross-entropy (one loss rule for every baseline).
- Part 1's `c1_equivalence_diagnostic.py` imports the moved test module; its HAN part referred to the removed model (kept as
  history: `git show 6ba6dec:docs/baselines_h2gb_evidence/c1_equivalence_tests.py`).
- Repeated NFS lag: a job started within seconds of an edit can import the old file (steps 86 and 88).

## Decisions / stop items for the author and the design session
| # | Item | Options |
|---|---|---|
| 1 | Lane notebooks for the large-LP cells (E4) | regenerate `notebooks/baselines/` via `scripts/generate_baseline_notebooks.py` (another card, or allow it here); then flip GCN/GAT on ogbn-mag P–P |
| 2 | H-Pokec U–U fan-out (E4) | a per-relation rule from H2GB's per-type lists (e.g. by destination type), or N/A |
| 3 | HINormer (H1) | H2GB's model as is (a GCN head under HINormer's name, full-batch), the authors' own HINormer with its full-graph GCN each step, or N/A |
| 4 | PSHGCN (P1) | MAG only with the label part off / with `sparse_tools` installed; per-dataset relation lists and label keys elsewhere; or N/A |
| 5 | H2SGNN (S4b) | a relation rule for the H2GB graphs, or N/A |
| 6 | H2G-former mag-year (a) | N/A (H2GB's own code cannot run it), or run without MetaPath2Vec (a deviation) |
| 7 | ChebNetII IEEE-CIS-G / H-Pokec (c) | learned-input variant without precomputation (not their scheme), or N/A |
| 8 | Hetero²Net (d) | metapath lists for OAG/IEEE/Pokec/PDNS or none (as their FreeBase); then the NeighborLoader path with a label-route test |
| 9 | SeHGNN MAG (b) | train ComplEx (their README) or use their random-init variant (README's "without extra embeddings"), or N/A |

branch=baselines-h2gb sha=<see final message> card=docs/H2S-NN_baselines_h2gb_part2_card.md base=6ba6dec00369ff590b341ce3771da90528fcc1cd
