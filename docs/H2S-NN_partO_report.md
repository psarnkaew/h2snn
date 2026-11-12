# Part O report — schema read-out and coverage of the main runs, fixed-literature-schema row

**Card:** `docs/H2S-NN_partO_card.md` (O0–O4). **Design authority:** spec v1.19 (§3.2 read-outs; §6 Ablation-table bullet "fixed-literature-schema row" and bullet "Schema read-out and coverage"); `docs/predefined_metapaths.md` §2; `docs/experiment_settings_decisions.md`. **Branch:** `part-o` from `v3b` at `af2b4e2`. **Evidence:** `docs/partO_evidence/`.
**Rule kept throughout:** correctness tests only. No training run on real data, no metric, no `I_σ` of a trained model, no comparison between methods. The method (WISP, energy, maps, solver, head readouts) is unchanged.

---

## O0 — Preconditions

**Branch (VERIFIED-RUN).** `git checkout -b part-o af2b4e2` → `part-o`; `git rev-parse v3b` = `af2b4e2bebe8949ec333d0c0abed69f72b7343ba` (v3b is at the given commit).

**Decisions read (VERIFIED-SOURCE, `docs/experiment_settings_decisions.md`).**
- O-1 revised (2026-11-13 03:11): the row uses HGB's published meta-path table, paths starting at the target type, on ACM, DBLP, IMDB, Freebase; it supersedes v1.18's Hetero²Net list. No row on H2GB datasets.
- O-2 (2026-11-13 02:45): walks per anchor = the tuned `r`, split evenly over the listed meta-paths, no separate tuning.
- Z-3 (2026-11-13 02:45): schema read-out and coverage approved. Spec §6 then fixes the reading in advance: `c_σ`, `I_σ`, `ρ_σ`, mean `g`, top ten length-2 sub-path schemas, number of distinct schemas, Kendall τ across seeds. `ΔL_σ` is not reported (v2 operator).
- 2026-11-12 21:03: no smoke runs, correctness tests only. The card's 2-epoch smoke was removed.

**Code read (VERIFIED-CODE).**
- `src/analysis/wisp_readouts.py`. `pair_index` / `_canonical` (l.49–82) build the canonical sub-path signature `(t_0, r_1, d_1+1, t_1, …)`, which is the smaller of the two orientations. `i_sigma` (l.126–134) gives each pair `(i<j)` of cell `f` the weight `w_f m_f / (2|f|)` (Prop. K) and normalises the total to 1. `kendall_tau` (l.266–284) returns τ-b with a permutation p. `delta_loss` (l.230) uses the v2 operator (`build_operator`, `VertexPathSheafLaplacian`) and is not used here.
- `src/utils/checkpointer.py`. `save` (l.46–64) writes `state_dict`, the optimizer, metrics and metadata to `<exp_dir>/checkpoints/<tag>.pt`. `save_if_best` (l.77–97) writes `best.pt`. A strict `load` is at l.66–75.
- `src/analysis/model_loading.py` `load_trained_model` (l.14–36): `config.yaml` → `TrainConfig.from_dict` → `build_dataset` → `build_model` → `Checkpointer.load`.
- Main-table seed runs: `scripts/run_h2snn_wandb_tuning.py`. Tuning runs on seed 0 (`TUNING_SEED`, l.76). `--final_seeds 0 1 2 3 4` calls `run_final_seeds` (l.690–723). That calls `train_trial` (l.384–473) with `output_dir = <output_dir>/<exp>/final_seeds` and `exp_name = <exp>_tune_final_seed{s}_final_seed{s}`, so each seed has its own directory. `train_trial` sets `raw["seed"] = seed` (l.411), builds `TrainConfig.from_dict(raw)` and writes `save_config(cfg, exp_dir/"config.yaml")` (l.420–421) before training. The trainer writes `checkpoints/last.pt` every epoch and `checkpoints/best.pt` (`src/trainers/trainer.py:145-159`). The other drivers (`scripts/run_h2snn_runs.py:128-129`, `scripts/train_h2snn.py:46-47`) also save `config.yaml` first.
- Deletion: `grep -rn "rmtree|unlink|os.remove|.rmdir|shutil|TemporaryDirectory"` over `scripts/` and `src/` (vendored adapters excluded) finds deletions only in `scripts/lp/apply_feature_rule.py:114` and `scripts/lp/write_hgb_negatives.py:70,92`. Both are data-preparation scripts. Nothing deletes a run directory (VERIFIED-RUN, command in O1).
- Evaluation draw: `build_dataset` builds the cells with `seed=config.seed` (`src/data/dataset.py:236-246`). The trainer evaluates on these cells (`trainer.py:141-142`). Training draws use `seed + epoch` (`trainer.py:68-84`).

**Findings that shape O1–O3 (VERIFIED-CODE).**
1. `DataConfig._resolved_node_classification` (`src/configs/config.py:745-790`) fills `metapath_templates` from the old presets (ACM PAP, PSP; DBLP APA, APTPA; IMDB MAM, MDM) for every NC configuration, including `path_source: wisp`. A saved tuned `full` configuration therefore carries those presets. If only `path_source` were changed, the row would sample the old presets. O3 deals with this.
2. `DataConfig.__post_init__` (l.517–525) rejects `wisp_walks_per_anchor ≠ 0` and `wisp_max_length ≠ 4` under `path_source: schemas`. A tuned `full` configuration has both, so the builder could not change only the path source.
3. The model treats schema groups differently from WISP groups. For a schema group, `g`'s hop token is one relation embedding per group, with no direction one-hot (`h2snn_model.py:858-876`), and `face_lengths` come from the draw (`build_model`, l.1419–1431). For example, ACM PcPAP and PrPAP would give `g` identical hop tokens.
4. H2S-NN has no Freebase node-classification loader. `DATASET_TASK_TYPES` (`config.py:399-413`) and `_load_hetero_data` (`dataset.py:26-45`) cover ACM, DBLP and IMDB only. See out-of-scope findings.

**Pre-state of the suite.** No code changed between Part N-fix F5 (`5038744`) and `af2b4e2`: `git diff --stat 5038744 af2b4e2` lists only `docs/` files (VERIFIED-RUN). F5's full suite gave **Ran 583; failures=2, errors=2**, all in `test_baselines` (vendor clones ×3, notebook) (`docs/partNfix_evidence/f5_full_suite_output.txt`). This is the comparison set for O4. The working tree also holds uncommitted edits to `src/baselines/adapters/*` from another session. They are not part of this branch and are never staged.

---

## O1 — What the server runs keep

**O1.1 Retention: no script changed.**
- Every main-table seed run writes its resolved configuration and checkpoints into its own directory, and nothing in the campaign deletes them (VERIFIED-CODE):
  - `run_final_seeds` → `train_trial` writes `config.yaml` (`scripts/run_h2snn_wandb_tuning.py:420-421`), with `seed` = the final seed, which is also the base seed of the evaluation draw (l.411).
  - The NC trainer writes `checkpoints/best.pt` and `last.pt` (`src/trainers/trainer.py:145-159`); the LP trainer does the same (`src/trainers/link_prediction_trainer.py:97-111`).
  - The directory name carries the seed, `<exp>/final_seeds/<exp>_tune_final_seed{s}_final_seed{s}` (l.713–714), so seeds never share a directory.
- The resolved dict holds the seed, `data.wisp_max_length` (`ℓ_max`), `data.wisp_walks_per_anchor` (`r`, already resolved by `TrainConfig.from_dict`) and `data.path_source` (resolved to `wisp`).
- Deletion grep over `scripts/` and `src/` (VERIFIED-RUN, `o1_deletion_grep.txt`): only `scripts/lp/apply_feature_rule.py:114` and `scripts/lp/write_hgb_negatives.py:70,92` delete anything. Both remove stale LP data files, never run directories.
- **GC** keeps its final weights as its protocol requires: `checkpoints/last.pt` is rewritten every epoch before the early-stopping check (`src/trainers/graph_classification_trainer.py:85-91`, break at l.144). After the loop, `last.pt` therefore holds the final weights the GC test metrics are computed from. `best.pt` (by `checkpoint_metric`) is also written, as an artifact only (l.71–75).

**O1.2 Reload test (VERIFIED-RUN).** New in this unit:
- `src/analysis/model_loading.py`:
  - `evaluation_cell_weights(model, data)`: `w_f = exp(g(ξ_f))` through the model's own `H2SNN._cell_weights` (eval mode, no gradient, training flag restored).
  - `nonempty_groups`.
  - `load_trained_model` is unchanged apart from its docstring.
- `tests/partO_fixtures.py`: a synthetic ACM-shaped `HeteroData` with HGB's eight ACM relation names. `patched_acm_loader` makes `build_dataset` read it.
- `tests/partO_reload_child.py`: the fresh-process reload.
- `tests/test_partO_reload.py`, two tests:
  - `RetentionTests`: the real `run_final_seeds`/`train_trial` on seeds 0 and 1, 2 epochs, anchored `full`, WISP `ℓ_max = 3`, `r = 4`. W&B is replaced by a stub. Each seed directory holds `config.yaml` (seed, `path_source: wisp`, `ℓ_max`, `r`) and a loadable `best.pt`, and the second seed leaves the first untouched.
  - `ReloadTests`: the same steps as `train_trial` for one seed. After every epoch (after its checkpoint save), an epoch callback records `w_f` on the evaluation draw. The test then reloads `config.yaml` + `best.pt` in a fresh process (`python -m tests.partO_reload_child`, through `load_trained_model`) and compares with `torch.equal` (same dtype and device) against the recorded tensors of the best epoch: node, fact and sign tensors, relation and signature ids, `m_f` and `w_f`. Negative control: the run directory is copied with `seed + 1` in `config.yaml`, and that reload must differ.
- `python -m unittest -v tests.test_partO_reload` → **Ran 2 tests in 50.5 s, OK** (`o1_tests_output.txt`). The reload is exact. The INFO log lines of the synthetic 2-epoch trainings are removed from the evidence file: they are fixture logs, not results.
- **Mutation (card) (VERIFIED-RUN):**
  - (a) `load_trained_model` with `cfg.seed + 1` → **FAILED** at the seed check, `AssertionError: 6 != 5` (`o1_mutation_output.txt`).
  - (b) Only the draw rebuilt with `seed + 1`, reported seed unchanged → **FAILED**: `first differing tensor: group 0 nodes` (`o1_mutation_b_output.txt`).
  - The file was restored after each mutation. `git diff` shows only the O1 additions.

---

## O2 — Read-out script (evaluation only)

**New code.**
- `src/analysis/schema_readout.py` (imports `wisp_readouts`, `schemas`, `fact_layer`, `config`; no training code):
  - `schema_statistics`: per sub-path schema σ (canonical signature of positions i..j of a cell, as `wisp_readouts`, every span) it computes:
    - `I_σ`: share of the pair weight `w_f m_f / (2|f|)`;
    - `c_σ`: the same share with `w_f = 1`;
    - `ρ_σ = I_σ / c_σ` (`None` when `c_σ = 0`);
    - mean `g = log w_f` over the distinct cells containing σ;
    - the cell count. Everything is in float64.
  - `literature_signature(s)`: maps a listed meta-path to its signature by relation and type names, through `resolve_schema` / `FactLayer.resolve`. An unknown relation raises `KeyError`; a hop that does not join the named types raises `ValueError`; an unknown type raises `KeyError`. Two listed paths with one signature also raise.
  - `run_record`: the JSON of one run. A listed path that is absent from a draw is kept with `in_draw: false` and `c = I = 0`, `ρ = None`.
  - `kendall_matrix`: τ-b and permutation p between runs (`wisp_readouts.kendall_tau`), over all sub-path schemas or over length 2 only.
- `scripts/schema_readout.py --runs <dirs> --out_dir <dir>` writes:
  - per run: `<out_dir>/<dataset>/<run>.json`;
  - per dataset: `schema_readout.md` (listed meta-paths: c / I / ρ per seed; top ten length-2 schemas by I per seed; distinct schema counts; τ between seeds).
- Its imports beyond the analysis modules are `src.analysis.model_loading` (model loading: `load_trained_model`, `evaluation_cell_weights`). Node classification only (the read-out of spec §6). `ΔL_σ` is not computed.
- The literature table is `LITERATURE_METAPATHS` plus `LITERATURE_SCHEMA_SOURCE` in `src/configs/config.py`. It is added in this unit because the read-out maps the list; O3 gives it its role in the row. The HGB table, paths starting at the target type:
  - ACM: PAP, PSP, PcPAP, PcPSP, PrPAP, PrPSP;
  - DBLP: APA, APTPA, APVPA;
  - IMDB: MDM, MAM;
  - Freebase: BB, BFB, BLMB, BPB, BPSB, BOFB, BUB.
- How each path is defined at the relation level (VERIFIED-SOURCE):
  - Relation names are PyG `HGBDataset` names, from the real ACM, DBLP and IMDB graphs (VERIFIED-RUN: `_load_hetero_data` + `flatten_edges` on `data/NC/{acm,dblp,imdb}`).
  - ACM link ids come from `data/NC/acm/raw/ACM/info.dat` (0 paper-cite-paper, 1 paper-ref-paper, 2/3 paper-author, 4/5 paper-subject) with MAGNN's `etypes_list` (`vendor/HGB/NC/benchmark/methods/MAGNN/run_ACM.py:20`).
  - Freebase uses HAN's DGL edge names (`vendor/HGB/NC/benchmark/methods/HAN/utils.py:180-201,234`): `'ab'` = link a→b where stored, otherwise the transpose of the stored b→a link (index + 36). Freebase relation names follow PyG's `info.dat` parse (`<type>__<word>__<type>`; types and words from `data/NC/freebase/raw/Freebase/info.dat`).

**Tests (VERIFIED-RUN).** `python -m unittest -v tests.test_partO_readout` → **Ran 8 tests in 14.2 s, OK** (`o2_tests_output.txt`):
- Hand cells: two P-A-P cells (m = 0.5, 1; w = 2, 0.5) and one P-A-P-A cell (m = 0.25, w = 4). `c_σ`, `I_σ`, `ρ_σ` and mean g of PA, PAP, APA and PAPA equal the hand fractions to 12 places; cell counts and spans also match.
- Constant g gives `|ρ_σ − 1| ≤ 1e-12` and mean g = log 3.7 for every σ, on a WISP draw (`tests.test_wisp` fixture, ℓ_max 4).
- `evaluation_cell_weights` of a v3-B anchored model on a WISP draw equals `H2SNN._path_weights(_path_latent(...))` per group (`torch.equal`), with every module in eval mode.
- The same model with the path gate's weights set to zero (`g = b`) gives one value of `w_f` and `ρ_σ = 1` (1e-12) for every σ.
- The ACM list maps to the hand signatures, as typed relation sequences in the code's notation `(t0, r1, d1+1, t1, …)`:
  - PAP `(0,1,2,1,1,0,0)`; PSP `(0,2,2,2,2,0,0)`;
  - PcPAP `(0,0,2,0,1,2,1,1,0,0)`; PcPSP `(0,0,2,0,2,2,2,2,0,0)`;
  - PrPAP `(0,0,0,0,1,2,1,1,0,0)` (ref = cite against its orientation); PrPSP `(0,0,0,0,2,2,2,2,0,0)`;
  - the card's example PTP `(0,3,2,3,3,0,0)`;
  - short names `paper-author-paper`, etc.
- A misspelled relation (`paper__to__autor`) raises `KeyError`; a misspelled type raises `ValueError`.
- Listed paths absent from a draw are reported with `in_draw: false` (never dropped). A dataset without a list gives `literature: null`.
- Script end to end on two synthetic 1-epoch runs: one JSON per run (six listed entries, the I and c shares each sum to 1, schemas sorted by I) and one table with the four sections. No value is read or reported.

**Mutation (card) (VERIFIED-RUN, `o2_mutation_output.txt`).** `I_σ` computed without `m_f` (`learned_pair = w[pairs.cell] / (2|f|)`) → **FAILED (failures=88)**:
- `test_hand_values`, e.g. PA `0.549… != 0.583…`;
- both ρ = 1 tests.
- Restored; `git diff` of the module shows only the new file.

**Deviation from the card text.** The card's ACM example lists "PAP, PSP, PTP" (Hetero²Net's list, the v1.18 O-1). Spec v1.19 governs, so the ACM test checks the six paths of HGB's table. PTP is mapped as well, as the card's example of a typed relation sequence.

---

## O3 — The fixed-literature-schema row (Z-2)

**What the row is in code (VERIFIED-CODE).** The row is a resolved node-classification configuration with `path_source: schemas` and `wisp_walks_per_anchor = r ≥ 1` (`is_literature_row`, `src/configs/config.py:1480`). With `r = 0` the old legacy schema rows remain (see below). The method is unchanged: the row uses the same model, energy, maps, solver, `g` and head. Only the source of the cells changes.
- **List.** `LITERATURE_METAPATHS` and `LITERATURE_SCHEMA_SOURCE` (`config.py:293`, added in O2) form one table: HGB's published meta-path table, paths starting at the target type, on ACM, DBLP, IMDB and Freebase only. A row on any other dataset raises. The resolved row has no `metapath_templates`, because its list is the table, and an explicit list raises (`config.py:772-790`).
- **Old presets.** `NODE_CLASSIFICATION_PRESETS` (ACM PAP, PSP; DBLP APA, APTPA; IMDB MAM, MDM) now fill only the legacy schema rows, i.e. `path_source: schemas` with `r` unset (`config.py:880`). Those rows are the baseline bundles' pinned DFS row (`src/baselines/bundles.py:59-64`) and the existing schema tests. Code paths that still need them: `dataset.py` (DFS and the D8 walk sampler over `metapath_templates`) and `tests/test_cooperative.py`, `test_paths_and_facts.py`, `test_wisp.py:647`. **Behaviour change:** WISP configurations no longer carry the presets in their resolved and saved `metapath_templates` (now `()`). WISP never read them (R8). Without this change a tuned `full` configuration would hand the old presets to the row, as found in O0.
- **Budget (O-2).**
  - `literature_walk_split(r, k)` (`config.py:351`): with `r = qk + s`, the first `s` paths in table order get `q+1` walks per anchor and the rest get `q`. The walks sum to `r`, each path gets at least one, and `r < k` raises. The search space `r ∈ {8,…,64}` with `k ≤ 7` never reaches that case.
  - Walks start from every target node. Within a path, each hop picks a uniform fact of the hop's (relation, direction) at the current node; walks that revisit a node are rejected (D8's stated distribution, `path_sampler.sample_walks`). Each accepted walk of path π carries `1/r_π`.
  - `src/data/literature_paths.py` holds `build_literature_path_groups` and `literature_walk_plan`. `dataset.build_node_classification_paths` calls them for the row (`dataset.py:163`), for the evaluation draw (`seed`) and for the training draws (`seed + epoch`) alike.
- **Cell format.** The row's cells use WISP's format: one group per length, D8 identity, per-cell signature ids. They go through `merge_length_group` (`src/data/wisp_sampler.py:300`), which was extracted from `merge_wisp_cells` without change. `build_model` gives the row per-cell action tokens (`path_actions`) and one projection of `g` per listed length (`h2snn_model.py:1431`). As a result, `g` reads a listed cell exactly as it reads a WISP cell, with relation embeddings and hop directions. Without this, the legacy schema groups would give `g` one direction-free token per group, and ACM's PcPAP and PrPAP would look the same to `g` (O0 finding 3). This is a choice of the comparison row, not of the method (decision D-O2 below).
- **Builder.** `fixed_literature_schema_config(full)` (`config.py:1496`):
  - takes a resolved v3-B `full` WISP NC configuration (`propagation: anchored`, `arm: full`) of a dataset with a list, and raises otherwise;
  - sets `data.path_source: schemas` through the saved-configuration round trip (`dataclass_to_dict` → `TrainConfig.from_dict`);
  - raises if any other flattened setting differs.
  - `DataConfig.__post_init__` now accepts the tuned `ℓ_max` and `r` under `schemas` when `r ≥ 1` (`config.py:623`). `ℓ_max` is carried along unchanged and is not read by the row's draw. The proposal must stay uniform.
- **Still refused in tuning.** `reject_never_tuned_settings` refuses `path_source: schemas` for NC, as before (`config.py:143-147`), and refuses `path_source` as a search parameter.

**Stop condition checked (VERIFIED-RUN, data read only, `o3_literature_instances_output.txt`, script `o3_literature_instances.py`).** On the real HGB graphs (tied fact layer), every listed path resolves to the relation-level hops below and has simple instances from the target nodes. Counts are target nodes reaching an instance in 8 walks:

| dataset | paths (targets with an instance) |
|---|---|
| ACM | PAP 2373/3025, PSP 3012, PcPAP 1857, PcPSP 1980, PrPAP 1321, PrPSP 1445 |
| DBLP | APA 2351/4057, APTPA 4057, APVPA 4057 |
| IMDB | MDM 3428/4932, MAM 4368 |
| Freebase | BB 36670/40402, BFB 9534, BLMB 10857, BPB 7845, BPSB 5920, BOFB 9580, BUB 10720 |

- ACM's PrP resolves to `(paper__cite__paper, −1)` because `ref` is the transpose of `cite`.
- Freebase's `book__and__book` is stored in one direction only (not self-inverse), so BB = `(+1, +1)`, as HAN's forward-only `['00','00']`.
- **No stop.** Freebase was read through PyG's `HGBDataset` directly, because H2S-NN has no Freebase NC loader (out-of-scope finding 1).

**Tests (VERIFIED-RUN).** `python -m unittest -v tests.test_partO_literature_row` → **Ran 11 tests in 19.1 s, OK**; with `tests.test_wisp.WispConfigValidationTests` **Ran 14, OK** (`o3_tests_output.txt`):
- The builder changes only `data.path_source`, on ACM, DBLP and IMDB at `(ℓ_max, r)` ∈ {(2, 8), (5, 64), (4, 33)}, flattened diff = `{data.path_source}`. The result survives the `config.yaml` round trip unchanged.
- The builder refuses: another arm, the legacy propagation, an existing row, and graph classification. A resolution that also changes another setting (simulated) raises and names it.
- Row rules: an explicit list, `r < k` (ACM `r = 5`), an adaptive proposal, an untied layer, and a GC row all raise. The legacy row (`r` unset) keeps the old presets and its `ℓ_max` rule.
- The row is refused in tuning.
- Split: for `k = 1..7` and `r = k..69` the walks sum to `r`, each path gets ≥ 1, max − min ≤ 1, in table order. Examples: `(16, 6) → (3,3,3,3,2,2)`, `(64, 7) → (10,9,…,9)`, `(5, 6)` raises.
- The sampler asks `sample_walks` for exactly `(3,3,3,3,2,2)` walks from every target node (`r = 16`, not divisible by 6), with path lengths in table order.
- Every drawn cell's full signature is one of ACM's six listed signatures, every listed path is drawn, and every cell has a paper end with `m_f > 0`.
- The evaluation draw is identical (`torch.equal`) for the same base seed and differs for another.
- The anchored v3-B forward on an ACM-sized synthetic graph is finite in train and eval mode, and so is the loss. The graph has ACM's node counts (3025/5959/56/1902), 2 cites, 3 authors, 1 subject and 10 terms per paper; ACM's term degree is lower here, stated.
- On the real graphs (data read only), the four lists resolve and every path has instances.

**Existing tests run against the change (VERIFIED-RUN, `o3_related_modules_before_test_update.txt`).** Command: `python -m unittest tests.test_wisp tests.test_paths_and_facts tests.test_v3b_model tests.test_cooperative tests.test_search_spaces tests.test_wandb_and_runs tests.test_task_heads tests.test_v2_model` → **Ran 198 in 1,171 s; FAILED (failures=2)**.
- 196 pass. These include the bit-for-bit golden NC outputs (`test_task_heads`), so the `merge_length_group` extraction left WISP's output unchanged.
- The two failures are sub-cases asserting the v1.4 rule "`wisp_walks_per_anchor` is rejected under `path_source: schemas`":
  - `test_wisp.WispConfigValidationTests.test_direct_construction_is_validated` (`{'path_source': 'schemas', 'wisp_walks_per_anchor': 30}`);
  - `…test_wisp_keys_are_rejected_under_schemas_through_from_dict` (`{'wisp_walks_per_anchor': 30}`).

**Test change (author to confirm, Decision D-O1).** Spec v1.19 §6 defines the row's budget as the tuned `r`, and the card requires a builder that changes only the path source. A configuration with `schemas` and `r` is therefore the row, so the old rule can no longer hold. I changed only those two `r = 30` sub-cases:
- Each now asserts the new rule: `DataConfig(path_source="schemas", wisp_walks_per_anchor=30[, wisp_max_length=5])` is accepted, and through `from_dict` it is a row with no `metapath_templates`.
- I added a still-failing case in its place: `schemas` + `r` + an adaptive proposal raises.
- Every other rejected case stays, including `ℓ_max` without `r` and the adaptive proposal.
- This is the only change to an existing test in Part O. No tolerance changed and no test was deleted or skipped.

---

## O4 — Full suite and report

**Full suite, run once (VERIFIED-RUN).** `python -m unittest discover -s tests` from `H2S-NN/` on `f5c2449` → **Ran 604 tests in 2,755 s; FAILED (failures=2, errors=2)** (`o4_full_suite_output.txt`; INFO log lines of the synthetic fixture trainings removed).

| failing test | in F5's known set (O0 pre-state)? |
|---|---|
| `test_baselines.VendorIsolationTests.test_all_declared_vendor_clones_are_present` | yes |
| `test_baselines.VendorIsolationTests.test_colliding_vendor_module_names_do_not_leak` | yes |
| `test_baselines.ReproducibilityTests.test_vendor_model_init_is_reproducible_under_seed_everything` | yes |
| `test_baselines.BaselineNotebookTests.test_notebooks_are_committed_clean` | yes |

- **No new failure.** Test count: 583 (F5) + 21 (Part O: `test_partO_reload` 2, `test_partO_readout` 8, `test_partO_literature_row` 11) = 604.
- The working tree's uncommitted `src/baselines/adapters/*` edits from another session were present during the run. The four failures are identical to F5's, which ran without those edits.

**Summary per unit.**

| unit | commit | result |
|---|---|---|
| O0 | `f897b6a` | preconditions read; branch `part-o` from `af2b4e2` |
| O1 | `cb3ece5` | retention holds, no script changed; **exact fresh-process reload** of cells, `m_f` and `w_f`; both seed mutations caught |
| O2 | `b0b803a` | `src/analysis/schema_readout.py` + `scripts/schema_readout.py`; hand values, ρ = 1 under constant g, model `w_f` identity, name mapping, misspelling raises; the `m_f` mutation is caught |
| O3 | `f5c2449` | row = HGB table + tuned `r` split evenly; the builder changes only `path_source`; every listed path has instances on its real graph (no stop); one superseded test rule updated (D-O1) |
| O4 | this commit | full suite: only the four known failures |

No stop condition was met. No metric, `I_σ` of a trained model, or method comparison is reported. Local runs were tests on synthetic graphs, plus data-only reads of the HGB files.

### Scripts changed in O1
None. Retention already holds (O1.1).

---

## Decisions for the author

| ID | Question | What I did (recommendation) | Alternative |
|---|---|---|---|
| D-O1 | Two existing sub-tests in `tests/test_wisp.py` asserted that `wisp_walks_per_anchor` is rejected under `path_source: schemas` (the v1.4 rule). | Changed only those two sub-cases to the v1.19 rule (with `r`, `schemas` is the row) and added a replacement failing case (O3). **Confirm.** | Keep the old rule and let the row carry its budget in a new field. The builder would then change two fields, against the card. |
| D-O2 | How `g` reads the row's cells. | WISP cell format: one group per length, per-cell relation embedding and hop direction, one projection per listed length. `g` reads a listed cell exactly as it reads a WISP cell, so only the path source differs. | The legacy schema groups: one direction-free token per group. ACM's PcPAP and PrPAP would look identical to `g`, and `g`'s input would differ from `full`'s. |
| D-O3 | "Uniform instances within a meta-path" (O-2). | D8's walk: a uniform fact of the hop's relation and direction at each hop, simple walks only. The builder may not change `path_sampler` (it stays `walk`). | Draws uniform over instances: `path_sampler: uniform_instance` exists for L ≤ 2 only. It would need an extension to L ≤ 4 and a second changed field. |
| D-O4 | `m_f` of a listed path's walks. | `1/r_π` per accepted walk (D8): each path's cells estimate that path's sum over anchors and instances, whatever its share of the budget. The split sets precision, not weight. | `1/r`: each path's mass proportional to its walk share. Paths would then get unequal weight in the energy, by budget rather than by data. |
| D-O5 | `r < k` (fewer walks than listed paths). | Raises; one walk each would exceed `r`. Unreachable in the search space (`r ≥ 8`, `k ≤ 7`). | One walk each (sum `k > r`). |
| D-O6 | Where the row runs write. | The builder keeps `exp_name`/`output_dir` (it may change only `path_source`). The caller must give the row its own directory; otherwise the row's `best.pt` would overwrite the `full` run's, which O1 must keep. | Let the builder take an `exp_name` and exclude the run location from the diff. |
| D-O7 | Kendall τ "of the `I_σ` rankings". | The table gives τ over all sub-path schemas and over length-2 only. **Choose which one the paper reports.** | One of the two only. |
| D-O8 | Freebase "BB". | Two hops B→B→B, as HAN's code `['00','00']`, which spec v1.19 names as the relation-level definition. The table's name suggests one hop, but one fact is a 1-cell, not a path cell. | Drop BB, or keep it with one hop (not expressible as a cell). |

## Out-of-scope findings (not fixed)

1. **No Freebase NC loader in H2S-NN.** `DATASET_TASK_TYPES` and `_load_hetero_data` cover ACM, DBLP and IMDB only (`src/configs/config.py:399-413`, `src/data/dataset.py:27-46`). Until a loader exists, neither the row nor the read-out (nor any H2S-NN Freebase NC run) can run on Freebase. The list itself was checked on the real graph through PyG (O3).
2. **No campaign script for ablation rows.** `run_h2snn_wandb_tuning.py --final_seeds` runs `full` only. The row (and the C-arms) need a driver that applies `fixed_literature_schema_config` to the tuned seed configurations and writes to separate directories (D-O6).
3. **`docs/predefined_metapaths.md` §2.** For DBLP and IMDB, MAGNN's `etypes_list` numbers are MAGNN's internal edge-type ids, not HGB link ids: HGB's DBLP link 1 is paper-term (`data/NC/dblp/raw/DBLP/info.dat`), so `[0,1]` read as HGB ids would be A-P-T, not APA. The doc calls them "relation ids". This is harmless here, because the table is defined by names, but the doc could mislead. The Freebase HAN codes are now decoded to letters and relations (O2), which partly closes that doc's open item.
4. **Behaviour change visible in logs.** WISP configurations now resolve to `metapath_templates: ()` (O3). `scripts/train_h2snn.py:62` logs "Metapaths: []" for WISP runs (cosmetic). Saved configurations written before this branch still carry the presets, and the builder refuses them loudly (explicit list under the row).
5. **Card text.** O2's test example "PAP, PSP, PTP" is v1.18's list. v1.19 was followed (O2 deviation).
6. **Working tree.** `src/baselines/adapters/*` and files under "Claude outputs/" carry uncommitted edits from other sessions. They were never staged.
