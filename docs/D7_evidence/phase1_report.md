# D7–D9 Phase 1 — inventory, baseline, pre-registration, decisions

Branch `D1_add` at `c556e2a` (clean). Paths relative to `H2S-NN/`; line numbers at `c556e2a`.
Environment: conda `h2snn`, torch 2.6.0+cu124, CUDA available.

## 1. Pre-step inventory

### Item 1 — consumers (VERIFIED-CODE, grep at `c556e2a`)

| Interface | Definition | Consumers | Change in Part D |
|---|---|---|---|
| `FaceGroup` | `src/data/structures.py:9-35` | producers `preprocessing.py:467`, `graph_faces.py:203`; `structures.py:37` (`face_lengths_of`), `:70`; `link_pred_structures.py:67`; `h2snn_model.py:122,142` (face lengths), `:502-557` (latent, gate, path terms read `face_node_index`, `relation_index[0]`, `face_edge_index`), `:628`; `preprocessing.py:150-214` (diagnostics, reporting only); tests `test_factorized_maps.py:58-121` (`_random_pap_data`, golden fixture), `test_v2_model.py:747-753`, `test_task_heads.py:93,118-120`, `test_data.py:211-249`; evidence `../docs/heads_evidence/v2_golden.py:85,98-100` (serializes by `__slots__`), `../docs/D1_evidence/synthetic_graph.py:83-94` | gains `m_f` with a default (fixtures and golden file keep working); `face_edge_sign` becomes ±1 |
| `PathTransportOperator` | `src/models/map_operators.py:352-473` | `h2snn_model.py:547-550`; `vertex_path_laplacian.py:81-113` (`PathCellTerm` via `IncidenceOperator`, `:32-41`), `:144-169` (`HolonomyEdgeTerm`); tests `test_v2_model.py:97,164-213,390-391,509-514`, `test_factorized_maps.py:28`; evidence `D4_evidence/mutation_check_d4.py:20`, `u1_probe.py:29-55`, `D1_evidence/mutation_check.py:155` | U1 Horner + prefix methods; U2 optional `face_edge_sign` (default all +1, so every existing call is unchanged) |
| `enumerate_metapath_faces` | `src/data/preprocessing.py:360-477` | `dataset.py:151`; `tests/test_data.py:190-214` | kept as `path_sampler: dfs` |
| `build_typed_path_face_groups` | `src/data/graph_faces.py:46-212` | `link_prediction_dataset.py:952`; `graph_classification_trainer.py:190`; comment `graph_classification_dataset.py:388`; `tests/test_data.py:218-277` | kept as `path_sampler: dfs` |
| `validate_face_orientation` | `src/data/face_geometry.py:15-62` | `graph_faces.py:191`; `preprocessing.py:456`; `map_operators.py:398`; evidence `D4_evidence/acm_face_check.py:58` | U2: signs ±1 |
| `flatten_edges` | `src/data/hetero_flatten.py:57-84` | `dataset.py:126`; `link_prediction_dataset.py:650`; `tests/test_data.py:200`; docstring `src/baselines/bundles.py:93` | unchanged; the fact layer runs after it |
| `edge_type` / `relation_names` / `relation_tuples` | `structures.py:56-58`, `link_pred_structures.py:29-30` | `dataset.py:126-190`; `link_prediction_dataset.py:650,791,819,884,892-933` (reverse mirroring, leak checks), `:952-972`; `graph_classification_dataset.py:61-121` (TU types), `:131,142,384`; `h2snn_model.py:116,135` (`num_relations`), `:570`; `sheaf_maps.py:109-243,309-342` (frames, side code, scale — shared roles go here); trainers `graph_classification_trainer.py:192,204`, `link_prediction_trainer.py:206`; `preprocessing.py:150-305` (reporting); tests `test_task_heads.py:172-184`, `test_data.py:428-463`, `test_factorized_maps.py:53-160` | relations re-indexed to the kept set under `tied` |
| Tuning search spaces | `scripts/run_h2snn_wandb_tuning.py:193-243` (`validate_trial_params`), `:265-320` (`build_sweep_config`), `:246` (base config); `scripts/run_graph_classification_repeated_cv_wandb_tuning.py:87,265,343,400`; notebooks `h2snn_*_wandb_tuning_colab.ipynb` (3) | — | U3: reject `fact_layer` as a sweep parameter and `untied` in any tuning base config |

Out of scope: `src/baselines/adapters/{hetero2net,_gc_typed,_hetero2net_lp,_polynomial_lp,_target_block}.py` read edge types of their own bundles, not H2S-NN's prepared data.

### Item 2 — ρ per dataset (VERIFIED-RUN)

Script `inventory_rho.py` (independent of `src/`; computes ρ from the definition only). Raw output `inventory_rho_{hgb,tu,amazonbook}.{txt,json}`.

| Dataset | ρ | Facts before → after | Unmatched | Reciprocal (ρ(r) ≠ r) | Self-loops | Near partners |
|---|---|---|---|---|---|---|
| ACM | cite↔ref, paper→author↔author→paper, paper→subject↔subject→paper, paper→term↔term→paper | 547,872 → **273,936** | 0 | cite 25 pairs (and ref 25) | cite 8, ref 8 | cite vs cite: 58 of 5,343 reversed edges shared (not equal) → cite not self-inverse |
| DBLP | 3 mirror pairs (author–paper, paper–term, paper–venue) | 239,566 → **119,783** | 0 | 0 | 0 | none |
| IMDB | 3 mirror pairs (movie–director, movie–actor, movie–keyword) | 86,642 → **43,321** | 0 | 0 | 0 | none |
| AmazonBook (LP) | rates↔rated_by | 4,761,460 → 2,380,730 | 0 | 0 | 0 | none |
| LastFM, PubMed, amazon, youtube (LP, local HGB) | **UNVERIFIED** — files not on disk (`data/` holds only acm, dblp, imdb); PyG has no loader for them | — | — | — | — | — |
| PROTEINS | ρ(r) = r (1 relation) | 162,088 → 81,044 | 0 | — | 0 | — |
| ENZYMES | ρ(r) = r (1 relation) | 74,564 → 37,282 | 0 | — | 0 | — |
| NCI1 | ρ(r) = r (1 relation) | 265,506 → 132,753 | 0 | — | 0 | — |
| MUTAG, pipeline edge types | **only label 3 self-inverse**; labels 0, 1, 2 untied | 7,442 → 7,441 | 4,708 / 2,008 / 724 | 1,953 / 379 / 47 | 0 | label 0 vs 0: 3,906 of 4,708; 0 vs 1: 711; … |
| MUTAG, PyG `edge_attr` labels | **ρ(r) = r for all 4 labels** | 7,442 → 3,721 | 0 | — | 0 | none |

HGB: all 10 pairs exact, 0 unmatched, `ρ(cite) = ref`, 25 reciprocal citations, 8 self-loops in cite, facts 273,936 / 119,783 / 43,321: **identical to the card**.

TU: PROTEINS, ENZYMES, NCI1 have no edge labels (one relation) and ρ(r) = r. MUTAG's raw bond labels are symmetric (`mutag_label_check_output.txt`: 0 reversed edges with a different label; PyG's labels match raw on 0-mismatch), but **the pipeline's `_load_tu_type_tensors` (`graph_classification_dataset.py:61-121`) assigns raw labels by position, while only 943 of 7,442 PyG edges are in raw-file order: 1,771 bonds get the wrong type.** Pre-existing bug, out of scope; see decision D3.

### Item 3 — where each trainer builds or reuses face groups (VERIFIED-CODE)

- Node classification: built once in `dataset.py:151` (DFS seeded by `data.acm_split_seed`); the trainer passes `data` to the model for train (`trainer.py:129`) and eval (`:146`); the model reads `data.face_groups` (`h2snn_model.py:628`).
- Link prediction: built once on the message graph in `link_prediction_dataset.py:952` (seed `config.seed`); used for train and eval in `link_prediction_trainer.py:207`.
- Graph classification: rebuilt per batch in `graph_classification_trainer.py:190-199` (seed `config.seed`, per-graph seeding), train and eval.

## 2. Baseline (VERIFIED-RUN)

- Golden: `conda run -n h2snn python -m unittest tests.test_task_heads.NodeClassificationGoldenTests -v` → `Ran 1 test … OK` (`u0_golden.txt`).
- Full suite: `… -m unittest discover -s tests -v` → `Ran 195 tests`, `FAILED (failures=11, errors=13)`; all 24 in `test_baselines`, same names as `../heads_evidence/final_fail_names.txt` (`u0_full_suite.txt`, `u0_fail_names.txt`).

## 3. Pre-registration U1–U8

| Unit | Change | Test / measurement | Expected | Wrong if |
|---|---|---|---|---|
| U1 | `PathTransportOperator.transport_sum` (Horner, L steps), `adjoint_prefix` (t_i = T_iᵀz, L steps); `PathCellTerm.apply` = `w m (s_i² x_i − t_i)` for transport incidences; free maps and `HolonomyEdgeTerm` unchanged | T1 (L ∈ {2,3,4}, per-edge and per-side; signed hops added in U2) vs dense `Σ w Bᵀ(I−11ᵀ/n)B`; step counter; golden max-abs diff of logits and gradients vs `v2_golden.pt`; full suite | T1 ≤ 1e-10 (f64); 2L steps (L=2: 6→4, L=4: 20→8); golden: 5 cases (`no_path_cells`, `free_path_maps`, `endpoint_edges`, `holonomy_edges`, `h0_only_readout`) bit-identical, the other 8 differ by ≤ 1e-6, init state identical; re-captured fixture tensors bit-identical to stored | T1 > 1e-10; golden > 1e-5 (stop); one of the 5 cases not bit-identical; fixture tensors differ |
| U2 | sign per (path, hop); −1 hop = `F_tgt⁻¹F_src` by selecting factors per row (no extra map application); signs in `apply`, `apply_T`, holonomy, Horner, prefix, `squared_scales`; validator accepts −1 on reversed facts | T2, T3, T1 with signs; validator rejects a wrong sign, sign ∉ {±1}, non-simple | T2 ≤ 1e-10; T3 ≤ 1e-10 under per-edge; under per-side the reversed cell differs by exactly `s_{v_L}²` | T2 or T3 > 1e-10 after 3 attempts (stop) |
| U3 | `src/data/fact_layer.py` (ρ, tying, kept relations, unmatched counts) in the 3 data builders; shared roles for ρ(r)=r in both map families; `ablation.direction_blind_maps`; `data.fact_layer`; tuning check | T4 (fixtures + real HGB counts), T5, T6, T16 (untied, direction-blind) | counts exact; T5 ≤ 1e-10 vs control > 1e-3; T6 kernel dims equal | any count off; T6 fails after 3 attempts (stop) |
| U4 | `MetaPathConfig.hops`; resolution under the active layer; schema + reverse = one group | T10; presets resolve uniquely | exact | a default schema needs an untied relation (stop) |
| U5 | vectorized walk sampler (CSR per relation and direction), fact-tuple identity, summed `m_f`, `uniform_instance` (L ≤ 2), `data.path_sampler` | T7, T8, T9, T13, T15, T16 | thresholds below | T13 fails after 3 attempts, or vectorization impossible (stop) |
| U6 | `data.resample_paths`: train draw seed `base + epoch`, eval draw `base` | T12 | exact | — |
| U7 | LP: held-out positive removes its fact; leak checks in both orientations, on the message graph and sampled paths, at train and eval | T11 with injection control | exact | leakage freedom not guaranteed (stop) |
| U8 | measurements, smoke run, mutation check (8 defects), report, CLAUDE.md/AGENTS.md | — | Δ₀ edge ratio exactly 0.500 on HGB; transport steps L=2 6→4; ACM peak (d=32, k=2, tied + walk) 1.2–2.6 GiB | peak > 2.6 GiB; a mutation not caught |

Thresholds the card leaves to me:
- **T6:** kernel = eigenvalues ≤ 1e-9; the smallest eigenvalue outside the kernel is reported.
- **T7:** 200 draws × r = 20 walks on each fixture. Statistic: Σ over draws of the per-draw Pearson χ² of intermediate counts vs the walk probabilities conditioned on acceptance (2 dof per draw). Pass if p ≥ 1e-3. DFS control (quota 20) must give p < 1e-6. Fixtures: (a) anchor paper with 3 authors × 30 other papers; (b) hub, 3 authors with 300 / 3 / 3 other papers.
- **T8:** 4,000 draws, r = 3, palindromic and mixed-direction schemas, w ≡ 1, fixed maps, exact target by enumerating walks. Pass: ‖mean − target‖/‖target‖ ≤ 0.02 and max |z| ≤ 5 per coordinate. Dedup control: max |z| > 10.
- **T13:** 60 fixtures, `λ_max ≤ 2 + 1e-9`, m_f ∈ (0, 3], mixed signs, per-edge and per-side.
- **T15:** CPU, synthetic bipartite graph, 20k vs 200k anchors, same degree law, median of 5 after 1 warm-up, time ratio ≤ 15.

## Decisions (user approval, 2026-11-07)

1. LP data (LastFM, PubMed, amazon, youtube) not available: proceed; real-data T4 checks run only when the files are present; reported UNVERIFIED.
2. Local-HGB LP: compute ρ on `link.dat` before splitting; hold out positives by fact (both orientations leave together); the leak check covers both orientations. Verify that the baselines use the same split function; if they use a different split path, stop and report.
3. Fix the MUTAG loader in a separate commit (read PyG `edge_attr`), correct `tests/test_data.py:428`; report graphs/edges changed and whether any baseline goes through the same loader.
4. LP and GC: schema-free walk (uniform over all facts at each hop, cells grouped by the realized (relation, direction) sequence); state its target distribution as for D8; report the number of groups per dataset.
5. `path_sampler: dfs` with `fact_layer: tied` is rejected.
6. Seeds: walks use `config.seed` for all tasks; DFS keeps its current seeds.
7. Per-side ablation: build every merged cell once from its canonical orientation (deterministic); document the s² dependence.
8–12. Approved as proposed (gate reads stored orientation; T4 real-data via skipUnless + evidence script; data-level untied+dfs regression; c_e symmetrized as softplus of the mean; compact relation ids keeping original names).
13. Approved; document that GC evaluation draws depend on the evaluation batch composition.

Scope reductions requested by the user: T3, T7 and T15 dropped (sampler timing reported in the U8 measurements instead); mutation check reduced to three mutations (−1 hop without the adjoint; duplicates dropped instead of summed; reversed orientation of a held-out fact not removed); no independent sub-agent review.
