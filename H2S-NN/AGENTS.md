# AGENTS.md - H2SNN Agent Instructions

This file gives Codex and other coding agents the project-specific rules for working in this repository. It is based on the current H2SNN implementation, `docs/h2snn_current_model_documentation.html`, and `CLAUDE.md`.

## Environment - Always Use `h2snn`

Never run Python, tests, training, `pip`, or `conda` work for this project in the `base` environment or a system Python. The required libraries are installed in the Conda environment named `h2snn`.

Use one of these forms:

```powershell
# Preferred for scripts and tests from any shell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -m unittest discover -s tests -v
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\train_h2snn.py --config configs\experiments\acm_h2snn.yaml --device cuda --exp_name acm_h2snn_test
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -m pip install <package>
```

```powershell
# Also acceptable in an interactive PowerShell session
conda activate h2snn
python -m unittest discover -s tests -v
python scripts\train_h2snn.py --config configs\experiments\acm_h2snn.yaml --device cuda --exp_name acm_h2snn_test
```

If using an absolute Python path in automation, use:

```powershell
C:\Users\Acer\anaconda3\envs\h2snn\python.exe -m unittest discover -s tests -v
```

Current dependency notes:

- Python: `3.12` in `h2snn`.
- PyTorch target: `torch==2.6.*` with CUDA 12.4 wheels.
- PyG packages: `torch_geometric`, `torch_scatter`, `torch_sparse`.
- `torch_householder` is no longer required by the current `src/` model; the active `fasthpp` map uses the pure-PyTorch compact-WY implementation in `src/models/fasthpp.py`.
- Do not install `requirements.txt` into `base`.

Sanity check:

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -c "import torch, torch_geometric, torch_scatter, torch_sparse; print(torch.__version__, torch.cuda.is_available())"
```

## Active Codebase

The current H2SNN implementation lives in:

- `src/`
- `scripts/train_h2snn.py`
- `configs/experiments/`
- `tests/`
- `docs/h2snn_current_model_documentation.html`

The older `models/`, `exp/`, and top-level `utils/` code is a legacy homogeneous-graph CSNN baseline. Do not add new H2SNN features there unless the task explicitly asks for baseline comparison work.

## Current Model Summary

H2SNN is a heterogeneous sheaf-Hodge neural network for typed graphs. It turns selected typed path instances into face-like higher-order structures, learns sheaf restriction maps, runs Hodge-style diffusion over node and edge cochains, and predicts labels for a target node type.

Current pipeline:

1. `src/data/dataset.py` loads ACM, DBLP, or IMDB via PyTorch Geometric and flattens heterogeneous stores.
2. `src/data/preprocessing.py` pads typed node features, computes training-label-only propagation metrics, and appends relation metrics.
3. Manual schema mode enumerates configured meta-paths such as `PAP`, `PSP`, `APA`, and `MAM`; auto mode uses hard task-aligned AMPSS Stage-I pool construction followed by `schema_conditioned_v1` instance weighting inside the upper-Hodge operator.
4. `TypeSpecificNodeEncoder` embeds typed node features into shared hidden stalk dimension `d`.
5. `NodeToEdgeSheaf` learns relation-aware source and target maps for `delta0`.
6. `SequenceEdgeToFaceSheaf` uses an ordered path encoder (concatenation by default; LSTM as an ablation) to build edge-to-face maps for `delta1`.
7. `HodgeDiffusion` updates node states through `L0` and edge states through `L1`.
8. `NodeReadout` combines final node states, initial node states, and incident edge-state pooling for target-node classification.

Key files:

- `src/models/h2snn_model.py`: end-to-end model wiring.
- `src/models/sheaf_maps.py`: node-to-edge maps, edge initializer, edge-to-face maps.
- `src/models/coboundary_ops.py`: matrix-free `delta0`, `delta1`, `L0`, `L1`.
- `src/models/diffusion.py`: K-step Hodge diffusion.
- `src/models/readout.py`: target-node classifier.
- `src/data/{node_stage1,link_stage1,graph_classification_dataset}.py`: task-aligned AMPSS Stage-I selection and task-specific sparse coverage.
- `src/trainers/trainer.py`: loss computation, optimization, metrics, checkpointing.

## Research And Correctness Status

Keep claims code-grounded. The living documentation is `docs/h2snn_current_model_documentation.html`; read and update it when code changes affect model claims.

Verified fixes as of 2026-07-29:

- Closure edges are first-class members of the extended edge domain. They enter `delta0`, `L0`, `L1_down`, edge diffusion/readout, and the closed boundary consumed by `delta1`; their maps are defined by path transport rather than separately learned.
- `closure_scale` is consumed by the transported closure restrictions and must remain positive.
- Orientation is explicit through `face_edge_sign` on `FaceGroup` and `FaceMapCache`, and `CoboundaryOps.delta1` / `delta1_T` apply the signs.
- AMPSS uses protocol version `schema_conditioned_v1`: Stage I greedily selects exact post-quota support by held-out task gain under hard schema/instance/incidence budgets. Coverage is diagnostic only. Stage II learns positive schema-conditioned instance weights only through the linear PSD upper-Hodge term `B^T W B`, with a neutral-to-learned operator ramp.
- Link prediction pools the extended edge cochain into endpoint representations. Auto graph discovery requires `SheafCovPool`; `auto + MeanPool` and direct face-curl readouts are rejected.
- AMPSS has no schema gate, instance-sparsity gate, partial-schema dropout, or auxiliary discovery loss. Training uses the task loss only; coverage and incidence statistics are selection diagnostics or hard Stage-I constraints.
- Link selection removes inner-selection positives and reverse counterparts, then rematerializes Stage II on the full outer-training graph. Selector and final support hashes are distinct.
- Link coverage uses a fixed seeded subset of positive outer-training target edges; graph quotas are per graph and graph costs are training-fold mean incidences.
- Legacy entropy/redundancy keys, old checkpoints, and old tuning states are rejected. Active configs support only the task-aligned probe selector; DCV remains historical study code.

Remaining cautions:

- Production meta-path enumeration currently walks stored directed relations forward, so generated `face_edge_sign` values are always `+1`; non-trivial signs are supported and tested, but not produced by the default DFS path.
- Schema discovery is a prototype and is not the default experiment mode.
- Schema preselection is still a non-differentiable candidate-pool construction step; concrete instances from the selected pool are weighted differentiably afterward.
- Stage-I forward selection is a deterministic task-aligned heuristic under hard budgets; its refitted probe loss is not a submodular objective and carries no approximation guarantee.
- AMPSS results produced before `schema_conditioned_v1` do not validate the current protocol. Run fresh sweeps and repeated-seed ablations with frozen Stage-I selector and materialization manifests.
- Do not claim broad benchmark validation from correctness tests alone. Broader repeated-seed experiments and baselines are still needed for paper-level claims.
- Full-size FastHPP orthogonal configs can be memory-heavy in constrained environments. For smoke tests, use smaller `d` and optionally `use_orthogonal: false`; this is an execution constraint, not a closure/sign/discovery-loss bug.
- The current `src/` model accepts only `orthogonal_map: fasthpp`; its exact non-power-of-two fallback also lives in `src/models/fasthpp.py`. Householder/Cayley/Euler/matrix-exponential API choices are not part of the current model. The legacy `models/` baseline remains separate.
- CUDA node-to-edge map materialization is guarded by `model.node_edge_memory_guard_fraction` (default `0.20`). W&B/Optuna defaults use `d in {16,32}` for full-graph tasks and `{16,32,64}` for graph classification.
- Node- and link-prediction primary tuning remains full-graph; there is no query-batch-size tuning parameter. SheafCovPool always consumes three semantic views and therefore a fixed raw `3x3` Gram descriptor. For graph classification with fixed `data.pooling_type: sheafcov`, tune only the post-Gram width with `--sheafcov_feat_dims 4 8 16`; do not combine it with `--include_pooling_type`.
- W&B tuning state is versioned and fingerprinted. With `--resume`, compatible per-run states
  resume their original sweeps; missing runs start fresh sweeps, and incompatible existing runs
  are preserved while a numbered continuation sweep is created. Each W&B trial runs in a fresh
  worker process. Use `--run_start` and `--run_count` to partition a fixed total `--num_runs`
  campaign across machines while keeping `--base_seed` unchanged; an explicit `--sweep_id`
  remains strict exact-resume mode.

## Coding Rules

- Prefer the current `src/` architecture and existing dataclass config system.
- Keep entry scripts thin: parse args, load config, build components, call `trainer.fit()`.
- Pass config, model, optimizer, logger, and checkpointer explicitly.
- Do not use globals for experiment state.
- Keep model files focused on architecture; training side effects belong in trainers/loggers/checkpointers.
- Use typed returns such as `NamedTuple` for multi-tensor model outputs.
- Save reproducibility artifacts at run start: `config.yaml`, `run.log`, `metrics.jsonl`, `checkpoints/last.pt`, and `checkpoints/best.pt`.
- Do not commit or preserve generated experiment artifacts unless explicitly requested.

## Testing

Use `unittest`, not bare `pytest`, for the current test suite unless the task specifically adds pytest support.

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -m unittest discover -s tests -v
```

The current suite discovers 118 tests. In a clean environment the expected result is:

```text
Ran 118 tests
OK
```

On the current Windows machine, PyG-dependent tests can instead be blocked while importing
`xxhash._xxhash` by Application Control. Treat that as an environment-policy blocker, not an
H2SNN defect; run the complete suite in the clean Colab environment and do not weaken the policy
or add model workarounds.

If a test writes trainer artifacts, make sure file handlers are closed before deleting temporary directories on Windows. `ExperimentLogger.close()` exists for this.

## Training Commands

Manual-schema H2SNN examples:

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\train_h2snn.py --config configs\experiments\acm_h2snn.yaml --device cuda --exp_name acm_h2snn_test
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\train_h2snn.py --config configs\experiments\dblp_h2snn.yaml --device cuda --exp_name dblp_h2snn_test
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\train_h2snn.py --config configs\experiments\imdb_h2snn.yaml --device cuda --exp_name imdb_h2snn_test
```

Prototype auto-discovery configs:

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\train_h2snn.py --config configs\experiments\acm_h2snn_ampss.yaml --device cuda --exp_name acm_h2snn_ampss_test
```

For CPU smoke tests:

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\train_h2snn.py --config configs\experiments\acm_h2snn.yaml --device cpu --exp_name acm_smoke_cpu
```

W&B tuning preflight (does not contact W&B or train):

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\run_h2snn_wandb_tuning.py --config configs\experiments\acm_h2snn_ampss.yaml --num_runs 5 --n_trials 30 --d_values 16 32 --validate_only
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\run_h2snn_wandb_tuning.py --config configs\experiments\mutag_h2snn.yaml --num_runs 5 --n_trials 30 --d_values 16 32 64 --sheafcov_feat_dims 4 8 16 --validate_only
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\run_graph_classification_repeated_cv_wandb_tuning.py --config configs\experiments\mutag_h2snn.yaml --num_repetitions 10 --num_folds 10 --n_trials 30 --unit_start 0 --unit_count 1 --validate_only
```

Graph-classification configs continue to default to five independent stratified
80/10/10 holdout runs. The separate repeated-CV entry point implements the legacy-style
unstratified protocol: 10 shuffled KFold partitions repeated 10 times, with an inner 10%
validation split and one independent W&B sweep per repetition/fold pair. A full campaign is
100 sweeps and approximately `100 * n_trials` model fits. Partition it with `--unit_start` and
`--unit_count` while keeping `--base_seed`, `--split_seed`, `--num_repetitions`, and
`--num_folds` fixed.

For a five-seed campaign split across machines, keep `--num_runs 5` and the same
`--base_seed`, then use `--run_start 0 --run_count 2` on one machine and
`--run_start 2 --run_count 3` on the other. Both machines must use the same experiment/state
directory and must not execute the same run index concurrently.

## Documentation Updates

When changing current model behavior:

- Update tests first or alongside the code.
- Update `docs/h2snn_current_model_documentation.html` if the architecture, limitations, or research claims change.
- Keep `requirements.txt` synchronized with actual imports.
- If command guidance changes, update this `AGENTS.md` and `CLAUDE.md` together so future agents do not fall back to `base`.

## LaTeX Research-Paper Writing And Editing

These instructions apply automatically whenever creating, revising, reviewing, or fact-checking
`paper/*.tex`, `paper/references.bib`, paper figures or captions, appendices, supplementary
material, or a submission/rebuttal derived from them. Write as a careful AAAI researcher and
reviewer: the paper must be technically exact, easy to follow, and accessible to an AI or computer
science reader who is not a specialist in heterogeneous sheaves or Hodge theory.

### Required Grounding Before Writing

- Read the surrounding manuscript section and the relevant entries in `paper/references.bib` before
  editing. Preserve the paper's established macros, notation, labels, terminology, and venue format
  unless the task explicitly changes them.
- Ground descriptions of the current model first in
  `docs/h2snn_current_model_documentation.html`, then in the implemented `src/` code, tests,
  experiment configs, and completed result artifacts. Treat the manuscript, specifications, and
  plans as claims to verify, not as proof that a feature or result exists.
- Use `HETSHEAF.pdf`, `research_paper/HeterSEED.pdf`, and
  `research_paper/Hetero2Net.pdf` as rhetorical style references. Extract their structural habits;
  never copy or closely paraphrase their sentences, and never cite them as evidence for a claim
  merely because they are style references.
- If sources, code, and prose conflict, do not silently choose the stronger claim. State or mark the
  discrepancy and use the narrowest claim supported by the current evidence.

### Adversarial Technical Review Mode

When the user asks to review the paper, switch to an
adversarial technical-review stance. Thinking about writing quality includes where wording hides a technical
ambiguity. Focus on algorithmic novelty, mathematical formulation, optimization design,
technical soundness, evaluation validity, and whether the claimed contribution is actually
supported by the implemented method and evidence. Assume the burden of proof is on the paper:
list every reject-level weakness you can find, explain why a skeptical reviewer would care, and
propose concrete repairs that preserve the overall H2S-NN framework rather than replacing it with a
different project. Prioritize fatal conceptual or formulation problems first, then optimization and
algorithmic gaps, then experimental or ablation gaps. Do not soften the critique with reassurance
unless a specific component is technically solid and important for deciding what to keep.

### Story And Section Architecture

- Build the narrative in the order **context -> concrete problem -> why existing mechanisms are
  insufficient -> key intuition -> method -> evidence -> bounded conclusion**. Each paragraph
  should do one job, open with its main point, and end by resolving that point or motivating the
  next paragraph.
- Follow the useful habits of the three reference papers:
  - **HetSheaf:** motivate the geometry in plain language before formalizing it; connect each new
    construction to a specific limitation; position contributions precisely against adjacent work.
  - **HeterSEED:** make the Background, Motivation, Method, and Contributions beats unmistakable;
    use a concrete observation to motivate each mechanism; explain complementary components before
    giving their technical details.
  - **Hetero2Net:** separate heterogeneity from heterophily, define the challenge and measurements
    before drawing conclusions, and organize empirical claims around explicit questions.
- For the introduction, use a narrowing funnel: real heterogeneous graphs; the joint
  heterogeneity--heterophily problem; the precise representational or operator-level gap; the
  H2S-NN intuition; a compact method preview; then non-overlapping, verifiable contributions.
- Explain a method component in the order **need -> intuition -> mathematical mechanism -> effect
  on information flow -> limitation or scope**. Give a plain-language overview and, when helpful, a
  small typed-path example before dense notation.
- Synthesize related work by mechanism and limitation rather than writing a catalogue of one-sentence
  paper summaries. Compare only along dimensions that matter to the claimed gap, and distinguish
  "not designed for this purpose" from "incapable of it."
- Structure experiments around research questions such as effectiveness, component contribution,
  efficiency, robustness, and failure modes. A results paragraph should state the question, cite the
  relevant table or figure, report the concrete observation, and give a restrained interpretation.

### Academic Style And Accessibility

- Prefer direct, concrete sentences and precise verbs. Define specialized terms at first use and
  distinguish frequently confused concepts explicitly, especially heterogeneity versus heterophily,
  a meta-path schema versus a concrete path instance, and a strict cell/face versus the manuscript's
  "face-like" support.
- Introduce equations with their purpose, define every symbol and index, and explain the operational
  meaning immediately afterward. Include tensor shapes when they prevent ambiguity. Do not make the
  reader reverse-engineer an equation from notation alone.
- Use consistent terminology and notation throughout. Preserve established commands such as
  `\method`, `\ampss`, `\thspool`, `\cF`, `\delta_0`, and `\delta_1`; do not create a synonym merely
  for stylistic variation.
- Prefer one strong claim per sentence. Use signposting and transitions, but avoid repeated previews,
  inflated adjectives ("novel," "powerful," "significant," "comprehensive"), vague references
  ("this," "it," or "the above" without a clear noun), and generic openings that delay the problem.
- Use polished academic English without unnecessary jargon. Keep the grammatical voice and English
  variant consistent within the manuscript; do not globally restyle unrelated passages unless asked.
- Captions must be self-contained enough to identify the setup, compared objects, direction of
  improvement, and meaning of error bars or visual encodings. Do not make a caption assert more than
  the figure or its source data shows.

### Claims, Citations, And Research Integrity

- Cite every substantive external claim close to the clause it supports. Prefer the original paper,
  official proceedings version, or authoritative dataset/documentation source; use surveys for broad
  context, not as a substitute for a primary technical source.
- Verify that a cited source actually supports the exact statement. Do not attach a long citation pile
  to a compound sentence when different sources support different clauses. Split the sentence and map
  citations to claims concretely.
- Never invent or guess a citation key, title, author list, venue, year, DOI, page number, result, or
  BibTeX field. Cross-check each new entry against the source PDF and, when needed, an official
  proceedings page or arXiv record. If verification is unavailable, insert an explicit
  `% TODO(citation): ...` note or report the unresolved claim instead of fabricating support.
- Use the citation commands already established by the manuscript and ensure every cited key exists in
  `paper/references.bib`. Before handoff, audit for undefined citations, uncited new bibliography
  entries when relevant, duplicate entries, and keys whose bibliographic identity does not match the
  prose.
- Attribute comparisons fairly: say what another method does before identifying the remaining gap.
  Avoid universal statements such as "all existing methods" unless the review is demonstrably
  exhaustive. Prefer bounded formulations such as "the compared meta-path methods" or "existing
  node-edge sheaf models in our review."
- Label statements accurately as definitions, intuitions, propositions, empirical observations, or
  hypotheses. Do not present intuition as a theorem, correlation as causation, or a correctness test as
  benchmark validation.
- Report quantitative claims only from traceable tables, logs, or source papers. Include the metric,
  dataset/split, comparison scope, and variability or number of runs when available. Never manufacture
  missing values or use placeholders as completed evidence.
- Keep the implementation boundary explicit: closure edges affect `closure_L0`; the learned
  `delta1` remains path-edge-only; the exact chain condition is encouraged rather than guaranteed;
  default DFS orientations currently yield `+1`; and schema preselection is non-differentiable.
  `SheafCovPool` is the finalized single-level graph readout: invariant Gram descriptors feed
  attention-weighted means and covariances. THS-Pool, Galerkin coarsening, spectral pooling, and
  hierarchical variants are superseded historical experiments; do not describe them as the current readout.
- Never claim broad state of the art, scalability, parameter efficiency, theoretical guarantees, or
  generalization beyond the datasets and evidence actually evaluated. Prefer "best among the compared
  methods under this protocol" over an unqualified "state of the art."

### Safe LaTeX Editing And Verification

- Preserve valid LaTeX structure, cross-references, labels, bibliography style, and the current
  `acmart` setup unless the user requests a venue conversion. Use nonbreaking references such as
  `Table~\ref{...}`, `Figure~\ref{...}`, and `Section~\ref{...}` consistently.
- Edit source files only. Do not hand-edit generated `.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`,
  `.log`, `.out`, `.pdf`, or `.synctex.gz` files.
- After a substantive paper edit, perform a claim--evidence pass, a notation/definition pass, a
  grammar/story pass, and a duplication pass. Remove repeated related-work summaries and repeated
  contribution claims unless the repetition serves a clear navigational purpose.
- When the LaTeX toolchain is available, compile from `paper/` with
  `latexmk -pdf -interaction=nonstopmode -halt-on-error h2snn_manuscript.tex`. Inspect the log for
  undefined citations/references, multiply defined labels, missing files, and material layout
  warnings. Report any blocker; do not conceal it by weakening content or editing generated files.
