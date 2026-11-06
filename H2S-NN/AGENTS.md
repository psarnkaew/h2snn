# AGENTS.md - H2S-NN Agent Instructions

This file gives Codex and other coding agents the project-specific rules for this repository.
`CLAUDE.md` holds the same rules in more detail (coding-style templates, architecture tables);
when the two differ, report it and follow `CLAUDE.md`.

> **H2S-NN is mid-revision.** `src/` holds one model (vertex-2-cell sheaf Laplacian, plan §4) with
> node-classification, link-prediction and graph-classification heads; the submitted model (v1)
> and AMPSS are on the git tag `v1-submitted`. The plan lives in `../docs/`. Read "Revision
> Status" below before any task.

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
```

If using an absolute Python path in automation, use:

```powershell
C:\Users\Acer\anaconda3\envs\h2snn\python.exe -m unittest discover -s tests -v
```

Dependency notes:

- Python `3.12`; PyTorch `torch==2.6.*` with CUDA 12.4 wheels; `torch_geometric`, `torch_scatter`, `torch_sparse`.
- `src/` does not need `torch_householder`; v1 orthogonal maps use the pure-PyTorch compact-WY implementation in `src/models/fasthpp.py`. Only the legacy CSNN/NSD baselines use `torch_householder`.
- Pins live in `requirements/` (`core.txt` is required; the `baselines-*.txt` groups back individual baselines). `requirements.txt` aggregates them. Do not install it into `base`.

Sanity check:

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -c "import torch, torch_geometric, torch_scatter, torch_sparse; print(torch.__version__, torch.cuda.is_available())"
```

## Revision Status

`src/` holds one model: H2S-NN with the vertex-2-cell sheaf Laplacian (plan §4), shared by three task heads (node classification, link prediction, graph classification; plan §4.8, decision D6). The submitted model (v1) and AMPSS were deleted and are preserved on the git tag `v1-submitted`. No accuracy result exists yet for any task.

### Sources of truth, in this order

The planning documents live in `../docs/` (from `H2S-NN/`).

1. `CLAUDE.md` and this file - environment and coding conventions. Binding.
2. `../docs/H2S-NN_v2_plan.md` - agreed decisions D1-D6. Do not reopen an agreed decision; if evidence contradicts one, stop and report.
3. `../docs/H2S-NN_fix_spec.md` - verified facts F1-F18, hypotheses H1-H12, protocol fixes (§6), writing decisions (§7). In force except where the v2 plan supersedes it (P2, P3, P5, P6).
4. `../docs/H2S-NN_agent_prompt.md` - role, evidence rules, the mandatory self-correction loop, and task cards. Task cards D1 and D4 are done (reports `../docs/H2S-NN_D1_report.md`, `../docs/H2S-NN_D4_report.md`).
5. The code. When a document and the code disagree, the code describes what runs; report the disagreement.

Context: `../docs/critical_reviewer_reviews.md` (reviews); `../docs/h2snn_manuscript_paper.pdf` and `../docs/h2snn_manuscript_material.pdf` (submitted version); manuscript source in `paper/`.

### Decisions

| ID | Decision | State |
|---|---|---|
| D1 | Factorized, matrix-free, conformal-orthogonal restriction maps `F = c * R_{psi,side} * H_k(u)`; `model.map_family: factorized \| dense` (dense = comparison family), `model.householder_k`. D1a: one scale `c_e` per edge, shared by both sides | Implemented, unit-tested; accuracy experiments (plan §1.6) not run |
| D2 | Redesign AMPSS; mechanism open. Interim: fixed literature schemas (the presets) | Open; v1 AMPSS deleted |
| D3 | Scalability: normalized Laplacians (full-graph degrees), `eta = sigmoid(theta)`, per-batch path cells, `NeighborLoader` | Normalization and bounded step implemented (D4); mini-batching not started |
| D4 | Path 2-cells couple to every boundary node through the vertex-2-cell sheaf Laplacian | Implemented with the plan §4.6 ablation switches; ACM memory and smoke run only |
| D5 | Baseline provenance and IMDB heterophily fixes (parallel) | Not started |
| D6 | One model; link prediction and graph classification ported to it (heads change only where forced) | Implemented; no accuracy run |

### Rejected - do not build (plan §4.8)

A sheaf-consistency link decoder; per-relation / per-schema energy features for graph classification; a disjoint message/supervision split for link prediction (the HGB protocol is kept).

### Evidence and conduct rules

- Tag every non-trivial claim: `VERIFIED-CODE` (`file:line`), `VERIFIED-RUN` (command + output), `VERIFIED-SOURCE`, `DERIVED`, `UNVERIFIED`, or `PROPOSAL`. Never claim a test passes, a number improved, or a bug is fixed without showing the run output.
- Never invent file paths, functions, APIs, config keys, numbers, citations or results. Confirm an API exists in the `h2snn` environment before using it.
- No long training jobs, sweeps or multi-seed experiments without explicit approval. Short smoke tests and microbenchmarks are fine.
- Results produced before the revision do not validate the current model. Label every baseline number rerun / published / quoted and trace it to an artifact.
- Config loading is strict: every section rejects unknown keys, so old v1/AMPSS configs and checkpoints fail loudly. Do not add compatibility shims.
- Forbidden fixes: loosening tolerances, deleting or skipping tests, silencing exceptions, special-casing test inputs, editing the spec to fit the code.
- Stay inside the task card's scope; list out-of-scope problems instead of fixing them. Do not commit or push unless asked.

## Codebase Map

- `src/`, `scripts/`, `configs/experiments/`, `tests/` - H2S-NN.
- `vendor/` - every external baseline: `vendor/csnn/` (legacy homogeneous CSNN: `models/`, `exp/`, `utils/`, `gnn_evaluation.py`), the committed `vendor/HINPool/` and `vendor/HINormer/`, and empty clone targets (`vendor/H2GB/`, `vendor/SeHGNN/`, `vendor/HGB/`, ...). Leave the empty folders in place. `src/baselines/runtime/vendor_paths.py` is the single place mapping each baseline to its folder; `src/baselines/` holds the adapters. Do not add H2S-NN features under `vendor/`.
- `folds/` - TU graph-classification splits, read relative to the working directory: run from the repository root. Legacy CSNN runs also need `PYTHONPATH=vendor/csnn` (see `CLAUDE.md`).
- `references/` - reference PDFs (`HETSHEAF.pdf`, `background/`, `metapath_search/`, `rl_and_search/`, `readout/`, `misc/`).
- `docs/h2snn_current_model_documentation.html` and the `docs/sheafcovpool_*` notes - describe the deleted v1 model (tag `v1-submitted`); history, not `src/`. `docs/pyg_dataset_eda_homophily_report.html` - dataset homophily EDA cited by the plan.
- `notebooks/` - Colab notebooks checked by `tests/test_baselines.py` and `tests/test_wandb_and_runs.py` (`baselines_*.ipynb`, `h2snn_*_colab.ipynb`, and `notebooks/baselines/`, generated by `scripts/generate_baseline_notebooks.py` - edit the generator, never those files). Keep them committed without outputs.

Key files: `src/models/h2snn_model.py` (`H2SNN`: encoder, ablation switches, node head; `ModelDims`), `vertex_path_laplacian.py` (path-cell operator, normalized `Delta_0 + gamma L_P`), `map_operators.py` (D1/D1a operators, `PathTransportOperator`), `sheaf_maps.py` (node-to-edge maps), `link_readout.py` (`dot` / `distmult` / `concat_mlp` on `[x_K, h0]`), `graph_readout.py` (sum of `[x_K, h0]` -> BatchNorm -> MLP); `src/trainers/{trainer,link_prediction_trainer,graph_classification_trainer}.py`; config `src/configs/config.py` (`data`, `model`, `ablation`, `optim`). Tests: `tests/test_factorized_maps.py`, `tests/test_v2_model.py`, `tests/test_task_heads.py`. The model's equations and the head table are under "Architecture" in `CLAUDE.md`.

## Coding Rules

- Prefer the current `src/` architecture and existing dataclass config system. Validate every new config key in its dataclass; no hard-coded hyperparameters outside the config.
- Keep entry scripts thin: parse args, load config, build components, call `trainer.fit()`.
- Pass config, model, optimizer, logger, and checkpointer explicitly. No globals.
- Keep model files focused on architecture; training side effects belong in trainers/loggers/checkpointers.
- Use typed returns such as `NamedTuple` for multi-tensor model outputs.
- Save reproducibility artifacts at run start: `config.yaml`, `run.log`, `metrics.jsonl`, `checkpoints/last.pt`, and `checkpoints/best.pt`.
- Minimal diffs in the repo's style. Do not commit generated experiment artifacts unless explicitly requested.

## Testing

Use `unittest`, not bare `pytest`:

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -m unittest discover -s tests -v
```

Do not rely on a remembered test count: record the count and result before your change and compare after it. Any newly failing existing test blocks progress. `tests/test_task_heads.py` compares node-classification outputs and gradients bit-for-bit with `../docs/heads_evidence/v2_golden.pt`.

On this Windows machine, PyG-dependent tests can be blocked while importing `xxhash._xxhash` by Application Control. Treat that as an environment-policy blocker, not an H2S-NN defect; report it, run the complete suite in the clean Colab environment, and do not weaken the policy or add model workarounds.

If a test writes trainer artifacts, close file handlers (`ExperimentLogger.close()`) before deleting temporary directories on Windows.

## Training Commands

Respect the approval rule for anything longer than a smoke test. The task follows from `data.dataset_name`.

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\train_h2snn.py --config configs\experiments\acm_h2snn.yaml --device cuda --exp_name acm_h2snn_test
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\train_h2snn.py --config configs\experiments\acm_h2snn.yaml --device cpu --exp_name acm_smoke_cpu
```

Configs: node classification `acm`, `dblp`, `imdb`; link prediction `lastfm`, `amazonbook`, `pubmed`, `youtube`, `amazon` (decoder per dataset via `data.decoder_type`); graph classification `mutag`, `enzymes`, `nci1`, `proteins` (all `*_h2snn.yaml`). Ablations go in the `ablation:` block (all off by default).

W&B tuning preflight (does not contact W&B or train):

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\run_h2snn_wandb_tuning.py --config configs\experiments\acm_h2snn.yaml --num_runs 5 --n_trials 30 --d_values 16 32 --validate_only
```

W&B tuning state is versioned and fingerprinted; `--resume` continues compatible sweeps and creates numbered continuation sweeps for incompatible ones. To split a campaign across machines, keep `--num_runs` and `--base_seed` fixed and use `--run_start`/`--run_count`; never run the same run index concurrently.

## Documentation Updates

- Update tests first or alongside the code.
- When a planned component lands, record it in the task's final report (format in `../docs/H2S-NN_agent_prompt.md`) and update the "State" column of the decision tables in this file and `CLAUDE.md`.
- Keep `requirements/` synchronized with actual imports, and `src/baselines/runtime/dependencies.py` with the baseline groups.
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
- The manuscript describes **v1**. Ground any description of what v1 does in the code on the git
  tag `v1-submitted` (its `src/` code, tests, experiment configs and completed result artifacts), then in
  `docs/h2snn_current_model_documentation.html` (v1 only). Ground any description of the revised
  model in `../docs/H2S-NN_v2_plan.md` and `../docs/H2S-NN_fix_spec.md` (§7, Claims and writing), and describe
  a v2 component as implemented only once its code and tests exist. Treat the manuscript,
  specifications and plans as claims to verify, not as proof that a feature or result exists.
- Use `references/HETSHEAF.pdf`, `references/metapath_search/HeterSEED.pdf`, and
  `references/metapath_search/Hetero2Net.pdf` as rhetorical style references. Extract their structural habits;
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
- Keep the implementation boundary explicit.
  - **v1 (submitted model, git tag `v1-submitted`):** closure edges are first-class 1-cells inside `delta0`;
    `delta1` spans the full boundary cycle and `delta1 . delta0 = 0` holds by construction;
    edge-to-face maps are built by transport from one free per-instance map `A_f`; faces are simple
    paths; incidence signs are `+1` because enumeration is forward-only; schema preselection is
    non-differentiable. `SheafCovPool` is v1's single-level graph readout; THS-Pool, Galerkin
    coarsening, spectral pooling and hierarchical variants are superseded historical experiments.
  - **v2 wording rules (plan §4.5), for the revision:** the whole 2-complex carries a genuine
    cellular sheaf (functoriality enforced by transport); states are cochains, not sections, so write
    "diffuses over a cellular sheaf on the 2-complex, including its node→2-cell maps", never "the
    model makes the graph a sheaf". Call the node operator the **vertex–2-cell sheaf Laplacian**, not
    the Hodge Laplacian of the 2-complex (Lemma A′ rules that out). The closure edge belongs to the
    definition of the 2-cell, not the computation. Distinguish the construction from sheaf
    hypergraph models (Duta et al. 2023) and MAGNN as plan §4.5 describes.
  - **Scope of claims:** follow fix spec §7 decision A/B on whether "heterophilic" stays in the
    framing; SheafCovPool and graph classification move to the appendix or a separate paper.
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
