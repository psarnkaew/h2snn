# M3a-fix F0: every hit and the unit that handles it

Command (login01, branch `part-m3a-fix` at the merge `c9b1f3e`, vendor excluded):

```
cd H2S-NN
grep -rn "concat_mlp\|include_decoder_type\|INCLUDE_DECODER_TYPE" . --include=*.py --include=*.ipynb \
    --include=*.md --include=*.html --include=*.yaml --exclude-dir=vendor
```

Paths relative to `H2S-NN/`. "Kept" = the line refuses `concat_mlp` / the flag, or asserts its absence; it is correct after this card.

| File:line | Text (short) | Unit / action |
|---|---|---|
| `src/configs/config.py:109-112` | `REMOVED_DECODER_MESSAGE` | kept (used by F1's config refusal too) |
| `src/configs/config.py:206-208` | comment "`concat_mlp` (model code kept in `link_readout.py`) is refused …" | **F1**: comment corrected (model code removed) |
| `src/configs/config.py:209`, `:217` | the M3a refusal in `reject_never_tuned_settings` | kept (card: "stays") |
| `src/configs/config.py:672` | `decoder_type` comment `"dot" \| "distmult" \| "concat_mlp"` | **F1**: `"dot" \| "distmult"` |
| `src/configs/config.py:958` | validation `{"dot", "distmult", "concat_mlp"}` | **F1**: only {dot, distmult}; `concat_mlp` raises `REMOVED_DECODER_MESSAGE` |
| `src/models/link_readout.py:21`, `:48`, `:76` | docstring, `__init__` branch, `forward` branch | **F1**: removed |
| `src/models/h2snn_model.py:15` | docstring "`dot` / `distmult` / `concat_mlp` decoders" | **F1** |
| `scripts/run_h2snn_runs.py:39` | comment "its model code stays" | **F1** |
| `scripts/train_h2snn.py:24` | comment "its model code stays" | **F1** |
| `scripts/run_h2snn_wandb_tuning.py:76-77` | comment "(its model code in `src/models/link_readout.py` stays …)" | **F1** |
| `scripts/run_h2snn_wandb_tuning.py:311`, `:318`, `:336` | refusal docstring / message / comment on the removed branch | kept (correct after F1) |
| `tests/test_task_heads.py:149` | loop over (`dot`, `distmult`, `concat_mlp`) | **F1** (stated test change: the `concat_mlp` subcase removed) |
| `tests/test_partM3a_decoder.py:5-149` (many) | refusal tests and docstring | kept |
| `tests/test_search_spaces.py:47`, `:56-57` | refusal assertion | kept |
| `tests/test_wandb_and_runs.py:484`, `:528` | `SimpleNamespace(..., include_decoder_type=False, ...)` in two tuning-state tests (an attribute no code reads any more) | kept, no change (harmless; the card names no change here) |
| `tests/test_wandb_and_runs.py:749` | asserts a notebook source does not contain `--include_decoder_type` | kept (F2 extends the notebook test) |
| `notebooks/h2snn_link_prediction_colab.ipynb:119` | `DECODER_TYPE = 'dot'  # one of: 'dot', 'distmult', 'concat_mlp'` | **F2** |
| `notebooks/h2snn_link_prediction_colab.ipynb:288`, `:331-332` | `INCLUDE_DECODER_TYPE_IN_TUNING = False`; `if …: command.append('--include_decoder_type')` | **F2** |
| `notebooks/h2snn_graph_classification_colab.ipynb:296`, `:339-340` | the same switch | **F2** |
| `notebooks/h2snn_node_classification_colab.ipynb:1` (one-line JSON) | the same switch (two places) | **F2** |
| other `h2snn_*` notebooks (`*_wandb_tuning_colab`, `*_repeated_cv_*`, `h2snn_wandb_paper_tuning_colab`) | no hit | — (covered by F2's test) |
| `CLAUDE.md:884` | "`data.decoder_type` (`dot` \| `distmult` \| `concat_mlp`)" | **F3** |
| `CLAUDE.md:962` | LP head row "`dot` / `distmult` / `concat_mlp`" | **F3** |
| `AGENTS.md:164` | key files: "`dot` / `distmult` / `concat_mlp`" (the same text in AGENTS.md) | **F3** |
| `docs/h2snn_current_model_documentation.html:1553` | changelog: "`--include_pooling_type`/`--include_decoder_type` pattern" | **F3** (note added; text only) |
| `docs/h2snn_current_model_documentation.html:1590` | "`dot` and `concat_mlp` stay relation-agnostic" | **F3** (text only) |

No non-test caller builds `concat_mlp` (grep: the only constructions are `LinkPredictionReadout` via `h2snn_model.py:669-677` from `config.data.decoder_type`, which F1's config validation now refuses), so the stop condition "a caller outside tests that builds `concat_mlp` from code the card does not name" does not apply.
