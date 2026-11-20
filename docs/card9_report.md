# Card 9 report — ablation and analysis scripts to pre-registration v3 (spec 1.29)

Card: `docs/H2S-NN_card9_card.md`. Binding: `docs/ablation_analysis_preregistration_v3.md`.

## W0 — branch, merge, scope

- `origin/v3b` = `fc0ebaa605280f2d987c4848cc3109c7d3b66680` (checked); `origin/design-docs` contains `b9a2a2f0d6d6a8df255d476108b73e458f7cc693` (checked).
- `card9` created from `fc0ebaa`; `git merge --no-ff b9a2a2f` → `313eb23`.

Files this card touches (one line each):

| Unit | File |
|---|---|
| W1 | `H2S-NN/scripts/ablation/arms.py` |
| W1 | `H2S-NN/scripts/ablation/seed_runs.py` |
| W2 | `H2S-NN/scripts/ablation/param_table.py` |
| W3 | `H2S-NN/scripts/ablation/stats.py` |
| W4 | `H2S-NN/scripts/analysis/q3_schema.py` |
| W5 | `H2S-NN/scripts/perturbation/rules.py` (new) |
| W5 | `H2S-NN/scripts/perturbation/figure.py` (absent methods) |
| W6 | `H2S-NN/scripts/analysis/face_effect.py` |
| W6 | `H2S-NN/scripts/analysis/mechanism_common.py` |
| W7 | `H2S-NN/scripts/analysis/two_regime.py` |
| W8 | `H2S-NN/scripts/tsubame/make_cell_list.py` |
| W9 | `H2S-NN/tests/test_card7_ablation.py`, `test_card7_stats.py`, `test_card7_q3.py`, `test_card7_tsubame.py` (v2 → v3 assertions) |
| W9 | `H2S-NN/tests/test_card9_v3.py` (new: T1–T7) |
| W9 | `docs/card9_evidence/` (job scripts and logs) |
| — | `docs/card9_report.md` (this file) |
