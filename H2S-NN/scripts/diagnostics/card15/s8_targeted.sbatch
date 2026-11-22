#!/bin/bash
# Card 15 targeted tests (author 2026-11-22: no full suite for card 15): regenerate the baseline notebooks, then the
# test modules that list acm_pa / acm_pp / the derived LP tasks (LP, derived-task, registry, adapter tests). One GPU.
#SBATCH --job-name=c15-tests
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=2:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-card15/docs/card15_evidence/slurm-%j-targeted.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card15/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 WANDB_MODE=disabled
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
python scripts/generate_baseline_notebooks.py || exit 3
git status --short -- notebooks
python -m unittest -v tests.test_baselines tests.test_partP_derived_lp tests.test_hgb_oracle tests.test_lp_derived \
  tests.test_data_layout tests.test_lp_valid_files tests.test_lp_feature_rule tests.test_partP_h2gformer \
  tests.test_baselines_hetsheaf tests.test_experiment_configs tests.test_card8_lp_step tests.test_card8_magnn \
  tests.test_card8_rgcn tests.test_budget_table tests.test_heart_negatives tests.test_search_spaces \
  tests.test_sweep_batch_size tests.test_dry_run_flag tests.test_fit_check tests.test_tsubame_cell_list \
  tests.test_tsubame_phases
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
