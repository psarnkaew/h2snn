#!/bin/bash
# Card 15 targeted tests (author 2026-11-22: no full suite for card 15): regenerate the baseline notebooks, then the
# test modules that list acm_pa / acm_pp / the derived LP tasks (LP, derived-task, registry, adapter tests). One GPU.
#SBATCH --job-name=c15-fit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=0:40:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-card15/docs/card15_evidence/slurm-%j-fit.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card15/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 WANDB_MODE=disabled
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
python -m unittest -v tests.test_partP_derived_lp
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
