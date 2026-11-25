#!/bin/bash
# Card 22 tests 1-2 on the fix (card22 worktree) and on 2e3e160 (same test file), run_tests.sbatch settings.
#SBATCH --job-name=c22-test2-cpu
#SBATCH --partition=lab

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-card22/docs/card22_evidence/job%j_test2_cpu.log
set -uo pipefail
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID} node=$(hostname) WANDB_MODE=${WANDB_MODE:-unset}"
for tree in /mnt/data/peeradon/h2snn-card22 /mnt/data/peeradon/h2snn-c22old; do
  cd $tree/H2S-NN && echo "=== tree=$tree git=$(git rev-parse HEAD) dirty: $(git status --short src | tr '\n' ' ')"
  python -m unittest -v tests.test_card22_h2gb_failures.FeaturelessTypesTests 2>&1 | grep -v "can only test a child process" ; echo "exit=${PIPESTATUS[0]}"
done
