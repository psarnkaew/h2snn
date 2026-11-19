#!/bin/bash
#SBATCH --job-name=dryrun-d4b-tests
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/dryrun_scratch/logs/%x_%j.out
# Card 2 D4b: targeted tests on the CPU (no GPU), pinned kernel environment E. Submit from H2S-NN/.
set -u
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "head=$(git rev-parse HEAD) host=$(hostname) job=$SLURM_JOB_ID"
git status --short | grep -v '^?? data$' | sed 's/^/worktree: /'
python -m unittest -v ${TESTS:-tests.test_large_lp_h2snn tests.test_partMB_link tests.test_large_lp tests.test_baselines_large_lp tests.test_budget_table} 2>&1 | grep -E ' \.\.\. |^(FAIL|ERROR|Ran|OK|FAILED)|Error|Traceback|File "/mnt' | tail -80
