#!/bin/bash
# Merge c12+c14: unittest on the merge worktree (as scripts/slurm/run_tests.sbatch: pinned kernel environment E;
# W&B tests NOT under WANDB_MODE=disabled). No arguments = the full suite.
#   sbatch --gres=gpu:1 --mem=39G docs/merge_c12_c14_evidence/tests.sbatch [unittest targets...]
#SBATCH --job-name=mc-tests
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-merge/H2S-NN/logs/slurm-%j.out
set -uo pipefail
cd /mnt/data/peeradon/h2snn-merge/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
unset WANDB_MODE
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is) WANDB_MODE=${WANDB_MODE:-unset}"
echo "git=$(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())"
if [ "$#" -eq 0 ]; then
    python -m unittest discover -s tests -v
else
    python -m unittest -v "$@"
fi
suite=$?
echo "end=$(date -Is) suite_exit=${suite}"
exit "${suite}"
