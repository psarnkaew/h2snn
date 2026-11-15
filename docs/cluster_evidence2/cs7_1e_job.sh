#!/bin/bash
# CS7-1e: re-run the CS5 failures (DGL, torch_householder) and the CS7-1a tests in h2snn-b on one GPU.
#   srun --jobid=<allocation> --overlap bash docs/cluster_evidence2/cs7_1e_job.sh   (or sbatch --gres=gpu:1)
#SBATCH --job-name=cs7-1e
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=2:00:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
python -c "import torch, dgl, importlib.metadata as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'dgl', dgl.__version__, m.version('dgl'))"
python -m unittest -v tests.test_baselines tests.test_partP_freebase tests.test_partP_derived_lp \
    tests.test_partPfix_data_root tests.test_check_env tests.test_env_b_install 2>&1
status=$?
echo "unittest_exit=${status} end=$(date -Is)"
exit ${status}
