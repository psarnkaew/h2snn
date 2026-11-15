#!/bin/bash
# CS8-4: after editing requirements/baselines-dgl.txt, the tests that name the DGL requirement group.
#   srun --jobid=<allocation> --overlap bash docs/cluster_evidence3/cs8_4_job.sh   (or sbatch --gres=gpu:1)
#SBATCH --job-name=cs8-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
nvidia-smi --query-gpu=name --format=csv,noheader
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads())'
echo "===== baselines-dgl.txt code lines"; grep -v '^#' requirements/baselines-dgl.txt | grep -v '^\s*$'
echo "===== tests"
python -m unittest -v tests.test_baselines.RequirementGroupTests tests.test_baselines.DglCudaGateTests tests.test_baselines.SplitInstallNotebookTests 2>&1
echo "tests_exit=$? end=$(date -Is)"
