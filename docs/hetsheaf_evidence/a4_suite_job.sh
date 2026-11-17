#!/bin/bash
# Card 3 A4: full suite after the v3b merge, on one node (g2n01, its GPU visible as in the pre-merge suite).
#   sbatch docs/hetsheaf_evidence/a4_suite_job.sh
#SBATCH --job-name=hetsheaf-suite
#SBATCH --nodelist=g2n01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-hetsheaf/H2S-NN/logs/slurm-%j.out
set -uo pipefail
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
# pinned CPU kernel environment E, as scripts/slurm/run_tests.sbatch (v3b; tests/pinned_kernels.py)
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "unit=a4_full_suite_post_merge job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
sha256sum src/baselines/adapters/hetsheaf.py src/baselines/registry.py src/baselines/large_lp_training.py
python -c "import torch, os; print('cpu_capability', torch.backends.cpu.get_cpu_capability(), 'MKL_CBWR', os.environ.get('MKL_CBWR'))"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python -m unittest discover -s tests -t . 2>&1 | grep -vE "Warning|warn\(|^\s*$|An experimental|Using sheaf learner" | tail -60
echo "tests_exit=${PIPESTATUS[0]} end=$(date -Is)"
