#!/bin/bash
# Card 3 B2: full suite once at the B1 head, on one node (g2n01, its GPU visible as in A4), --mem 32G (prompt).
#   sbatch docs/hetsheaf_evidence/b2_suite_job.sh
#SBATCH --job-name=fb-b2-suite
#SBATCH --nodelist=g2n01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-hetsheaf/docs/freebase_baselines_evidence/b2_suite_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
# pinned CPU kernel environment E, as scripts/slurm/run_tests.sbatch (v3b; tests/pinned_kernels.py)
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "unit=b2_full_suite job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
sha256sum src/baselines/adapters/_hgb_generic.py src/baselines/adapters/pshgcn.py src/baselines/adapters/h2sgnn.py src/baselines/adapters/sehgnn.py src/baselines/registry.py src/baselines/minibatch.py
python -c "import torch, os; print('cpu_capability', torch.backends.cpu.get_cpu_capability(), 'MKL_CBWR', os.environ.get('MKL_CBWR'))"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python -m unittest discover -s tests -t . 2>&1 | grep -vE "Warning|warn\(|^\s*$|An experimental|Using sheaf learner" | tail -60
echo "tests_exit=${PIPESTATUS[0]} end=$(date -Is)"
