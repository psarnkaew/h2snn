#!/bin/bash
#SBATCH --job-name=c21-tests
#SBATCH --partition=lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-card21/H2S-NN/logs/slurm-%j.out
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card21/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
echo "job=${SLURM_JOB_ID} node=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"
git status --short -- src tests
python -c "import torch, torch_sparse, torch_geometric; print(torch.__version__, torch_sparse.__version__, torch_geometric.__version__)"
python -m unittest -v tests.test_card21_loader; s1=$?
echo "=== B1 probe (run 1)"; python ../docs/card21_evidence/b1_probe.py
echo "=== B1 probe (run 2)"; python ../docs/card21_evidence/b1_probe.py
echo "end=$(date -Is) tests_exit=${s1}"
exit $s1
