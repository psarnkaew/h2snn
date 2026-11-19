#!/bin/bash
# Card 5: one Python step from H2S-NN/ of this worktree on one GPU (sbatch; the GPU is the job's own).
set -uo pipefail
cd /mnt/data/peeradon/h2snn-ablprep/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"
echo "dirty: $(git status --short -- src tests scripts | tr '\n' ' ')"
sha256sum "$1" src/models/h2snn_model.py src/models/sheaf_energy.py src/configs/config.py
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$@"; status=$?
echo "end=$(date -Is) exit=${status}"; exit $status
