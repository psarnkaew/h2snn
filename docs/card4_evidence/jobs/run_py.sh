#!/bin/bash
# Card 4: run one Python script of this card from H2S-NN/ in the ablation-prep worktree.
#   srun --jobid=<id> --overlap bash docs/card4_evidence/jobs/run_py.sh scripts/ablation/variance_gate.py --out ...
#   sbatch [--array ...] docs/card4_evidence/jobs/run_py.sh scripts/known_answer/...py ...
# Prints the git head, node, job id and the sha256 of the script and of the edited src files (NFS lag).
set -uo pipefail
WT=/mnt/data/peeradon/h2snn-ablprep
cd "$WT/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
echo "job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "dirty: $(git status --short -- src tests scripts | tr '\n' ' ')"
sha256sum "$1" src/models/map_operators.py src/models/sheaf_maps.py src/models/h2snn_model.py \
    src/configs/config.py
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
python "$@"
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
