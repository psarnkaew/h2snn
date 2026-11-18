#!/bin/bash
# Model change TD-1, M6: cost smoke on ACM at the card 2 central configuration, v3b 9b7ae0b (detached
# worktree /mnt/data/peeradon/h2snn-model-base) and the 1.25 model (snapshot of model-td1) in one job on
# one GPU, alternating (v3b, td1, v3b, td1). The script is the model-td1 one (--code_root picks the
# tree). Host --mem: no recorded host peak for this script (card 2/v3B recorded GPU peaks only, 2.8 GB
# on ACM at r 42); 12G, peak RSS from sacct (MaxRSS) after the job.
#SBATCH --job-name=td1-cost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/model_change_td1_evidence/cost_%j.txt
set -uo pipefail
cd "${SNAP:?run through snapshot_submit.sh}/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
EVID=/mnt/data/peeradon/h2snn-model/docs/model_change_td1_evidence
BASE=/mnt/data/peeradon/h2snn-model-base/H2S-NN
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "snapshot=${SNAP} $(cat ${SNAP}/SNAPSHOT.txt)"
echo "git(model-td1)=$(git -C /mnt/data/peeradon/h2snn-model rev-parse HEAD) git(base)=$(git -C ${BASE} rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CFG=$EVID/acm_central.yaml
for rep in 1 2; do
  for tree in v3b td1; do
    if [ "$tree" = v3b ]; then ROOT=$BASE; else ROOT=$PWD; fi
    echo "===== ${tree} rep ${rep} (code_root ${ROOT})"
    python scripts/v3b_cost_smoke.py --code_root "$ROOT" --config "$CFG" --any_r \
        --device cuda --epochs 3 --final_tolerance_step --out "$EVID/cost_${tree}_rep${rep}.json" \
        2>&1 | grep -E '^\{"epoch"|draw_seed_offset|wrote|Error|error|Traceback' | cut -c1-400
  done
done
echo "end=$(date -Is)"
