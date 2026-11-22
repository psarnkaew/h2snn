#!/bin/bash
# Card 17 smoke: GCN and H2G-former on IEEE-CIS-G, published files, seed 0, 4 steps, one lab GPU.
# --mem: measured peak 11.1 GiB (sacct MaxRSS, measuring job 18382 at 48G) + 30 % = 14.5 -> 15G.
#SBATCH --job-name=c17-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=1:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-card17/docs/card17_evidence/slurm-%j-smoke.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card17/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b WANDB_MODE=disabled
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is) mem=${SLURM_MEM_PER_NODE:-?}M"
echo "git=$(git rev-parse HEAD) dirty=$(git status --porcelain -- src configs | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
EV=../docs/card17_evidence
status=0
for model in gcn h2gformer; do
  echo "== ${model}"
  python $EV/smoke_ieee.py --model ${model} --steps 4 --eval-cap 512 \
      --out $EV/smoke_${model}_${SLURM_JOB_ID}.json 2> $EV/smoke_${model}_${SLURM_JOB_ID}.err | tail -30
  rc=${PIPESTATUS[0]}
  grep -v -i warn $EV/smoke_${model}_${SLURM_JOB_ID}.err | grep -E "Error|Traceback|error" | tail -20
  echo "smoke ${model} exit=${rc}"; [ ${rc} -ne 0 ] && status=1
done
echo "end=$(date -Is) exit=${status}"
exit ${status}
