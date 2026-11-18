#!/bin/bash
#SBATCH --job-name=dryrun-d6b-nci1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/dryrun_scratch/logs/%x_%j.out
# Card 2 D6b (amended): nci1 log-weight diagnosis, lr 1e-3 then 1e-4, in one GPU job (16G; no A6000 / g3an01).
# Submit from /mnt/data/peeradon/h2snn-dryrun/H2S-NN. Records outside the repository; d6b_tables reads them.
set -u
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
echo "head=$(git rev-parse HEAD) host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for lr in 1e-3 1e-4; do
  python ../docs/dryrun_evidence/d6b_nci1_logw.py --lr $lr --out /mnt/data/peeradon/dryrun_scratch/d6b_nci1/lr$lr.json
done
