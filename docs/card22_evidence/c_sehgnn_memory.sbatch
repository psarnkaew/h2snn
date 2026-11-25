#!/bin/bash
# Card 22 C: SeHGNN IEEE-CIS-G killed attempt 3 (sweep cj1c2xop run 4), peak host RSS with 6 and 0 loader workers.
#SBATCH --job-name=c22-sehgnn-mem
#SBATCH --partition=lab
#SBATCH --nodelist=g3an01
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-card22/docs/card22_evidence/job%j_c_sehgnn_memory.log
set -uo pipefail
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
cd /mnt/data/peeradon/h2snn-card22/H2S-NN
echo "job=${SLURM_JOB_ID} node=$(hostname) git=$(git rev-parse HEAD) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
P='{"num_hops": 4, "embed_size": 256, "hidden": 256, "batch_size": 256, "n_layers_2": 3, "dropout": 0.006378, "input_drop": 0.73, "lr": 0.00060573, "residual": true, "weight_decay": 0.0001}'
for w in 6 0; do
  H2SNN_LOADER_WORKERS=$w python ../docs/card22_evidence/c_sehgnn_memory.py "$P" 2>&1 | grep -v "can only test a child"
  echo "workers=$w exit=${PIPESTATUS[0]}"
done
