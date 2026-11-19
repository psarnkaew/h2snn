#!/bin/bash
# Copy of the script job 17862 ran (it ran from /mnt/data/peeradon/h2snn-model-snap/etadiag/; a first submission
# from the node-local scratchpad, job 17861, found no script). Diagnostic only; not a card unit.
#SBATCH --job-name=c7-etadiag
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=0:30:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/phaseb_eta_diag_%j.txt
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES=""
echo "job=${SLURM_JOB_ID} node=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"
python /mnt/data/peeradon/h2snn-model/docs/card7_evidence/phaseb_eta_diag.py 2>&1 | grep -vE "Warning|warn|Processing|Done!|torch_geometric"
echo "end=$(date -Is)"
