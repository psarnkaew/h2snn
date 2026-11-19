#!/bin/bash
# Card 7 amendments 9-10: the job-17862 table recomputed with eta_v = per-path mean and the eta groups, on ACM, DBLP,
# IMDB and Freebase, from structure only (CPU; nothing trained).
# --mem: job 17862 (the same meta-path pass on ACM + DBLP) peaked at 1.7 GB; + 30 % = 2.2 -> 3G. Card 6's Freebase
# (symmetrised) meta-path pass peaked at 0.7 GB.
#SBATCH --job-name=c7-etagrp
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=1:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/eta_groups_%j.txt
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID} node=$(hostname) git=$(git rev-parse HEAD) dirty=$(git status --short -- src scripts tests | wc -l) start=$(date -Is)"
python /mnt/data/peeradon/h2snn-model/docs/card7_evidence/eta_groups_table.py 2>&1 | grep -vE "Warning|warn|Processing|Done!|torch_geometric|sparse.mm"
echo "exit=${PIPESTATUS[0]} end=$(date -Is)"
