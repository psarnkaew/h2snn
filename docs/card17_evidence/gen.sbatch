#!/bin/bash
# Card 17 step 2: the C1 generator: check it reproduces every existing H2GB file, then write the five IEEE-CIS-G files.
#SBATCH --job-name=c17-gen
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-card17/docs/card17_evidence/slurm-%j-gen.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card17/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES=""
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is) git=$(git rev-parse HEAD)"
python ../docs/card17_evidence/gen_published.py check || { echo "check failed: not writing"; exit 1; }
python ../docs/card17_evidence/gen_published.py write gcn:ieee-cis-g gat:ieee-cis-g rgcn:ieee-cis-g han:ieee-cis-g h2gformer:ieee-cis-g
echo "end=$(date -Is) exit=$?"
