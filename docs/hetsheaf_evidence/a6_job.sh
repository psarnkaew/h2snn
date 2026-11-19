#!/bin/bash
# Card 3 A6.1: H-Pokec U-U at the A5.3 central point (d 3, layers 5, hidden_channels 16; card 2's central_point rule,
# categoricals at the adapter's default: sheaf diag/type_concat, decoder_type dot), seed 0, one run, one epoch + validation.
# No metric. Not rerun on failure.  sbatch docs/hetsheaf_evidence/a6_job.sh
#SBATCH --job-name=hetsheaf-a6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --exclude=g2n01,g3an01
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-hetsheaf/docs/hetsheaf_evidence/a6_pokec_uu_central.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
sha256sum src/baselines/adapters/hetsheaf.py ../docs/hetsheaf_evidence/a4_dryrun.py ../docs/hetsheaf_evidence/a5_dryrun.py
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python ../docs/hetsheaf_evidence/a5_dryrun.py --cell link_prediction:h-pokec_uu --kind central
echo "end=$(date -Is)"
