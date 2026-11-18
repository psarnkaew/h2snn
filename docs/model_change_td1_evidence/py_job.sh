#!/bin/bash
# Model change TD-1: one Python script on CPU (environment E) in a snapshot (snapshot_submit.sh).
#SBATCH --job-name=td1-py
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/model_change_td1_evidence/py_%j.txt
set -uo pipefail
cd "${SNAP:?}/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "snapshot=${SNAP} $(cat ${SNAP}/SNAPSHOT.txt)"
echo "git=$(git rev-parse HEAD)"
python "$@" 2>&1 | grep -vE "Warning|warn|import torch_geometric|autocast"
echo "exit=${PIPESTATUS[0]} end=$(date -Is)"
