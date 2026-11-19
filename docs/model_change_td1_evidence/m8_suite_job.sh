#!/bin/bash
# Model change TD-1, M8: on the merge head (model-td1 + v3b 87c2e4a), one full suite and the golden / mb0
# compares (environment E, same node as the M4 capture, g2n01). One GPU not shared. --mem = the highest
# recorded peak among the merged tests (freebase baselines, job 17690: 26.9 GiB) + 30 % = 35G.
# Runs in a snapshot (snapshot_submit.sh).
#SBATCH --job-name=td1-m8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=35G
#SBATCH --time=4:00:00
#SBATCH --nodelist=g2n01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/model_change_td1_evidence/m8_suite_%j.txt
set -uo pipefail
cd "${SNAP:?run through snapshot_submit.sh}/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
WT=/mnt/data/peeradon/h2snn-model/docs
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "snapshot=${SNAP} $(cat ${SNAP}/SNAPSHOT.txt)"
echo "git=$(git -C /mnt/data/peeradon/h2snn-model rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
lscpu | grep 'Model name'
echo "===== suite"
python -m unittest discover -s tests -v 2>&1 | grep -E " \.\.\. |^Ran |^OK|^FAILED|^ERROR:|^FAIL:|Error|assert" | grep -vE "Warning"
echo "suite_exit=${PIPESTATUS[0]}"
F='GBWarning|expandable|allocation pattern|PYTORCH_CUDA|gb_warning|An experimental|See https|and set the|FutureWarning|autocast|_flatten\("", torch.load|torch.load|DeprecationWarning|import torch_geometric'
export CUDA_VISIBLE_DEVICES=""
echo "golden sha256: $(sha256sum ${WT}/legacy_evidence/v3b_golden_cluster.pt | cut -d' ' -f1)"
echo "mb0 sha256:    $(sha256sum ${WT}/partMB_evidence/mb0_reference_cluster.pt | cut -d' ' -f1)"
echo "===== compare golden"
python ../docs/legacy_evidence/v3b_golden.py compare ${WT}/legacy_evidence/v3b_golden_cluster.pt 2>&1 | grep -vE "${F}"; echo "golden_compare_exit=${PIPESTATUS[0]}"
echo "===== compare mb0"
python ../docs/partMB_evidence/mb0_reference.py compare ${WT}/partMB_evidence/mb0_reference_cluster.pt 2>&1 | grep -vE "${F}"; echo "mb0_compare_exit=${PIPESTATUS[0]}"
echo "end=$(date -Is)"
