#!/bin/bash
# Model change TD-1, M4: re-capture the v3-B golden and the mb0 reference on one node (environment E,
# 1 thread, as LR-1 / CS8), then compare both once on the same node (second run). Runs in a snapshot
# (snapshot_submit.sh); the references are written to the worktree's docs/.
#SBATCH --job-name=td1-m4
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --nodelist=g2n01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/model_change_td1_evidence/m4_capture_%j.txt
set -uo pipefail
cd "${SNAP:?run through snapshot_submit.sh}/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
WT=/mnt/data/peeradon/h2snn-model/docs
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "snapshot=${SNAP} $(cat ${SNAP}/SNAPSHOT.txt)"
echo "git=$(git -C /mnt/data/peeradon/h2snn-model rev-parse HEAD)"
lscpu | grep 'Model name'
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability())'
F='GBWarning|expandable|allocation pattern|PYTORCH_CUDA|gb_warning|An experimental|See https|and set the|FutureWarning|autocast|_flatten\("", torch.load|torch.load|DeprecationWarning|import torch_geometric'
echo "old golden sha256: $(sha256sum ${WT}/legacy_evidence/v3b_golden_cluster.pt | cut -d' ' -f1)"
echo "old mb0 sha256:    $(sha256sum ${WT}/partMB_evidence/mb0_reference_cluster.pt | cut -d' ' -f1)"
echo "===== capture golden"
python ../docs/legacy_evidence/v3b_golden.py capture ${WT}/legacy_evidence/v3b_golden_cluster.pt 2>&1 | grep -vE "${F}"; echo "capture_exit=${PIPESTATUS[0]}"
echo "===== save mb0"
python ../docs/partMB_evidence/mb0_reference.py save ${WT}/partMB_evidence/mb0_reference_cluster.pt 2>&1 | grep -vE "${F}"; echo "save_exit=${PIPESTATUS[0]}"
echo "new golden sha256: $(sha256sum ${WT}/legacy_evidence/v3b_golden_cluster.pt | cut -d' ' -f1)"
echo "new mb0 sha256:    $(sha256sum ${WT}/partMB_evidence/mb0_reference_cluster.pt | cut -d' ' -f1)"
echo "===== compare golden (second run, same node)"
python ../docs/legacy_evidence/v3b_golden.py compare ${WT}/legacy_evidence/v3b_golden_cluster.pt 2>&1 | grep -vE "${F}"; echo "golden_compare_exit=${PIPESTATUS[0]}"
echo "===== compare mb0 (second run, same node)"
python ../docs/partMB_evidence/mb0_reference.py compare ${WT}/partMB_evidence/mb0_reference_cluster.pt 2>&1 | grep -vE "${F}"; echo "mb0_compare_exit=${PIPESTATUS[0]}"
echo "end=$(date -Is)"
