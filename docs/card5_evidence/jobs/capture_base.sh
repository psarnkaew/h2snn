#!/bin/bash
# Card 5 T4/T9: capture the reference at the base commit (scratch detached worktree h2snn-card5-base, 894fd74).
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card5-base/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"
python /mnt/data/peeradon/h2snn-ablprep/docs/card5_evidence/capture_base.py /mnt/data/peeradon/h2snn-ablprep/docs/card5_evidence/base_capture_894fd74.json
status=$?; echo "end=$(date -Is) exit=${status}"; exit $status
