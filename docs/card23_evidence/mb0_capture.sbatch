#!/bin/bash
# Card 23 follow-up 4: re-capture docs/partMB_evidence/mb0_reference_cluster.pt at v3b b0f13fb (the runs checkout,
# read only; its own mb0_reference.py `save`), E pinned, 1 thread (the script's rule), then compare b0f13fb to it.
#SBATCH --job-name=c23-mb0cap
#SBATCH --partition=lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/mnt/data/peeradon/h2snn-card23/docs/card23_evidence/job%j_mb0_capture.log
set -uo pipefail
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
export PYTHONDONTWRITEBYTECODE=1
OUT=/mnt/data/peeradon/h2snn-card23/docs/partMB_evidence/mb0_reference_cluster.pt
BASE=/mnt/data/peeradon/h2snn-runs/b0f13fb43bcc77bdf15f73a2caa026140179e441
cd $BASE/H2S-NN || exit 2
echo "job=${SLURM_JOB_ID} node=$(hostname) $(lscpu | grep 'Model name') git=$(git rev-parse HEAD)"
python -c "import torch; print('torch', torch.__version__, 'cpu_capability', torch.backends.cpu.get_cpu_capability())"
python ../docs/partMB_evidence/mb0_reference.py save $OUT 2>/dev/null; echo "save exit=$?"
sha256sum $OUT
python ../docs/partMB_evidence/mb0_reference.py compare $OUT 2>/dev/null; echo "b0f13fb compare exit=$?"
