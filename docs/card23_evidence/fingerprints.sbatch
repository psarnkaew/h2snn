#!/bin/bash
# Card 23 fingerprint check on b0f13fb (the runs checkout, read only) and on card23; run_tests.sbatch env.
#SBATCH --job-name=c23-fp
#SBATCH --partition=lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/mnt/data/peeradon/h2snn-card23/docs/card23_evidence/job%j_fingerprints.log
set -uo pipefail
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
export PYTHONDONTWRITEBYTECODE=1
unset WANDB_MODE
EV=/mnt/data/peeradon/h2snn-card23/docs/card23_evidence
BASE=/mnt/data/peeradon/h2snn-runs/b0f13fb43bcc77bdf15f73a2caa026140179e441
echo "job=${SLURM_JOB_ID} node=$(hostname) base_git=$(git -C $BASE rev-parse HEAD) card23_git=$(git -C /mnt/data/peeradon/h2snn-card23 rev-parse HEAD) card23_dirty=$(git -C /mnt/data/peeradon/h2snn-card23 status --porcelain --untracked-files=no | wc -l)"
python $EV/fingerprints.py $BASE > $EV/fp_b0f13fb.jsonl; echo "base exit=$?"
python $EV/fingerprints.py /mnt/data/peeradon/h2snn-card23 --pack > $EV/fp_card23.jsonl; echo "card23 exit=$?"
head -n -1 $EV/fp_card23.jsonl | diff $EV/fp_b0f13fb.jsonl - && echo "FINGERPRINTS IDENTICAL" || echo "FINGERPRINTS DIFFER"
tail -1 $EV/fp_card23.jsonl
