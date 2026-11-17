#!/bin/bash
# Card 3 (update f37e247): regenerate notebooks/baselines/ with the generator (generator not edited), then --check.
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN
echo "HEAD $(git rev-parse HEAD) host $(hostname) job $SLURM_JOB_ID step ${SLURM_STEP_ID:-none}"
sha256sum src/baselines/registry.py scripts/generate_baseline_notebooks.py
CUDA_VISIBLE_DEVICES= python scripts/generate_baseline_notebooks.py 2>&1 | tail -5
CUDA_VISIBLE_DEVICES= python scripts/generate_baseline_notebooks.py --check 2>&1 | tail -3; echo "check_exit=${PIPESTATUS[0]}"
