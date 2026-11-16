#!/bin/bash
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN
echo "HEAD $(git rev-parse HEAD) host $(hostname) job $SLURM_JOB_ID"
sha256sum requirements/linux-lock-b.txt
python scripts/setup/check_env.py
