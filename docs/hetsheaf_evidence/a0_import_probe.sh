#!/bin/bash
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
cd /mnt/data/peeradon/h2snn-hetsheaf
echo "HEAD $(git rev-parse HEAD) host $(hostname) job $SLURM_JOB_ID"
sha256sum docs/hetsheaf_evidence/a0_import_probe.py
CUDA_VISIBLE_DEVICES= python docs/hetsheaf_evidence/a0_import_probe.py
