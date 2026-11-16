#!/bin/bash
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN
echo "HEAD $(git rev-parse HEAD) host $(hostname) job $SLURM_JOB_ID"
CUDA_VISIBLE_DEVICES= python ../docs/hetsheaf_evidence/a1_lp_relations.py 2>&1 | grep -v Warning
