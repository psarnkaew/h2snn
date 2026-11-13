#!/bin/bash
# CS2: comparator unit tests, then check_env.py against the installed h2snn environment.
set -u
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn || exit 2
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m unittest -v tests.test_check_env 2>&1 | tail -4
echo "=== python scripts/setup/check_env.py"
python scripts/setup/check_env.py; echo "check_env_exit=$?"
