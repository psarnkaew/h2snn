#!/bin/bash
# TSUBAME card T1: build the h2snn-b environment in a cpu_8 job (guide section 4); never on the login node.
#
#   cd $WORK/h2snn-runs/<sha>/H2S-NN
#   qsub -g $(cat ~/.t4_group) -l cpu_8=1 -l h_rt=2:00:00 scripts/tsubame/install_env.sh [DGL_FILE]
#
# Creates the conda env h2snn-b (python 3.12) if it does not exist, then runs the existing
# scripts/setup/install_env_b.sh h2snn-b <DGL file> and scripts/setup/build_sparse_tools.sh unchanged, then prints
# pip freeze. DGL_FILE defaults to ~/wheels/dgl-2.4.0.th24.cu124-py312_0.tar.bz2 (copied there per guide section 3).
#$ -cwd
#$ -j y
set -uo pipefail
DGL_FILE="${1:-${HOME}/wheels/dgl-2.4.0.th24.cu124-py312_0.tar.bz2}"
echo "head=$(git rev-parse HEAD) host=$(hostname) job=${JOB_ID:-none} start=$(date -Is)"
if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module > /dev/null 2>&1; then module purge; echo "module purge: done"; fi
# shellcheck disable=SC1090
. ~/miniforge3/etc/profile.d/conda.sh || { echo "FAIL: no ~/miniforge3 (guide section 4)"; exit 2; }
[ -f "${DGL_FILE}" ] || { echo "FAIL: no DGL package at ${DGL_FILE}"; exit 2; }
if ! conda env list | awk '{print $1}' | grep -qx h2snn-b; then
    conda create -n h2snn-b python=3.12 -y || { echo "FAIL: conda create"; exit 1; }
fi
bash scripts/setup/install_env_b.sh h2snn-b "${DGL_FILE}" || { echo "FAIL: install_env_b.sh"; exit 1; }
bash scripts/setup/build_sparse_tools.sh || { echo "FAIL: build_sparse_tools.sh"; exit 1; }
conda activate h2snn-b
echo "--- pip freeze"
python -m pip freeze
echo "end=$(date -Is) exit=0"
