#!/bin/bash
# Download / preprocess H2GB datasets (scripts/data/h2gb_use_dataset.py) in one SLURM job.
#
#   cd /mnt/data/peeradon/h2snn/H2S-NN
#   sbatch scripts/slurm/run_use_dataset.sbatch                 # all H2GB datasets of the helper's list
#   sbatch scripts/slurm/run_use_dataset.sbatch oag-cs Pokec    # only some
#
# Any interactive prompt the download/preprocessing step raises (e.g. an "overwrite
# existing file?" confirmation) is auto-answered "yes" via `yes |`.
#
# SLURM jobs do not read ~/.bashrc: this script initialises the author's Miniforge and
# activates `h2snn-b` itself (the default env since cluster-setup3). Never the lab's /opt/conda
# (its package cache is locked for this user). Log: logs/slurm-<jobid>.out under H2S-NN/ (git-ignored).
# The vendor patch (vendor/patches/H2GB.patch) must be applied to vendor/H2GB first
# (scripts/setup/apply_vendor_patches.sh; clone_vendors.sh calls it).
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: DGL graphbolt sets exactly this value itself when the
# variable is unset and prints a GBWarning about it; setting it to True "acknowledges and disables the
# warning" (dgl/graphbolt/__init__.py), so the allocator behaves as before and the log has no warning
# (Part M3b B1; CS7/CS8 had no handling).
#
# Unlike srun --pty, this runs detached from the login session: it keeps going even if
# your SSH connection drops.
#SBATCH --job-name=h2snn-use-dataset
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/H2S-NN/logs/slurm-%j.out

set -uo pipefail

cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD 2>/dev/null) branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

yes | python scripts/data/h2gb_use_dataset.py "$@"
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
