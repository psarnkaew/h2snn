#!/bin/bash
# Run the H2S-NN unittest suite (or the given unittest targets) in one SLURM GPU job.
#
#   cd /mnt/data/peeradon/h2snn/H2S-NN
#   sbatch scripts/slurm/run_tests.sbatch                              # whole suite
#   sbatch scripts/slurm/run_tests.sbatch tests.test_v2_model          # unittest targets
#   sbatch --time=0:30:00 scripts/slurm/run_tests.sbatch tests.test_x  # override a limit
#
# SLURM jobs do not read ~/.bashrc: this script initialises the author's Miniforge and
# activates `h2snn-b` itself (torch 2.4.1+cu124, DGL 2.4.0 conda package; requirements/linux-lock-b.txt,
# scripts/setup/install_env_b.sh; default since cluster-setup3 CS8-3). Never the lab's /opt/conda (its
# package cache is locked for this user). Log: logs/slurm-<jobid>.out under H2S-NN/ (git-ignored).
#
# Pinned CPU kernel environment E (cluster-setup3 card): every node uses the AVX2 kernel path of torch,
# MKL and oneDNN, so the bit-exact tests (cluster golden and mb0 references, captured with E and 1 thread)
# give identical results on the AVX2 (i9-10900K) and AVX512 (i9-10900X) nodes; without E those tests
# fail with "bit-exact test needs the pinned kernel environment".
#SBATCH --job-name=h2snn-tests
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/H2S-NN/logs/slurm-%j.out

set -uo pipefail

cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2
export MKL_CBWR=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2

echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD 2>/dev/null) branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
lscpu | grep 'Model name'
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cpu_capability', torch.backends.cpu.get_cpu_capability(), 'threads', torch.get_num_threads())"

if [ "$#" -eq 0 ]; then
    python -m unittest discover -s tests -v
else
    python -m unittest -v "$@"
fi
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
