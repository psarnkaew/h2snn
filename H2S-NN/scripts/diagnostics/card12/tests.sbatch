#!/bin/bash
# Card 12 unit P: the full unittest suite on the card12 worktree (as scripts/slurm/run_tests.sbatch,
# pinned kernel environment E), then the three mutations of the card (p_mutations.py).
#   sbatch --gres=gpu:3090:1 scripts/diagnostics/card12/tests.sbatch [unittest targets...]
#SBATCH --job-name=c12-tests
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/data/peeradon/card12_runs/logs/slurm-%j.out

set -uo pipefail
cd /mnt/data/peeradon/h2snn-card12/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
md5sum src/models/anchored_solve.py tests/test_anchored_solve.py | sed "s/^/md5 /"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
lscpu | grep 'Model name'
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cpu_capability', torch.backends.cpu.get_cpu_capability(), 'threads', torch.get_num_threads())"
if [ "$#" -eq 0 ]; then
    python -m unittest discover -s tests -v
else
    python -m unittest -v "$@"
fi
suite=$?
echo "suite_exit=${suite} at $(date -Is)"
python -u scripts/diagnostics/card12/p_mutations.py
mut=$?
echo "end=$(date -Is) suite_exit=${suite} mutations_exit=${mut}"
[ "${suite}" -eq 0 ] && [ "${mut}" -eq 0 ]
