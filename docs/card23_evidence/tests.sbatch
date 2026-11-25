#!/bin/bash
# Card 23 tests (one CPU job): golden + mb0 (bit identity with every option off), tests/test_card23.py and the
# quick existing tests of the touched files. Environment as scripts/slurm/run_tests.sbatch (h2snn-b, pinned
# kernel environment E); WANDB_MODE unset.
#SBATCH --job-name=c23-tests
#SBATCH --partition=lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-card23/docs/card23_evidence/job%j_tests.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card23/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2
export MKL_CBWR=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2
unset WANDB_MODE
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is) WANDB_MODE=${WANDB_MODE:-unset}"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
lscpu | grep 'Model name'
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cpu_capability', torch.backends.cpu.get_cpu_capability(), 'threads', torch.get_num_threads())"
echo "=== mb0_reference.py compare (cluster reference)"
python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt; echo "mb0_compare_exit=$?"
echo "=== unittest"
python -m unittest -v tests.test_card23 tests.test_v3b_golden tests.test_mb0_reference_compare \
    tests.test_data tests.test_trainer tests.test_full_graph_early_stopping tests.test_search_spaces \
    tests.test_sweep_batch_size tests.test_wandb_and_runs "$@"
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
