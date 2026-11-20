#!/bin/bash
# Card 9 W9: the full suite in one job, in the worktree (vendor complete, vendor/sparse_tools built in place),
# environment h2snn-b, one GPU (as card 7's reference suite 17858: 879 tests). No golden / mb0 compare: src/ does not
# change (card 9 scope rule; checked below). --mem 35G as card 7 (card 5 suite peaked at 28.2 GiB RSS).
# Never the A6000 / g3an01.
#   sbatch docs/card9_evidence/suite_job.sh
#SBATCH --job-name=c9-suite
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=35G
#SBATCH --time=3:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card9_evidence/suite_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 MPLBACKEND=Agg
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "dirty: $(git status --short -- src tests scripts configs | tr '\n' ' ')"
echo "src+configs vs fc0ebaa: [$(git diff --stat fc0ebaa605280f2d987c4848cc3109c7d3b66680 HEAD -- src configs | tail -n 1)]"
ls vendor/sparse_tools/build > /dev/null && echo "vendor/sparse_tools: built"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())"
status=0
echo "=== suite"
python -m unittest discover -s tests -v || status=1
echo "end=$(date -Is) exit=${status}"
exit "${status}"
