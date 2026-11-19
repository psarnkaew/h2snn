#!/bin/bash
# Card 7 Phase B: the full suite and the bit-exact compares (golden v3b, mb0) in the worktree itself (vendor complete,
# vendor/sparse_tools built in place), environment E, one GPU not shared. --mem 35G: the card's figure (card 5 suite
# 17835 peaked at 28.2 GiB RSS; + 30 % = 36.7, the card sets 35G). Never the A6000 / g3an01.
#   sbatch docs/card7_evidence/suite_job.sh
#SBATCH --job-name=c7-suite
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=35G
#SBATCH --time=3:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/suite_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 MPLBACKEND=Agg
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "dirty: $(git status --short -- src tests scripts | tr '\n' ' ')"
echo "src vs v3b 365a757: $(git diff --stat 365a75763f45dee4b8853cc85ae06f0084295c47 HEAD -- src | tail -n 1)"
sha256sum ../docs/legacy_evidence/v3b_golden_cluster.pt ../docs/partMB_evidence/mb0_reference_cluster.pt
ls vendor/sparse_tools/build > /dev/null && echo "vendor/sparse_tools: built"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cpu_capability', torch.backends.cpu.get_cpu_capability(), 'threads', torch.get_num_threads())"
status=0
echo "=== v3b_golden compare"
python ../docs/legacy_evidence/v3b_golden.py compare ../docs/legacy_evidence/v3b_golden_cluster.pt || status=1
echo "=== mb0_reference compare"
python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt || status=1
echo "=== suite"
python -m unittest discover -s tests -v || status=1
echo "end=$(date -Is) exit=${status}"
exit "${status}"
