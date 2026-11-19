#!/bin/bash
# Card 5: unittest targets (or --suite) and the bit-exact compares (--compare) in this worktree, pinned kernel
# environment E (as scripts/slurm/run_tests.sbatch, which cd's into another worktree).
set -uo pipefail
cd /mnt/data/peeradon/h2snn-ablprep/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "dirty: $(git status --short -- src tests scripts | tr '\n' ' ')"
sha256sum src/trainers/minibatch_trainer.py src/models/h2snn_model.py src/configs/config.py \
    src/analysis/mechanism_hooks.py scripts/run_h2snn_wandb_tuning.py scripts/tsubame/make_cell_list.py \
    tests/test_card5_*.py ../docs/legacy_evidence/v3b_golden_cluster.pt ../docs/partMB_evidence/mb0_reference_cluster.pt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cpu_capability', torch.backends.cpu.get_cpu_capability(), 'threads', torch.get_num_threads())"
status=0; compare=0; suite=0; targets=()
for arg in "$@"; do case "$arg" in --compare) compare=1 ;; --suite) suite=1 ;; *) targets+=("$arg") ;; esac; done
if [ "$compare" -eq 1 ]; then
    echo "=== v3b_golden compare"
    python ../docs/legacy_evidence/v3b_golden.py compare ../docs/legacy_evidence/v3b_golden_cluster.pt || status=1
    echo "=== mb0_reference compare"
    python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt || status=1
fi
if [ "$suite" -eq 1 ]; then python -m unittest discover -s tests -v || status=1
elif [ "${#targets[@]}" -gt 0 ]; then python -m unittest -v "${targets[@]}" || status=1; fi
echo "end=$(date -Is) exit=${status}"; exit "${status}"
