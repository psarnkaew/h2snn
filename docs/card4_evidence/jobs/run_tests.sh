#!/bin/bash
# Card 4: unittest targets (or the whole suite) and the bit-exact compares in the ablation-prep worktree.
#   srun --jobid=<id> --overlap bash docs/card4_evidence/jobs/run_tests.sh [--compare] [unittest targets...]
#   sbatch ... docs/card4_evidence/jobs/run_tests.sh --suite
# The repo's scripts/slurm/run_tests.sbatch cd's into /mnt/data/peeradon/h2snn (another agent's
# worktree), so this card uses its own copy. Pinned kernel environment E as in run_tests.sbatch.
set -uo pipefail
WT=/mnt/data/peeradon/h2snn-ablprep
cd "$WT/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2
export MKL_CBWR=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2

echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "dirty: $(git status --short -- src tests scripts | tr '\n' ' ')"
sha256sum src/models/map_operators.py src/models/sheaf_maps.py src/models/h2snn_model.py \
    src/configs/config.py tests/test_card4_*.py 2>/dev/null
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cpu_capability', torch.backends.cpu.get_cpu_capability(), 'threads', torch.get_num_threads())"

status=0
compare=0
suite=0
targets=()
for arg in "$@"; do
    case "$arg" in
        --compare) compare=1 ;;
        --suite) suite=1 ;;
        *) targets+=("$arg") ;;
    esac
done
if [ "$compare" -eq 1 ]; then
    echo "=== v3b_golden compare"
    python ../docs/legacy_evidence/v3b_golden.py compare ../docs/legacy_evidence/v3b_golden_cluster.pt || status=1
    echo "=== mb0_reference compare"
    python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt || status=1
fi
if [ "$suite" -eq 1 ]; then
    python -m unittest discover -s tests -v || status=1
elif [ "${#targets[@]}" -gt 0 ]; then
    python -m unittest -v "${targets[@]}" || status=1
fi
echo "end=$(date -Is) exit=${status}"
exit "${status}"
