#!/bin/bash
# Card 8: targeted unittest job on CPU (no GPU), and optional mutation checks.
#
#   sbatch docs/card8_evidence/card8_tests.sbatch tests.test_card8_published ...      # targets
#   MUTATIONS=docs/card8_evidence/mutations_c1.tsv sbatch docs/card8_evidence/card8_tests.sbatch tests.x
#
# MUTATIONS: a TSV (name, file, old, new, target test); each row: back up the file, replace `old` by `new` (exactly
# once, else the row is an error), run the target test (expected: FAIL), restore the file from the backup.
#SBATCH --job-name=card8-tests
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=3:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn/H2S-NN/logs/slurm-%j.out

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
export CUDA_VISIBLE_DEVICES=""

echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- src scripts tests configs | head -50
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

status=0
if [ "$#" -gt 0 ]; then
    python -m unittest -v "$@"
    status=$?
    echo "targets exit=${status}"
fi

if [ -n "${MUTATIONS:-}" ]; then
    while IFS=$'\t' read -r name file old new target; do
        [ -z "${name}" ] && continue
        case "${name}" in \#*) continue ;; esac
        cp "${file}" "${file}.card8bak"
        if ! python - "${file}" "${old}" "${new}" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2].encode().decode("unicode_escape"), sys.argv[3].encode().decode("unicode_escape")
text = open(path, encoding="utf-8").read()
if text.count(old) != 1:
    sys.exit(f"mutation anchor found {text.count(old)} times in {path}")
open(path, "w", encoding="utf-8").write(text.replace(old, new))
PY
        then
            echo "MUTATION ${name}: ERROR (anchor)"; mv "${file}.card8bak" "${file}"; status=3; continue
        fi
        python -m unittest "${target}" > "logs/card8_mut_${SLURM_JOB_ID:-x}_${name}.log" 2>&1
        rc=$?
        mv "${file}.card8bak" "${file}"
        tail -3 "logs/card8_mut_${SLURM_JOB_ID:-x}_${name}.log"
        if [ "${rc}" -ne 0 ]; then echo "MUTATION ${name}: CAUGHT (test failed, rc=${rc})";
        else echo "MUTATION ${name}: NOT CAUGHT"; status=4; fi
    done < "${MUTATIONS}"
    git diff --stat -- src scripts tests configs
fi
echo "end=$(date -Is) exit=${status}"
exit "${status}"
