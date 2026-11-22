#!/bin/bash
# Card 17: targeted unittest job on CPU, the card's one mutation, and the five published cell-list lines.
#   MUTATIONS=../docs/card17_evidence/mutations.tsv sbatch docs/card17_evidence/card17_tests.sbatch tests.x ...
# MUTATIONS: a TSV (name, file, old, new, target test); each row: back up the file, replace `old` by `new` (exactly
# once), run the target test (expected: FAIL), restore the file from the backup.
#SBATCH --job-name=c17-tests
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-card17/docs/card17_evidence/slurm-%j-tests.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card17/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b WANDB_MODE=disabled
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
export CUDA_VISIBLE_DEVICES=""
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- src scripts tests configs | head -50
status=0
if [ "$#" -gt 0 ]; then
    python -m unittest -v "$@"
    status=$?
    echo "targets exit=${status}"
fi
if [ -n "${MUTATIONS:-}" ]; then
    while IFS=$'\t' read -r name file old new target; do
        [ -z "${name}" ] && continue
        cp "${file}" "${file}.card17bak"
        if ! python - "${file}" "${old}" "${new}" <<'PY'
import sys
path, old, new = sys.argv[1:4]
text = open(path, encoding="utf-8").read()
if text.count(old) != 1:
    sys.exit(f"mutation anchor found {text.count(old)} times in {path}")
open(path, "w", encoding="utf-8").write(text.replace(old, new))
PY
        then
            echo "MUTATION ${name}: ERROR (anchor)"; mv "${file}.card17bak" "${file}"; status=3; continue
        fi
        python -m unittest "${target}" > "../docs/card17_evidence/mut_${SLURM_JOB_ID:-x}_${name}.log" 2>&1
        rc=$?
        mv "${file}.card17bak" "${file}"
        tail -3 "../docs/card17_evidence/mut_${SLURM_JOB_ID:-x}_${name}.log"
        if [ "${rc}" -ne 0 ]; then echo "MUTATION ${name}: CAUGHT (test failed, rc=${rc})";
        else echo "MUTATION ${name}: NOT CAUGHT"; status=4; fi
    done < "${MUTATIONS}"
    git diff --stat -- src
fi
echo "== make_cell_list --phase published (ieee-cis-g lines)"
python scripts/tsubame/make_cell_list.py --phase published --out out/cells_published.txt > /dev/null 2>&1 \
    && grep -c . out/cells_published.txt && grep ieee-cis-g out/cells_published.txt; rm -rf out
echo "end=$(date -Is) exit=${status}"
exit "${status}"
