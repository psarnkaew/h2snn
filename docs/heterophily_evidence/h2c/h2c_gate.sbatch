#!/bin/bash
# Card 6 H2c: the whole harness test file, then the H2c mutations (each on a copy of src/ and tests/ in node-local
# /tmp, vendor and data linked: no file a running job reads is changed). CPU only (no GPU requested).
#   sbatch docs/heterophily_evidence/h2c/h2c_gate.sbatch <worktree root> <mutation set, e.g. 1 or 12>
#SBATCH --job-name=s5-het-h2c-gate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-heterophily-h2c/docs/heterophily_evidence/h2c/logs/h2c_gate_%j.txt

set -uo pipefail
ROOT="$1"; SET="${2:-1}"
WT="$ROOT/H2S-NN"
cd "$WT" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export TQDM_DISABLE=1

echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is) root=$ROOT set=$SET"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- src scripts tests | sed 's/^/status: /'

echo "=== 1. harness tests"
python -m unittest -v tests.test_heterophily_harness 2>&1
t1=$?

echo "=== 2. H2c mutations (copy in /tmp)"
T=$(mktemp -d /tmp/s5_h2c_XXXX)
cp -r src tests "$T"/ && ln -s "$WT/vendor" "$T/vendor" && ln -s "$WT/data" "$T/data"
mkdir -p "$T/scripts" && cp scripts/heterophily_stats.py "$T/scripts/"
mutate() {  # name, file (relative to H2S-NN), sed expression, test
    cp "$WT/$2" "$T/$2"
    sed -i "$3" "$T/$2"
    if cmp -s "$WT/$2" "$T/$2"; then echo "MUTATION $1: sed did not change the file"; return; fi
    (cd "$T" && python -m unittest "$4" > "$T/out.txt" 2>&1)
    if [ $? -eq 0 ]; then echo "MUTATION $1: SURVIVED (gate passed)"; else echo "MUTATION $1: KILLED (gate failed)"
        grep -E "^(FAIL|ERROR|AssertionError)" "$T/out.txt" | head -3; fi
    cp "$WT/$2" "$T/$2"
}
case "$SET" in *1*)
mutate "M4: clique closed form without the self-pair term (degree n-1, n(n-1) pairs)" src/analysis/heterophily.py \
    's/d = size if self_pairs else size - 1  #/d = size - 1  #/' \
    tests.test_heterophily_harness.H2cCliqueTests.test_clique_closed_form_equals_the_chunked_and_reference_mde ;;
esac
rm -rf "$T"
echo "end=$(date -Is) tests=$t1"
