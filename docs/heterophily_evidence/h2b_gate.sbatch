#!/bin/bash
# Card 6 H2b: gate H2b-eq (closed form = the reference functions on synthetic graphs with a hub; no hub refused), the
# whole harness test file, and the self-pair mutations. CPU only (no GPU requested).
# Mutations run on a copy of src/ and tests/ in node-local /tmp (vendor linked), so no file a running job reads is
# changed (CLAUDE.md: never mutate a file that a running job reads).
#   sbatch docs/heterophily_evidence/h2b_gate.sbatch           (from the worktree root)
#SBATCH --job-name=s5-het-h2b
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-heterophily/docs/heterophily_evidence/h2b_gate_%j.txt

set -uo pipefail
WT=/mnt/data/peeradon/h2snn-heterophily/H2S-NN
cd "$WT" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export TQDM_DISABLE=1

echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- src scripts tests | sed 's/^/status: /'

echo "=== 1. harness tests (H0 + gate H2b-eq)"
python -m unittest -v tests.test_heterophily_harness 2>&1
t1=$?

echo "=== 2. self-pair mutations (copy in /tmp; target: the gate test)"
T=$(mktemp -d /tmp/s5_h2b_XXXX)
cp -r src tests "$T"/ && ln -s "$WT/vendor" "$T/vendor" && ln -s "$WT/data" "$T/data"
GATE=tests.test_heterophily_harness.H2bClosedFormTests.test_closed_form_equals_the_reference_functions
mutate() {  # name, sed expression
    cp "$WT/src/analysis/heterophily.py" "$T/src/analysis/heterophily.py"
    sed -i "$2" "$T/src/analysis/heterophily.py"
    if cmp -s "$WT/src/analysis/heterophily.py" "$T/src/analysis/heterophily.py"; then
        echo "MUTATION $1: sed did not change the file"; return; fi
    (cd "$T" && python -m unittest "$GATE" > "$T/out.txt" 2>&1)
    if [ $? -eq 0 ]; then echo "MUTATION $1: SURVIVED (gate passed)"; else echo "MUTATION $1: KILLED (gate failed)"
        grep -E "^(FAIL|ERROR|AssertionError)" "$T/out.txt" | head -3; fi
}
mutate "M3a: label closed form without self-pairs (n(n-1) pairs)" \
    's/return (counts \* counts).sum(), n \* n/return (counts * counts - counts).sum(), n * n - n/'
mutate "M3b: MDE closed form without self-pairs (degree n-1, n(n-1) pairs)" \
    's/deg, pairs = n, n \* n  #/deg, pairs = n - 1, n * (n - 1)  #/'
rm -rf "$T"

echo "end=$(date -Is) tests=$t1"
