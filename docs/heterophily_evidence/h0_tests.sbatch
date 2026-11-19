#!/bin/bash
# Card 6 H0: harness tests, the two bit-exact gates (v3-B golden, mb0 compare) and the mutation checks.
# CPU only (no GPU requested). Pinned kernel environment E as scripts/slurm/run_tests.sbatch (bit-exact gates).
#   sbatch docs/heterophily_evidence/h0_tests.sbatch           (from the worktree root)
#SBATCH --job-name=s5-het-h0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-heterophily/docs/heterophily_evidence/h0_tests_%j.txt

set -uo pipefail
cd /mnt/data/peeradon/h2snn-heterophily/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
export TQDM_DISABLE=1

echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- src scripts tests | sed 's/^/status: /'
python -c "import sys, torch, torch_geometric, torch_sparse, tqdm; print('python', sys.version.split()[0], 'torch', torch.__version__, 'pyg', torch_geometric.__version__, 'torch_sparse', torch_sparse.__version__, 'tqdm', tqdm.__version__, 'cpu', torch.backends.cpu.get_cpu_capability())"

echo "=== 1. new tests"
python -m unittest -v tests.test_heterophily_harness 2>&1
t1=$?
echo "=== 2. bit-exact gates"
python -m unittest -v tests.test_v3b_golden 2>&1
t2=$?
python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt 2>&1 | tail -5
t3=${PIPESTATUS[0]}

echo "=== 3. mutations (each on a copy of the module, restored after; target test only)"
MOD=src/analysis/heterophily.py
cp "$MOD" /tmp/het_backup_$$.py
mutate() {  # name, sed expression, target test
    cp /tmp/het_backup_$$.py "$MOD"
    sed -i "$2" "$MOD"
    if cmp -s /tmp/het_backup_$$.py "$MOD"; then echo "MUTATION $1: sed did not change the file"; return; fi
    python -m unittest "$3" > /tmp/het_mut_$$.txt 2>&1
    if [ $? -eq 0 ]; then echo "MUTATION $1: SURVIVED ($3 passed)"; else echo "MUTATION $1: KILLED ($3 failed)"; grep -E "^(FAIL|ERROR|AssertionError)" /tmp/het_mut_$$.txt | head -3; fi
}
mutate "M1 card: chunked MDE without the degree clamp" 's/\.clamp(min=1)\.sqrt()/.sqrt()/g' \
    tests.test_heterophily_harness.ChunkedEqualsReferenceTests.test_chunked_mde_equals_the_reference_function
mutate "M1b replacement: destination endpoints scaled by the source degree" 's/x\[c\] \/ dst_scale\[c\]/x[c] \/ src_scale[c]/' \
    tests.test_heterophily_harness.ChunkedEqualsReferenceTests.test_chunked_mde_equals_the_reference_function
mutate "M2 card: M2 from get_metapaths for acm instead of the list" 's/if list_key is not None and list_key in HETERO2NET_LISTS:/if False:/' \
    tests.test_heterophily_harness.MetapathSetTests.test_acm_uses_the_predefined_list_even_when_the_schema_offers_more
cp /tmp/het_backup_$$.py "$MOD"
cmp "$MOD" /tmp/het_backup_$$.py && echo "module restored"
rm -f /tmp/het_backup_$$.py /tmp/het_mut_$$.txt

echo "end=$(date -Is) tests=$t1 golden=$t2 mb0=$t3"
