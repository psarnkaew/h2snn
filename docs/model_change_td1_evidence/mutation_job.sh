#!/bin/bash
# Model change TD-1, M3: the two mutation checks of the card, each against its target test only
# (environment E, CPU), run in a snapshot (snapshot_submit.sh); the snapshot's source is restored and
# compared after each mutation.
#SBATCH --job-name=td1-mut
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0:30:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/model_change_td1_evidence/mutation_%j.txt
set -uo pipefail
cd "${SNAP:?run through snapshot_submit.sh}/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "snapshot=${SNAP} $(cat ${SNAP}/SNAPSHOT.txt)"
echo "git=$(git -C /mnt/data/peeradon/h2snn-model rev-parse HEAD)"
S=src/models/sheaf_maps.py
cp $S /tmp/sheaf_maps.orig.$$
mutate() {  # $1 label, $2 old, $3 new, $4 test id
    python - "$2" "$3" <<'PY'
import sys
from pathlib import Path
p = Path("src/models/sheaf_maps.py"); t = p.read_text()
old, new = sys.argv[1], sys.argv[2]
assert t.count(old) == 1, old
p.write_text(t.replace(old, new))
PY
    echo "===== mutation $1 -> target $4"
    python -m unittest -v "$4" 2>&1 | grep -E "^(test_|AssertionError|Ran |OK|FAILED)" | cut -c1-300
    cp /tmp/sheaf_maps.orig.$$ $S && cmp $S /tmp/sheaf_maps.orig.$$ && echo "restored: identical"
}
mutate "1: drop the antisymmetrisation of C_e on rho(r) = r facts" \
    "skew = skew.index_put((rows,), 0.5 * (skew[rows] - swapped[:, :-1]))" "skew = skew" \
    tests.test_td1_maps.OrientationTests.test_T4_flipping_shared_role_facts_leaves_the_operator_unchanged
mutate "2: exp(C) instead of exp(C) - I in the closed form" \
    "return scale, torch.linalg.matrix_exp(generator) - eye" "return scale, torch.linalg.matrix_exp(generator)" \
    tests.test_td1_maps.ClosedFormTests.test_T3_closed_form_equals_the_dense_matrix_exponential
echo "===== unmutated targets"
python -m unittest tests.test_td1_maps.OrientationTests tests.test_td1_maps.ClosedFormTests 2>&1 | grep -E "^(Ran |OK|FAILED)"
echo "end=$(date -Is)"
