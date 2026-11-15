#!/bin/bash
# M3a-A3: mutations that pass the orientation check but break functoriality, against
# tests/test_partMBfix_functoriality.py only; each applied to src/models/map_operators.py and restored
# inside this job (no src/ change is kept). Then the gate. With E exported.
#   (i)  the transport ignores the hop sign (every hop treated as forward)
#   (ii) the near/far maps of one reversed hop swapped in the transport (the first reversed hop met)
#   srun --jobid=<allocation> --overlap bash docs/partM3a_evidence/a3_job.sh   (or sbatch)
#SBATCH --job-name=m3a-a3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
export CUDA_VISIBLE_DEVICES=""
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
T=tests.test_partMBfix_functoriality
F=src/models/map_operators.py
TMP=${TMPDIR:-/tmp}/m3a_a3_$$; mkdir -p ${TMP}; cp ${F} ${TMP}/map_operators.orig

echo "===== unmutated: ${T}"
python -m unittest -v ${T} 2>&1 | grep -vE "FutureWarning|autocast"

echo; echo "===== mutation (i): the transport ignores the hop sign (every hop forward)"
python - <<'EOF'
from pathlib import Path
p = Path("src/models/map_operators.py"); t = p.read_text()
old = "            reversed_hop = None if face_edge_sign is None else face_edge_sign[:, j] < 0\n"
assert t.count(old) == 1
p.write_text(t.replace(old, "            reversed_hop = None  # M3a-A3 MUTATION (i): hop sign ignored\n"))
EOF
git diff --stat -- ${F}
python -m unittest -v ${T} 2>&1 | grep -vE "FutureWarning|autocast" | grep -E "\.\.\. |Error|residual|Traceback|File \"/mnt|^Ran|^(OK|FAILED)"
cp ${TMP}/map_operators.orig ${F}; cmp ${F} ${TMP}/map_operators.orig && echo "restored: identical"

echo; echo "===== mutation (ii): near/far maps of one reversed hop swapped (the first reversed hop met)"
python - <<'EOF'
from pathlib import Path
p = Path("src/models/map_operators.py"); t = p.read_text()
old = "            reversed_hop = None if face_edge_sign is None else face_edge_sign[:, j] < 0\n"
new = old + (
    "            # M3a-A3 MUTATION (ii): the first reversed hop met keeps the forward near/far maps\n"
    "            if reversed_hop is not None and bool(reversed_hop.any()) and not _A3_DONE:\n"
    "                reversed_hop = reversed_hop.clone()\n"
    "                reversed_hop[int(reversed_hop.nonzero()[0])] = False\n"
    "                _A3_DONE.append(True)\n")
assert t.count(old) == 1
t = t.replace(old, new)
anchor = "class PathTransportOperator:\n"
assert t.count(anchor) == 1
p.write_text(t.replace(anchor, "_A3_DONE: list = []  # M3a-A3 MUTATION (ii)\n\n\n" + anchor))
EOF
git diff --stat -- ${F}
python -m unittest -v ${T} 2>&1 | grep -vE "FutureWarning|autocast" | grep -E "\.\.\. |Error|residual|Traceback|File \"/mnt|^Ran|^(OK|FAILED)"
cp ${TMP}/map_operators.orig ${F}; cmp ${F} ${TMP}/map_operators.orig && echo "restored: identical"
git diff --quiet -- ${F} && echo "git diff ${F}: clean"
python -m unittest ${T} 2>&1 | tail -n 1

echo
bash ../docs/partM3a_evidence/gate.sh
echo "end=$(date -Is)"
