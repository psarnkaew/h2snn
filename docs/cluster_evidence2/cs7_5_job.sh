#!/bin/bash
# CS7-5: the float64 scope test (atol 1e-12, negative control), CPU only; with MUTATE=1 also the
# mutation (scope encoder reads global_node_id shifted by one), applied and restored in this job.
#   bash cs7_5_job.sh [mutate]    (as an srun step of an allocation, or sbatch with the argument)
#SBATCH --job-name=cs7-5
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00

set -uo pipefail
MUTATE="${1:-}"
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES=""
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
python -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads()); print([l for l in torch.__config__.show().splitlines() if 'CPU capability' in l])"
T=tests.test_partMB_scopes

echo "===== ${T} (whole module)"
python -m unittest -v ${T} 2>&1; echo "test_exit=$?"

if [ "${MUTATE}" = "mutate" ]; then
    F=src/models/node_encoder.py
    echo; echo "===== mutation: ${F} reads global_node_id shifted by one"
    cp ${F} "${TMPDIR:-/tmp}/node_encoder.py.orig.$$"
    sed -i 's/ids = rows if global_node_id is None else global_node_id\[rows\]$/ids = rows if global_node_id is None else global_node_id[rows] + 1/' ${F}
    grep -n "global_node_id\[rows\] + 1" ${F}
    python -m unittest ${T}.ScopeTests.test_featureless_node_in_two_scopes_has_its_global_h0 2>&1 | grep -E "Error|^(OK|FAILED)|CS7-5"
    cp "${TMPDIR:-/tmp}/node_encoder.py.orig.$$" ${F}
    cmp ${F} "${TMPDIR:-/tmp}/node_encoder.py.orig.$$" && echo "restored: identical"
    git diff --quiet -- ${F} && echo "git diff ${F}: clean"
fi
echo "end=$(date -Is)"
