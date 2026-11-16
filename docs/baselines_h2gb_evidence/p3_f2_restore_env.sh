#!/bin/bash
# Card 1c F2: restore h2snn-b after the failed label-part install (sortednp 0.5.0 needs numpy<2): uninstall sortednp and
# sparse_tools; the freeze and pip check must equal the state before F2.
set -uo pipefail
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
python -m pip uninstall -y sortednp sparse_tools 2>&1 | tail -4
python -m pip freeze | grep -iE "sortednp|sparse.tools" || echo "neither package is installed"
python -m pip check
python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"
echo "end=$(date -Is)"
