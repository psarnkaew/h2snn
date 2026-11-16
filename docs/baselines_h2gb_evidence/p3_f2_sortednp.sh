#!/bin/bash
# Card 1c F2: sparse_tools imports `sortednp` at import time (sparse_tools/utils.py:5; not in its install_requires):
# install it alone (no dependency changes), record pip freeze diff / pip check, and import sparse_tools.
set -uo pipefail
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is) repo git=$(git -C /mnt/data/peeradon/h2snn rev-parse HEAD)"
python -m pip freeze > /tmp/fb.$$ 2>/dev/null; python -m pip check > /tmp/cb.$$ 2>&1
python -m pip install --no-deps sortednp 2>&1 | tail -8; echo "install_exit=${PIPESTATUS[0]}"
python -m pip freeze > /tmp/fa.$$ 2>/dev/null; python -m pip check > /tmp/ca.$$ 2>&1
echo "pip freeze diff:"; diff /tmp/fb.$$ /tmp/fa.$$
echo "pip check diff:"; diff /tmp/cb.$$ /tmp/ca.$$ && echo "(unchanged)"
cd /tmp && python -c "import numpy, torch, sortednp, sparse_tools; print('numpy', numpy.__version__, 'torch', torch.__version__, 'sortednp', sortednp.__version__ if hasattr(sortednp,'__version__') else '?', 'sparse_tools ok', sparse_tools.spspmm_diag_sym_ABA.__name__)"
echo "end=$(date -Is)"
