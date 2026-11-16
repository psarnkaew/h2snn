#!/bin/bash
# Card 1c F2 (G-2): install sparse_tools from source as SeHGNN/Readme.md:22-27 says (git clone --depth=1; python setup.py
# develop) into env h2snn-b; record commit, build log, pip freeze diff, pip check, torch version.
#   srun --jobid=16741 --overlap bash docs/baselines_h2gb_evidence/p3_f2_build_sparse_tools.sh
set -uo pipefail
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "repo git=$(git -C /mnt/data/peeradon/h2snn rev-parse HEAD)"
SRC=/mnt/home/peeradon/src/sparse_tools
echo "sparse_tools commit=$(git -C $SRC rev-parse HEAD)"
python -m pip freeze > /tmp/freeze_before.$$ 2>/dev/null
python -m pip check > /tmp/check_before.$$ 2>&1; echo "pip check before:"; cat /tmp/check_before.$$
cd $SRC || exit 2
python setup.py develop > /tmp/develop.$$ 2>&1; echo "setup.py develop exit=$? (first errors:)"
grep -m5 -iE "error|no module" /tmp/develop.$$
# setuptools >= 70 runs `develop` as `pip install -e . --use-pep517` (an isolated build env without torch); the same
# editable install without build isolation and without any dependency change:
python -m pip install -e . --no-build-isolation --no-deps 2>&1 | tail -15; build=${PIPESTATUS[0]}
echo "build_exit=${build}"
python -m pip freeze > /tmp/freeze_after.$$ 2>/dev/null
echo "pip freeze diff:"; diff /tmp/freeze_before.$$ /tmp/freeze_after.$$
python -m pip check > /tmp/check_after.$$ 2>&1; echo "pip check after:"; cat /tmp/check_after.$$
echo "pip check diff:"; diff /tmp/check_before.$$ /tmp/check_after.$$ && echo "(unchanged)"
cd /tmp && python -c "import torch, sparse_tools; print('torch', torch.__version__, 'sparse_tools', sparse_tools.__file__, [n for n in dir(sparse_tools) if n.startswith('spspmm')])"
echo "end=$(date -Is)"
