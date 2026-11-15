#!/bin/bash
# CS7-1a rerun after the test fix (the first matched an echo line containing "conda install").
# Run as a step of the author's allocation: srun --jobid=16721 --overlap bash docs/cluster_evidence2/cs7_1a_tests_job.sh
set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
echo; echo "===== CS7-1a: unit tests (no install)"
bash -n scripts/setup/install_env_b.sh; echo "bash_n_exit=$?"
python -m unittest -v tests.test_env_b_install; echo "tests_exit=$?"

echo; echo "===== CS7-1a mutations (applied to the real file, restored from a copy)"
cp scripts/setup/install_env_b.sh ${TMPDIR:-/tmp}/install_env_b.sh.orig.$$
echo "--- mutation 1: the sha256 check removed"
python - <<'EOF'
from pathlib import Path
p = Path("scripts/setup/install_env_b.sh"); t = p.read_text()
start = t.index('SHA_WANT="'); end = t.index('echo "== step 6')
p.write_text(t[:start] + t[end:])
EOF
python -m unittest tests.test_env_b_install.InstallScriptBTests.test_repository_install_script 2>&1 | grep -E "AssertionError|^(OK|FAILED)"
cp ${TMPDIR:-/tmp}/install_env_b.sh.orig.$$ scripts/setup/install_env_b.sh
echo "--- mutation 2: dgl not filtered (grep -v line replaced by a copy)"
python - <<'EOF'
from pathlib import Path
p = Path("scripts/setup/install_env_b.sh"); t = p.read_text()
line = next(l for l in t.splitlines() if l.startswith("grep -v") and "dgl" in l)
p.write_text(t.replace(line, 'cp "${LOCK}" "${REQ}"'))
EOF
python -m unittest tests.test_env_b_install.InstallScriptBTests.test_repository_install_script 2>&1 | grep -E "AssertionError|^(OK|FAILED)"
cp ${TMPDIR:-/tmp}/install_env_b.sh.orig.$$ scripts/setup/install_env_b.sh
cmp scripts/setup/install_env_b.sh ${TMPDIR:-/tmp}/install_env_b.sh.orig.$$ && echo "restored: identical"
python -m unittest tests.test_env_b_install 2>&1 | tail -n 1

echo "end=$(date -Is)"
