#!/bin/bash
# CS7-1a tests + mutations, then CS7-1b: build h2snn-b (conda create + install_env_b.sh), check_env, pip freeze.
#SBATCH --job-name=cs7-1ab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/docs/cluster_evidence2/cs7_1ab_build.txt

set -uo pipefail
REPO=/mnt/data/peeradon/h2snn
EV=${REPO}/docs/cluster_evidence2
DGL_FILE=/mnt/home/peeradon/wheels/dgl-2.4.0.th24.cu124-py312_0.tar.bz2
cd ${REPO}/H2S-NN || exit 2
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "gcc: $(g++ --version | head -n1)"
. ~/miniforge3/etc/profile.d/conda.sh

echo; echo "===== CS7-1b: conda create (only if absent)"
if [ -d ~/.conda/envs/h2snn-b ]; then
    echo "h2snn-b exists: not recreated"
else
    conda create -n h2snn-b python=3.12 -y 2>&1 | grep -v -E '^\s*$' ; echo "conda_create_exit=${PIPESTATUS[0]}"
fi
conda activate h2snn-b || exit 2
echo "active=${CONDA_DEFAULT_ENV} python=$(command -v python) $(python --version)"

echo; echo "===== CS7-1a: unit tests (no install)"
bash -n scripts/setup/install_env_b.sh; echo "bash_n_exit=$?"
python -m unittest -v tests.test_env_b_install; echo "tests_exit=$?"

echo; echo "===== CS7-1a mutations (applied to the real file, restored from a copy)"
cp scripts/setup/install_env_b.sh /tmp/install_env_b.sh.orig
echo "--- mutation 1: the sha256 check removed"
python - <<'EOF'
from pathlib import Path
p = Path("scripts/setup/install_env_b.sh"); t = p.read_text()
start = t.index('SHA_WANT="'); end = t.index('echo "== step 6')
p.write_text(t[:start] + t[end:])
EOF
python -m unittest tests.test_env_b_install.InstallScriptBTests.test_repository_install_script 2>&1 | grep -E "AssertionError|^(OK|FAILED)"
cp /tmp/install_env_b.sh.orig scripts/setup/install_env_b.sh
echo "--- mutation 2: dgl not filtered (grep -v line replaced by a copy)"
python - <<'EOF'
from pathlib import Path
p = Path("scripts/setup/install_env_b.sh"); t = p.read_text()
line = next(l for l in t.splitlines() if l.startswith("grep -v") and "dgl" in l)
p.write_text(t.replace(line, 'cp "${LOCK}" "${REQ}"'))
EOF
python -m unittest tests.test_env_b_install.InstallScriptBTests.test_repository_install_script 2>&1 | grep -E "AssertionError|^(OK|FAILED)"
cp /tmp/install_env_b.sh.orig scripts/setup/install_env_b.sh
cmp scripts/setup/install_env_b.sh /tmp/install_env_b.sh.orig && echo "restored: identical"
python -m unittest tests.test_env_b_install 2>&1 | tail -n 1

echo; echo "===== CS7-1b: install_env_b.sh h2snn-b <DGL file>"
SECONDS=0
bash scripts/setup/install_env_b.sh h2snn-b "${DGL_FILE}"
echo "install_exit=$? seconds=${SECONDS}"

echo; echo "===== check_env.py --lock requirements/linux-lock-b.txt"
python scripts/setup/check_env.py --lock requirements/linux-lock-b.txt; echo "check_env_exit=$?"
echo; echo "===== pip check"
python -m pip check; echo "pip_check_exit=$?"
echo; echo "===== conda list -n h2snn-b dgl"
conda list -n h2snn-b dgl
echo; echo "===== torch / numpy"
python -c "import torch, numpy; print('torch', torch.__version__, 'cuda_build', torch.version.cuda, 'numpy', numpy.__version__)"
python -m pip freeze > ${EV}/h2snn-b_pip_freeze.txt; echo "pip_freeze_lines=$(wc -l < ${EV}/h2snn-b_pip_freeze.txt)"
echo "end=$(date -Is)"
