#!/bin/bash
# Install the option-B environment (torch 2.4.1+cu124, DGL 2.4.0+cu124 conda package) into an
# EXISTING conda environment (cluster-setup2 card, CS7-1a).
#
#   conda create -n h2snn-b python=3.12 -y                     # first, in the same job
#   bash scripts/setup/install_env_b.sh h2snn-b /mnt/home/peeradon/wheels/dgl-2.4.0.th24.cu124-py312_0.tar.bz2
#
# Run it in a SLURM job (never on the login node). Steps:
#   1. torch from the cu124 index (pip --no-cache-dir)
#   2. the PyG extensions from the torch-2.4.0+cu124 PyG wheel index (versions read from the lock)
#   3. every other line of requirements/linux-lock-b.txt except the dgl line (filtered from the file)
#   4. torch.__version__ must be 2.4.1+cu124 (checked after every install step; exit 1 otherwise)
#   5. sha256 of DGL_FILE must equal the value in the lock's DGL comment (exit 1 otherwise)
#   6. conda install -n ENV_NAME --no-deps --offline DGL_FILE
#   7. pip check (output kept, exit code not used: DGL declares torch<=2.4.0)
# Idempotent: every step installs exact pins or a fixed file. It refuses to run without
# ENV_NAME and refuses the existing torch-2.6 environment, which it never modifies.

set -euo pipefail

ENV_NAME="${1:-}"
DGL_FILE="${2:-}"
if [ -z "${ENV_NAME}" ]; then
    echo "usage: install_env_b.sh ENV_NAME DGL_FILE (ENV_NAME is required)" >&2
    exit 2
fi
if [ "${ENV_NAME}" = "base" ] || [ "${ENV_NAME}" = "h2snn" ]; then
    echo "refused: ENV_NAME=${ENV_NAME} (this script installs only into a new option-B environment)" >&2
    exit 2
fi
if [ -z "${DGL_FILE}" ] || [ ! -f "${DGL_FILE}" ]; then
    echo "usage: install_env_b.sh ENV_NAME DGL_FILE (DGL_FILE must be an existing file: '${DGL_FILE}')" >&2
    exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="${REPO_DIR}/requirements/linux-lock-b.txt"
TORCH_WANT="2.4.1+cu124"
PYG_INDEX="https://data.pyg.org/whl/torch-2.4.0+cu124.html"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

. ~/miniforge3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"
if [ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]; then
    echo "refused: could not activate ${ENV_NAME} (active: ${CONDA_DEFAULT_ENV:-none})" >&2
    exit 2
fi
echo "env=${ENV_NAME} python=$(command -v python) lock=${LOCK}"

check_torch() {
    local have
    have="$(python -c 'import torch; print(torch.__version__)')"
    echo "[$1] torch ${have}"
    if [ "${have}" != "${TORCH_WANT}" ]; then
        echo "STOP: torch is ${have}, expected ${TORCH_WANT} after step $1" >&2
        exit 1
    fi
}

pin_of() {  # exact version of one package in the lock
    grep -E "^$1==" "${LOCK}" | head -n 1 | cut -d= -f3
}

echo "== step 1: torch"
python -m pip install --no-cache-dir --index-url "${TORCH_INDEX}" --extra-index-url https://pypi.org/simple \
    "torch==${TORCH_WANT}"
check_torch 1

echo "== step 2: PyG extensions"
python -m pip install --no-cache-dir -f "${PYG_INDEX}" \
    --only-binary torch_scatter,torch_sparse,torch_cluster \
    "torch_scatter==$(pin_of torch_scatter)" "torch_sparse==$(pin_of torch_sparse)" "torch_cluster==$(pin_of torch_cluster)"
check_torch 2

echo "== step 3: every other lock line except dgl"
REQ="$(mktemp)"
trap 'rm -f "${REQ}"' EXIT
grep -v -E '^[[:space:]]*dgl[[:space:]]*==' "${LOCK}" > "${REQ}"
if grep -q -E '^[[:space:]]*dgl' "${REQ}"; then
    echo "STOP: the filtered requirements still contain a dgl line" >&2
    exit 1
fi
python -m pip install --no-cache-dir -r "${REQ}"
check_torch 3

echo "== step 5: DGL file sha256"
SHA_WANT="$(grep -o -E 'sha256 [0-9a-f]{64}' "${LOCK}" | head -n 1 | cut -d' ' -f2)"
SHA_HAVE="$(sha256sum "${DGL_FILE}" | cut -d' ' -f1)"
echo "sha256 lock=${SHA_WANT} file=${SHA_HAVE} size=$(stat -c %s "${DGL_FILE}")"
if [ -z "${SHA_WANT}" ] || [ "${SHA_HAVE}" != "${SHA_WANT}" ]; then
    echo "STOP: DGL file sha256 differs from the lock" >&2
    exit 1
fi

echo "== step 6: conda install DGL (no deps, offline)"
conda install -n "${ENV_NAME}" --no-deps --offline -y "${DGL_FILE}"
check_torch 6

echo "== step 7: pip check (exit code not used)"
python -m pip check || true
echo "install_env_b.sh: done"
