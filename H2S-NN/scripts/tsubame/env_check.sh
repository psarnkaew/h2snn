#!/bin/bash
# TSUBAME card T1: GPU-stack check on one H100, as a trial run (no -g; <= 3 min; guide section 4).
#
#   cd $WORK/h2snn-runs/<sha>/H2S-NN
#   qsub -l gpu_1=1 -l h_rt=0:03:00 scripts/tsubame/env_check.sh
#
# Every check prints OK or FAIL; the script exits 1 on any FAIL. HTTPS checks print status codes only (no credentials).
#$ -cwd
#$ -j y
set -uo pipefail
FAILS=0
check() {  # check <name> <command...>
    local name="$1"; shift
    local out; out="$("$@" 2>&1)"; local code=$?
    if [ ${code} -eq 0 ]; then echo "OK   ${name}: $(echo "${out}" | tail -1)"; else
        echo "FAIL ${name} (exit ${code}):"; echo "${out}" | tail -20 | sed 's/^/     /'; FAILS=$((FAILS + 1)); fi
}
echo "head=$(git rev-parse HEAD) host=$(hostname) job=${JOB_ID:-none} start=$(date -Is)"
if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module > /dev/null 2>&1; then module purge; fi
# shellcheck disable=SC1090
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || { echo "FAIL conda activate h2snn-b"; exit 1; }

check "nvidia-smi" nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
check "torch CUDA" python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
x = torch.randn(64, 64, device='cuda'); y = (x @ x).sum().item()
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(),
      torch.cuda.get_device_name(0), 'sm_%d%d' % torch.cuda.get_device_capability(0), 'arch_list', torch.cuda.get_arch_list())"
check "DGL CUDA" python -c "
import torch, dgl
g = dgl.graph(([0, 1, 2], [1, 2, 0])).to('cuda')
h = dgl.ops.copy_u_sum(g, torch.ones(3, 4, device='cuda'))
assert h.is_cuda and float(h.sum()) == 12.0
print('dgl', dgl.__version__, 'copy_u_sum on', h.device)"
check "PyG sparse matmul CUDA" python -c "
import torch, torch_geometric, torch_sparse
a = torch_sparse.SparseTensor(row=torch.tensor([0, 1, 2]), col=torch.tensor([1, 2, 0]),
                              sparse_sizes=(3, 3)).to('cuda')
y = a.matmul(torch.ones(3, 4, device='cuda'))
assert y.is_cuda and float(y.sum()) == 12.0
print('torch_geometric', torch_geometric.__version__, 'torch_sparse', torch_sparse.__version__, 'matmul on', y.device)"
check "tests/test_check_env.py" python -m unittest tests.test_check_env
# the CUDA-only test must run (exactly one test, not skipped) and pass
check "CUDA-only test (cuda_float32)" bash -c 'out="$(python -m unittest -k cuda_float32 tests.test_factorized_maps 2>&1)"
    code=$?; echo "${out}" | tail -3
    [ ${code} -eq 0 ] && echo "${out}" | grep -q "^Ran 1 test" && ! echo "${out}" | grep -q "skipped"'
for url in https://api.wandb.ai https://pypi.org https://github.com; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "${url}")"
    if [ "${code}" != "000" ]; then echo "OK   HTTPS ${url}: status ${code}"; else
        echo "FAIL HTTPS ${url}: unreachable"; FAILS=$((FAILS + 1)); fi
done
echo "end=$(date -Is) fails=${FAILS}"
[ ${FAILS} -eq 0 ] || exit 1
