#!/bin/bash
# M3a gate after every unit (card "Rule"): golden tests green and
#   python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt
# -> "differing: none", exit 0; with the pinned kernel environment E exported (as run_tests.sbatch does).
# Called from a unit's job script (bash docs/partM3a_evidence/gate.sh), or alone in a job / allocation step.
#SBATCH --job-name=m3a-gate
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:20:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "===== GATE job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) git=$(git rev-parse HEAD) $(date -Is)"
lscpu | grep 'Model name'
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads())'
CUDA_VISIBLE_DEVICES="" python -m unittest -v tests.test_task_heads.NodeClassificationGoldenTests \
    tests.test_cooperative.SymmetricIsThePreviousModelTests 2>&1 | grep -vE "FutureWarning|autocast"
golden=${PIPESTATUS[0]}
echo "gate_golden_exit=${golden}"
CUDA_VISIBLE_DEVICES="" python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt 2>&1 \
    | grep -vE "FutureWarning|autocast|_flatten\(\"\", torch.load"
mb0=${PIPESTATUS[0]}
echo "gate_mb0_exit=${mb0}"
if [ "${golden}" = "0" ] && [ "${mb0}" = "0" ]; then echo "GATE PASS"; else echo "GATE FAIL"; exit 1; fi
