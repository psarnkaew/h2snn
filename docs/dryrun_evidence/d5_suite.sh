#!/bin/bash
# Card 2 D5: full H2S-NN suite once on one node. Run from /mnt/data/peeradon/h2snn-dryrun/H2S-NN inside a SLURM
# allocation. The full log (with the trainers' console lines of synthetic tests) stays outside the repository; only
# unittest's own result lines are printed.
set -u
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
# pinned CPU kernel environment E, as scripts/slurm/run_tests.sbatch (that script cd's into the main checkout, so its
# exports are repeated here for this worktree); the bit-exact golden / mb0 tests refuse to run without it
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2
export MKL_CBWR=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2
LOG=/mnt/data/peeradon/dryrun_scratch/logs/d5_suite_${SLURM_JOB_ID}_${SLURM_STEP_ID:-0}.log
echo "head=$(git rev-parse HEAD) host=$(hostname) job=${SLURM_JOB_ID} step=${SLURM_STEP_ID:-}"
git status --short | grep -v '^?? data$' | sed 's/^/worktree: /'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('cpu_capability', torch.backends.cpu.get_cpu_capability(), 'threads', torch.get_num_threads())"
start=$SECONDS
python -m unittest discover -s tests -t . -v > "$LOG" 2>&1; rc=$?
grep -E ' \.\.\. (ok|FAIL|ERROR|skipped.*|expected failure|unexpected success)$' "$LOG" | sed -E 's/^([^ ]+) \(([^)]+)\).* \.\.\. /\2.\1 ... /'
grep -E '^(FAIL|ERROR): ' "$LOG"
grep -E '^Ran [0-9]+ tests|^OK|^FAILED' "$LOG"
echo "exit=$rc seconds=$((SECONDS - start))"
