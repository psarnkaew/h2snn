#!/bin/bash
# W&B sweep agents on N GPUs (card 2 D3): a job array, one agent (one tuning launcher) per GPU, one cell per task.
#
#   cd /mnt/data/peeradon/h2snn/H2S-NN
#   sbatch --array=1-<cells>%<gpus> scripts/slurm/run_sweep_agents.sbatch <cell list>
#
# <cell list>: one launcher command per line (no leading "python"), e.g.
#   scripts/run_h2snn_wandb_tuning.py --config configs/experiments/acm_h2snn_v3b.yaml --output_dir experiments/tuning --wandb_project h2snn
#   scripts/run_baseline_tuning.py --task node_classification --models gcn --datasets acm --num_runs 1 --output_dir experiments/baselines --wandb_project h2snn_baseline
# Blank lines and lines starting with '#' are skipped by numbering (line N of the file is array task N).
#
# Resumable: every task runs its launcher with --resume. The launcher keeps a fingerprinted local state per sweep
# (src/utils/wandb_campaign.py: the sweep id and every finished trial, saved after each trial); a first run creates the
# sweep, a later run of the same line resumes it. An agent killed by SLURM (time limit, preemption, scancel) loses at
# most the trial it was running (left "crashed" in W&B; its attempt is re-run on resume) and the sweep stays usable:
#   sbatch --array=<N> scripts/slurm/run_sweep_agents.sbatch <cell list>      # resume line N
# Never run the same line in two tasks at once (one state file per sweep).
#SBATCH --job-name=sweep-agent
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
set -euo pipefail
LIST="${1:?usage: sbatch --array=1-N%G run_sweep_agents.sbatch <cell list>}"
LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")"
case "$LINE" in ''|'#'*) echo "line ${SLURM_ARRAY_TASK_ID} of $LIST is empty or a comment: nothing to run"; exit 0;; esac
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
echo "head=$(git rev-parse HEAD) host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "agent: python ${LINE} --resume"
# shellcheck disable=SC2086  # the line is a list of arguments
exec python ${LINE} --resume
