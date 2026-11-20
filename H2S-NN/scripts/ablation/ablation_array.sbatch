#!/bin/bash
# Card 4 K6 (5); card 9 W1: one SLURM array task per manifest row (`scripts/ablation/seed_runs.py`); since
# pre-registration v3 every row is a configuration (a) `runs` row (seeds through scripts/run_h2snn_runs.py).
#   sbatch --array=1-<rows> scripts/ablation/ablation_array.sbatch <manifest.tsv>
# Row i of the array = line i + 1 of the manifest (line 1 is the header). Never run in this card.
#SBATCH --job-name=h2snn-ablation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ablation-%A_%a.out

set -uo pipefail
manifest="${1:?usage: ablation_array.sbatch <manifest.tsv>}"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b

echo "job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
row=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:?} + 1 ))p" "$manifest")
[ -n "$row" ] || { echo "no manifest row ${SLURM_ARRAY_TASK_ID}"; exit 2; }
IFS=$'\t' read -r kind dataset arm configuration config base_seed num_runs exp_name command <<< "$row"
echo "row: kind=${kind} dataset=${dataset} arm=${arm} configuration=${configuration} config=${config}"
[ -f "$config" ] || { echo "missing configuration ${config} (a (b) row needs tuned_config first)"; exit 2; }
sha256sum "$config"
eval "$command"
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
