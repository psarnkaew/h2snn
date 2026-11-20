#!/bin/bash
# Card 10 L1: the lab-cluster (SLURM) counterpart of scripts/tsubame/sweep_slots.sh. One array task = one GPU = one line.
#
#   submitted by scripts/slurm/submit_lab.sh (--gres, --mem, --time, --array, --chdir, --output, --dependency are set
#   there), as
#   sbatch --array=1-N --gres=gpu:<type>:1 --mem=<M> --time=<T> --chdir=<checkout>/H2S-NN \
#       scripts/slurm/lab_slots.sbatch <list> <out> <gpu type>
#   from H2S-NN/ of a read-only checkout at /mnt/data/peeradon/h2snn-runs/<sha>/ (directory named by its HEAD).
#
# Runnable lines of <list> are numbered 1, 2, ... with blank and '#' lines skipped (scripts/tsubame/lines.py
# runnable_lines). Array task t runs runnable line t once, as `python <line> --resume`, on the GPU SLURM gave it
# (CUDA_VISIBLE_DEVICES as SLURM sets it), with the same outputs as sweep_slots.sh with S = 1 (slot k = 0):
#   line log    <out>/logs/lines/<list name>.line<NNN>.<job>.<task>.0.log
#   provenance  <out>/logs/prov/<job>.<task>.0.jsonl   (start and end events, the format collect.py reads)
# <job> is the array's job id (SLURM_ARRAY_JOB_ID), <task> the array task id, as JOB_ID / SGE_TASK_ID on TSUBAME.
# The start event also holds "site":"lab" and "gpu_type" (condition 1 of card 10; collect.py ignores extra keys).
# A finished line exits at once (--resume); a line killed by the 72 h limit resumes in the next leg. A --final_seeds
# line whose seed files all exist is skipped. The GPU's name must match <gpu type>, else the task is refused.
#SBATCH --job-name=lab-slots
#SBATCH --cpus-per-task=4
set -uo pipefail

LIST="${1:?usage: lab_slots.sbatch <list> <out> <gpu type>}"
OUT="${2:?usage: lab_slots.sbatch <list> <out> <gpu type>}"
GPU_TYPE="${3:?usage: lab_slots.sbatch <list> <out> <gpu type>}"
TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set (submit as an array job)}"
JOB="${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is not set}"
K=0  # one GPU per task: the slot index of sweep_slots.sh is always 0

# --- the checkout: clean, and HEAD = the SHA its directory is named by (as sweep_slots.sh) -----------------------
REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "refused: $(pwd) is not inside a git checkout"; exit 2; }
HEAD_SHA="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "refused: the checkout ${REPO} has uncommitted changes"; git status --short --untracked-files=no; exit 2
fi
if [ "$(basename "${REPO}")" != "${HEAD_SHA}" ]; then
    echo "refused: HEAD ${HEAD_SHA} is not the SHA in the directory name ${REPO}"; exit 2
fi

echo "head=${HEAD_SHA} host=$(hostname) job=${JOB} task=${TASK} slots=1 total_slots=${SLURM_ARRAY_TASK_COUNT:-?} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi failed"

# --- the GPU type: the one this list is recorded with (condition 1) ----------------------------------------------
CVD="${CUDA_VISIBLE_DEVICES:-}"
gpu="$(nvidia-smi -i "${CVD%%,*}" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
case "${GPU_TYPE}" in
    3090) want="RTX 3090";; titan_rtx) want="TITAN RTX";; a6000) want="A6000";; 2080ti) want="2080 Ti";;
    *) echo "refused: unknown GPU type ${GPU_TYPE}"; exit 2;;
esac
if [[ "${gpu}" != *"${want}"* ]]; then
    echo "refused: GPU '${gpu}' (CUDA_VISIBLE_DEVICES=${CVD}) is not of type ${GPU_TYPE}"; exit 2
fi

# --- environment: conda (as sweep_slots.sh; no module system on the lab cluster) --------------------------------
# shellcheck disable=SC1090
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || { echo "refused: conda env h2snn-b"; exit 2; }

mapfile -t LINES < <(grep -v -E '^[[:space:]]*(#|$)' "${LIST}" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')
N="${#LINES[@]}"
LISTNAME="$(basename "${LIST}" .txt)"
mkdir -p "${OUT}/logs/lines" "${OUT}/logs/prov"
echo "list=${LIST} runnable_lines=${N} gpu_type=${GPU_TYPE}"

json_str() { python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }

i=$(( TASK - 1 ))
if [ "${i}" -ge "${N}" ]; then echo "task ${TASK}: no runnable line ${TASK} (list has ${N}); nothing to run"; exit 0; fi
line="${LINES[$i]}"
prov="${OUT}/logs/prov/${JOB}.${TASK}.${K}.jsonl"
log="$(printf '%s/logs/lines/%s.line%03d.%s.%s.%s.log' "${OUT}" "${LISTNAME}" $((i + 1)) "${JOB}" "${TASK}" "${K}")"
# shellcheck disable=SC2086  # the line is a list of arguments
before="$(python scripts/tsubame/lines.py snapshot -- ${line} 2>/dev/null || echo null)"
# a --final_seeds line has no resume: once every seed file exists it is not run again in a later leg
# shellcheck disable=SC2086
if python scripts/tsubame/lines.py complete -- ${line} > /dev/null 2>&1; then
    echo "task ${TASK}: line $((i + 1)) complete (all final seeds written), skipped"; echo "end=$(date -Is)"; exit 0
fi
start="$(date -Is)"; t0=${SECONDS}
# the start event is written first: a run killed by the time limit still has its provenance
printf '{"event":"start","line_no":%d,"line":%s,"sha":"%s","host":"%s","job_id":"%s","task_id":"%s","slot":%d,"cuda_visible_devices":"%s","gpu_name":%s,"start":"%s","log":%s,"before":%s,"site":"lab","gpu_type":"%s"}\n' \
    $((i + 1)) "$(json_str "${line}")" "${HEAD_SHA}" "$(hostname)" "${JOB}" "${TASK}" "${K}" "${CVD}" \
    "$(json_str "${gpu}")" "${start}" "$(json_str "${log}")" "${before}" "${GPU_TYPE}" >> "${prov}"
echo "task ${TASK}: line $((i + 1)) -> ${log}"
{
    echo "head=${HEAD_SHA} host=$(hostname) job=${JOB} task=${TASK} slot=${K} gpu=${gpu} start=${start}"
    echo "agent: CUDA_VISIBLE_DEVICES=${CVD} python ${line} --resume"
} >> "${log}"
# shellcheck disable=SC2086  # the line is a list of arguments
python ${line} --resume >> "${log}" 2>&1
code=$?
end="$(date -Is)"
echo "exit=${code} end=${end}" >> "${log}"
# shellcheck disable=SC2086
after="$(python scripts/tsubame/lines.py snapshot -- ${line} 2>/dev/null || echo null)"
printf '{"event":"end","line_no":%d,"job_id":"%s","task_id":"%s","slot":%d,"start":"%s","end":"%s","wall_s":%d,"exit":%d,"after":%s}\n' \
    $((i + 1)) "${JOB}" "${TASK}" "${K}" "${start}" "${end}" $((SECONDS - t0)) "${code}" "${after}" >> "${prov}"
echo "end=$(date -Is)"
exit 0
