#!/bin/bash
# TSUBAME card T1: W&B sweep agents on Grid Engine. One array task = one node (or one GPU) running S slots.
#
#   submitted by scripts/tsubame/submit_chain.sh (resource type, -t, -o, -hold_jid and -g are set there), as
#   qsub -g <group> -l node_f=1 -t 1-K -o <out>/logs scripts/tsubame/sweep_slots.sh <list> <out> <S> <G>
#   from H2S-NN/ of a checkout at $WORK/h2snn-runs/<sha>/ (the repository directory is named by its HEAD).
#
# S = slots per task (GPUs of the resource type: gpu_1 1, node_q 1, node_h 2, node_f 4; card 10 A2.1, MIG types are
# refused by submit_chain.sh); G = total slots of the leg (K * S).
# Runnable lines of <list> are numbered 1, 2, ... with blank and '#' lines skipped (as scripts/slurm/
# run_sweep_agents.sbatch's header states). Slot k (0-based, k < S) of task t is global slot g = (t-1)*S + k and runs
# lines g+1, g+1+G, g+1+2G, ... one after another, each as `python <line> --resume` with CUDA_VISIBLE_DEVICES=k and
# its own log file <out>/logs/lines/<list name>.line<NNN>.<job>.<task>.<k>.log. A finished line exits at once (--resume); a line
# killed by the 24 h limit resumes in the next leg (src/utils/wandb_campaign.py; T2 guard). A --final_seeds line whose
# seed files all exist is skipped (the launcher has no resume for final seeds).
# Provenance of every line run (sha, host, job, task, slot, GPU, start/end, exit code, the line's state before and
# after) goes to <out>/logs/prov/<job>.<task>.<k>.jsonl (one writer per file); collect.py joins it to the records.
#$ -cwd
#$ -j y
#$ -l h_rt=24:00:00
set -uo pipefail

LIST="${1:?usage: sweep_slots.sh <list> <out> <S> <G>}"
OUT="${2:?usage: sweep_slots.sh <list> <out> <S> <G>}"
S="${3:?usage: sweep_slots.sh <list> <out> <S> <G>}"
G="${4:?usage: sweep_slots.sh <list> <out> <S> <G>}"
TASK="${SGE_TASK_ID:?SGE_TASK_ID is not set (submit as an array job)}"
JOB="${JOB_ID:?JOB_ID is not set}"

# --- the checkout: clean, and HEAD = the SHA its directory is named by (guide section 3) -------------------------
REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "refused: $(pwd) is not inside a git checkout"; exit 2; }
HEAD_SHA="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "refused: the checkout ${REPO} has uncommitted changes"; git status --short --untracked-files=no; exit 2
fi
if [ "$(basename "${REPO}")" != "${HEAD_SHA}" ]; then
    echo "refused: HEAD ${HEAD_SHA} is not the SHA in the directory name ${REPO}"; exit 2
fi

echo "head=${HEAD_SHA} host=$(hostname) job=${JOB} task=${TASK} slots=${S} total_slots=${G} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi failed"

# --- environment: module purge, then conda (no -V; guide section 5) -----------------------------------------------
if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
if command -v module > /dev/null 2>&1; then module purge; echo "module purge: done"; else echo "module: not available"; fi
# shellcheck disable=SC1090
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || { echo "refused: conda env h2snn-b"; exit 2; }

mapfile -t LINES < <(grep -v -E '^[[:space:]]*(#|$)' "${LIST}" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')
N="${#LINES[@]}"
LISTNAME="$(basename "${LIST}" .txt)"
mkdir -p "${OUT}/logs/lines" "${OUT}/logs/prov"
echo "list=${LIST} runnable_lines=${N}"

json_str() { python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }

run_slot() {
    local k="$1" g=$(( (TASK - 1) * S + $1 )) i gpu prov log line start end code before after t0
    prov="${OUT}/logs/prov/${JOB}.${TASK}.${k}.jsonl"
    gpu="$(nvidia-smi -i "${k}" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    for (( i = g; i < N; i += G )); do
        line="${LINES[$i]}"
        log="$(printf '%s/logs/lines/%s.line%03d.%s.%s.%s.log' "${OUT}" "${LISTNAME}" $((i + 1)) "${JOB}" "${TASK}" "${k}")"
        # shellcheck disable=SC2086  # the line is a list of arguments
        before="$(python scripts/tsubame/lines.py snapshot -- ${line} 2>/dev/null || echo null)"
        # a --final_seeds line has no resume: once every seed file exists it is not run again in a later leg
        # shellcheck disable=SC2086
        if python scripts/tsubame/lines.py complete -- ${line} > /dev/null 2>&1; then
            echo "slot ${k} (global ${g}): line $((i + 1)) complete (all final seeds written), skipped"; continue
        fi
        start="$(date -Is)"; t0=${SECONDS}
        # the start event is written first: a run killed by the 24 h limit still has its provenance
        printf '{"event":"start","line_no":%d,"line":%s,"sha":"%s","host":"%s","job_id":"%s","task_id":"%s","slot":%d,"cuda_visible_devices":"%s","gpu_name":%s,"start":"%s","log":%s,"before":%s}\n' \
            $((i + 1)) "$(json_str "${line}")" "${HEAD_SHA}" "$(hostname)" "${JOB}" "${TASK}" "${k}" "${k}" \
            "$(json_str "${gpu}")" "${start}" "$(json_str "${log}")" "${before}" >> "${prov}"
        echo "slot ${k} (global ${g}): line $((i + 1)) -> ${log}"
        {
            echo "head=${HEAD_SHA} host=$(hostname) job=${JOB} task=${TASK} slot=${k} gpu=${gpu} start=${start}"
            echo "agent: CUDA_VISIBLE_DEVICES=${k} python ${line} --resume"
        } >> "${log}"
        # shellcheck disable=SC2086  # the line is a list of arguments
        CUDA_VISIBLE_DEVICES="${k}" python ${line} --resume >> "${log}" 2>&1
        code=$?
        end="$(date -Is)"
        echo "exit=${code} end=${end}" >> "${log}"
        # shellcheck disable=SC2086
        after="$(python scripts/tsubame/lines.py snapshot -- ${line} 2>/dev/null || echo null)"
        printf '{"event":"end","line_no":%d,"job_id":"%s","task_id":"%s","slot":%d,"start":"%s","end":"%s","wall_s":%d,"exit":%d,"after":%s}\n' \
            $((i + 1)) "${JOB}" "${TASK}" "${k}" "${start}" "${end}" $((SECONDS - t0)) "${code}" "${after}" >> "${prov}"
    done
}

for (( k = 0; k < S; k++ )); do run_slot "${k}" & done
wait
echo "end=$(date -Is)"
exit 0
