#!/bin/bash
# TSUBAME card T1: submit a chain of sweep_slots.sh legs for one cell list (guide section 6).
#
#   cd $WORK/h2snn-runs/<sha>/H2S-NN
#   bash scripts/tsubame/submit_chain.sh --list <out>/cells_tuning.txt --type node_f --legs 6 \
#       [--out <out>] [--first-task a --last-task b]
#
# Leg 1 is an array of K = ceil(runnable lines / S) tasks (S = GPUs per task: gpu_1 1, node_q 1, node_h 2, node_f 4;
# node_o and gpu_h (MIG, half an H100) are refused (card 10 A2.1); G = K * S slots, so every
# line has its own slot); legs 2..L are the same array, each held (-hold_jid) on the previous leg. --first-task /
# --last-task submit only tasks a..b of that array (K and G unchanged, so each line keeps its slot).
# Lines of node_f task t are lines 4t-3..4t; as gpu_1 (S = 1) they are tasks 4t-3..4t (guide section 7).
# The group is read from ~/.t4_group and goes on the qsub command line only; it is never printed or written.
# Job ids go to stdout and to <out>/submitted.tsv. A list that already has a chain with a job still queued or
# running (qstat, called once) is refused: one chain per list.
set -euo pipefail

usage() { echo "usage: submit_chain.sh --list FILE --type {gpu_1,node_q,node_h,node_f} --legs L [--out DIR] [--first-task a --last-task b]" >&2; exit 2; }
LIST="" TYPE="" LEGS="" OUT="" FIRST="" LAST=""
while [ $# -gt 0 ]; do
    case "$1" in
        --list) LIST="$2"; shift 2;;
        --type) TYPE="$2"; shift 2;;
        --legs) LEGS="$2"; shift 2;;
        --out) OUT="$2"; shift 2;;
        --first-task) FIRST="$2"; shift 2;;
        --last-task) LAST="$2"; shift 2;;
        *) usage;;
    esac
done
[ -n "${LIST}" ] && [ -n "${TYPE}" ] && [ -n "${LEGS}" ] || usage
[ -f "${LIST}" ] || { echo "no list ${LIST}" >&2; exit 2; }
# card 10 A2.1: every whole-H100 type, S = its GPUs (handbook 5.1.1); MIG types are refused (one full H100 per trial)
case "${TYPE}" in
    gpu_1) S=1;; node_q) S=1;; node_h) S=2;; node_f) S=4;;
    node_o|gpu_h) echo "refused: --type ${TYPE} is a MIG slice (half an H100); every trial runs on one full H100 94 GB (spec range rule)" >&2; exit 2;;
    *) echo "--type must be gpu_1, node_q, node_h or node_f" >&2; exit 2;;
esac
[[ "${LEGS}" =~ ^[1-9][0-9]*$ ]] || { echo "--legs must be a positive integer" >&2; exit 2; }
LIST="$(cd "$(dirname "${LIST}")" && pwd)/$(basename "${LIST}")"
OUT="${OUT:-$(dirname "${LIST}")}"
mkdir -p "${OUT}/logs"
OUT="$(cd "${OUT}" && pwd)"

N="$(grep -c -v -E '^[[:space:]]*(#|$)' "${LIST}" || true)"
[ "${N}" -gt 0 ] || { echo "${LIST} has no runnable line" >&2; exit 2; }
K=$(( (N + S - 1) / S ))
G=$(( K * S ))
FIRST="${FIRST:-1}" LAST="${LAST:-${K}}"
if [ "${FIRST}" -lt 1 ] || [ "${LAST}" -gt "${K}" ] || [ "${FIRST}" -gt "${LAST}" ]; then
    echo "tasks ${FIRST}-${LAST} outside 1-${K}" >&2; exit 2
fi
GROUP_FILE="${HOME}/.t4_group"
[ -s "${GROUP_FILE}" ] || { echo "no TSUBAME group in ~/.t4_group" >&2; exit 2; }
GROUP="$(head -1 "${GROUP_FILE}" | tr -d '[:space:]')"

# --- one chain per list: refuse while a job of an earlier chain for this list is queued or running ---------------
TSV="${OUT}/submitted.tsv"
if [ -f "${TSV}" ]; then
    previous="$(awk -F'\t' -v l="${LIST}" 'NR > 1 && $2 == l {print $4}' "${TSV}" | sort -u)"
    if [ -n "${previous}" ]; then
        live="$(qstat | awk 'NR > 2 {print $1}' | sort -u)"
        busy="$(comm -12 <(echo "${previous}") <(echo "${live}"))"
        if [ -n "${busy}" ]; then
            echo "refused: ${LIST} already has a chain with jobs queued or running: $(echo ${busy})" >&2; exit 3
        fi
    fi
else
    printf 'submitted\tlist\tleg\tjob_id\ttype\ttasks\tslots_per_task\ttotal_slots\n' > "${TSV}"
fi

SCRIPT="scripts/tsubame/sweep_slots.sh"
[ -f "${SCRIPT}" ] || { echo "run from H2S-NN/ (no ${SCRIPT})" >&2; exit 2; }
NAME="t4-$(basename "${LIST}" .txt)"
prev=""
echo "list=${LIST} runnable_lines=${N} type=${TYPE} S=${S} K=${K} G=${G} tasks=${FIRST}-${LAST} legs=${LEGS}"
for (( leg = 1; leg <= LEGS; leg++ )); do
    hold=()
    [ -n "${prev}" ] && hold=(-hold_jid "${prev}")
    out="$(qsub -terse -g "${GROUP}" -l "${TYPE}=1" -l h_rt=24:00:00 -t "${FIRST}-${LAST}" -N "${NAME}-L${leg}" \
        -o "${OUT}/logs" "${hold[@]}" "${SCRIPT}" "${LIST}" "${OUT}" "${S}" "${G}")"
    job="${out%%.*}"
    [[ "${job}" =~ ^[0-9]+$ ]] || { echo "qsub returned '${out}' for leg ${leg}; stopping" >&2; exit 4; }
    printf '%s\t%s\t%d\t%s\t%s\t%s\t%d\t%d\n' "$(date -Is)" "${LIST}" "${leg}" "${job}" "${TYPE}" \
        "${FIRST}-${LAST}" "${S}" "${G}" >> "${TSV}"
    echo "leg ${leg}: job ${job}${prev:+ (held on ${prev})}"
    prev="${job}"
done
