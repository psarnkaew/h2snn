#!/bin/bash
# Card 10 L1: submit a chain of lab_slots.sbatch legs for one cell list on the lab cluster (counterpart of
# scripts/tsubame/submit_chain.sh).
#
#   cd /mnt/data/peeradon/h2snn-runs/<sha>/H2S-NN
#   bash scripts/slurm/submit_lab.sh --list <out>/cells_tuning.lab.txt --out <out> --gpu-type 3090 \
#       --mem 48G --time 72:00:00 [--legs L] [--cpus C] [--max-parallel P]
#
# Leg 1 is an array of N tasks (N = runnable lines; task t runs line t on one GPU of --gpu-type); legs 2..L are the
# same array, each with --dependency=afterany on the previous leg (72 h limit per job; a line resumes with --resume).
# --gpu-type, --mem and --time are required (no default GPU type); --time is at most 72 h.
# Condition 1 (one cell = one GPU type): every line's cell (scripts/tsubame/lines.py token: the state and output
# directory it writes) is recorded with the GPU type in <out>/lab_gpu_types.tsv; a list with a cell already recorded
# under another GPU type is refused, and so is a line lines.py cannot parse (its cell would be unknown).
# Job ids go to stdout and to <out>/submitted.tsv (columns of submit_chain.sh). A list that already has a chain with a
# job still pending or running (squeue, called once) is refused: one chain per list.
set -euo pipefail

usage() {
    echo "usage: submit_lab.sh --list FILE --out DIR --gpu-type {3090,titan_rtx,a6000,2080ti} --mem M --time T" \
         "[--legs L] [--cpus C] [--max-parallel P]" >&2
    exit 2
}
LIST="" OUT="" GPU_TYPE="" MEM="" TIME="" LEGS="1" CPUS="4" MAXPAR=""
while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || usage
    case "$1" in
        --list) LIST="$2";; --out) OUT="$2";; --gpu-type) GPU_TYPE="$2";; --mem) MEM="$2";; --time) TIME="$2";;
        --legs) LEGS="$2";; --cpus) CPUS="$2";; --max-parallel) MAXPAR="$2";;
        *) usage;;
    esac
    shift 2
done
[ -n "${LIST}" ] && [ -n "${OUT}" ] && [ -n "${GPU_TYPE}" ] && [ -n "${MEM}" ] && [ -n "${TIME}" ] || usage
[ -f "${LIST}" ] || { echo "no list ${LIST}" >&2; exit 2; }
case "${GPU_TYPE}" in 3090|titan_rtx|a6000|2080ti) ;; *) echo "--gpu-type must be 3090, titan_rtx, a6000 or 2080ti" >&2; exit 2;; esac
[[ "${LEGS}" =~ ^[1-9][0-9]*$ ]] || { echo "--legs must be a positive integer" >&2; exit 2; }
[[ "${CPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "--cpus must be a positive integer" >&2; exit 2; }
[ -z "${MAXPAR}" ] || [[ "${MAXPAR}" =~ ^[1-9][0-9]*$ ]] || { echo "--max-parallel must be a positive integer" >&2; exit 2; }
[[ "${MEM}" =~ ^[1-9][0-9]*[KMGT]?$ ]] || { echo "--mem must be a SLURM memory size (e.g. 48G)" >&2; exit 2; }
# --time: [D-]HH:MM:SS, at most 72 h (lab rule)
if [[ "${TIME}" =~ ^(([0-9]+)-)?([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]]; then
    secs=$(( (10#${BASH_REMATCH[2]:-0} * 24 + 10#${BASH_REMATCH[3]}) * 3600 + 10#${BASH_REMATCH[4]} * 60 + 10#${BASH_REMATCH[5]} ))
    [ "${secs}" -gt 0 ] && [ "${secs}" -le $(( 72 * 3600 )) ] || { echo "--time must be > 0 and at most 72:00:00" >&2; exit 2; }
else
    echo "--time must be [D-]HH:MM:SS" >&2; exit 2
fi
SCRIPT="scripts/slurm/lab_slots.sbatch"
[ -f "${SCRIPT}" ] || { echo "run from H2S-NN/ (no ${SCRIPT})" >&2; exit 2; }
H2="$(pwd)"
LIST="$(cd "$(dirname "${LIST}")" && pwd)/$(basename "${LIST}")"
mkdir -p "${OUT}/logs"
OUT="$(cd "${OUT}" && pwd)"

N="$(grep -c -v -E '^[[:space:]]*(#|$)' "${LIST}" || true)"
[ "${N}" -gt 0 ] || { echo "${LIST} has no runnable line" >&2; exit 2; }

# --- condition 1: one cell = one GPU type -------------------------------------------------------------------------
TYPES="${OUT}/lab_gpu_types.tsv"
NEW="$(mktemp)"; trap 'rm -f "${NEW}"' EXIT
python - "${LIST}" "${TYPES}" "${GPU_TYPE}" "${NEW}" <<'PY'
import csv, sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts.tsubame.lines import parse_line, runnable_lines

list_path, types_path, gpu_type, new_path = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
recorded = {}
if types_path.exists():
    with types_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            recorded.setdefault(row["cell"], row["gpu_type"])
new, refused = [], []
for number, text in enumerate(runnable_lines(list_path.read_text(encoding="utf-8")), start=1):
    try:
        cell = parse_line(text).token
    except ValueError as err:
        refused.append(f"line {number}: cannot name its cell ({err})")
        continue
    if cell in recorded and recorded[cell] != gpu_type:
        refused.append(f"line {number}: cell {cell} is recorded with GPU type {recorded[cell]}, not {gpu_type}")
    elif cell not in recorded:
        recorded[cell] = gpu_type
        new.append((cell, gpu_type, number, text))
if refused:
    print("refused (one cell = one GPU type):\n  " + "\n  ".join(refused), file=sys.stderr)
    raise SystemExit(3)
with new_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    for cell, gpu, number, text in new:
        writer.writerow([cell, gpu, str(list_path), number, text])
PY

# --- one chain per list: refuse while a job of an earlier chain for this list is pending or running ---------------
TSV="${OUT}/submitted.tsv"
if [ -f "${TSV}" ]; then
    previous="$(awk -F'\t' -v l="${LIST}" 'NR > 1 && $2 == l {print $4}' "${TSV}" | sort -u)"
    if [ -n "${previous}" ]; then
        live="$(squeue -h -o '%F' | sort -u)"
        busy="$(comm -12 <(echo "${previous}") <(echo "${live}"))"
        if [ -n "${busy}" ]; then
            echo "refused: ${LIST} already has a chain with jobs pending or running: $(echo ${busy})" >&2; exit 3
        fi
    fi
else
    printf 'submitted\tlist\tleg\tjob_id\ttype\ttasks\tslots_per_task\ttotal_slots\n' > "${TSV}"
fi
[ -f "${TYPES}" ] || printf 'cell\tgpu_type\tlist\tline_no\tline\n' > "${TYPES}"

NAME="lab-$(basename "${LIST}" .txt)"
ARRAY="1-${N}${MAXPAR:+%${MAXPAR}}"
prev=""
echo "list=${LIST} runnable_lines=${N} gpu_type=${GPU_TYPE} mem=${MEM} time=${TIME} array=${ARRAY} legs=${LEGS}"
for (( leg = 1; leg <= LEGS; leg++ )); do
    dep=()
    [ -n "${prev}" ] && dep=(--dependency="afterany:${prev}")
    out="$(sbatch --parsable --job-name="${NAME}-L${leg}" --array="${ARRAY}" --gres="gpu:${GPU_TYPE}:1" \
        --cpus-per-task="${CPUS}" --mem="${MEM}" --time="${TIME}" --chdir="${H2}" \
        --output="${OUT}/logs/slurm-%x-%A_%a.out" "${dep[@]}" "${SCRIPT}" "${LIST}" "${OUT}" "${GPU_TYPE}")"
    job="${out%%;*}"
    [[ "${job}" =~ ^[0-9]+$ ]] || { echo "sbatch returned '${out}' for leg ${leg}; stopping" >&2; exit 4; }
    if [ "${leg}" -eq 1 ]; then cat "${NEW}" >> "${TYPES}"; fi    # recorded once the first leg is accepted
    printf '%s\t%s\t%d\t%s\t%s\t%s\t%d\t%d\n' "$(date -Is)" "${LIST}" "${leg}" "${job}" "gpu:${GPU_TYPE}" \
        "1-${N}" 1 "${N}" >> "${TSV}"
    echo "leg ${leg}: job ${job}${prev:+ (afterany:${prev})}"
    prev="${job}"
done
