#!/bin/bash
# Card 9 (copy of card 7's script): freeze the working tree (src, scripts, tests, configs, docs) into a snapshot at
# submission time and sbatch a job script that runs there, so later edits cannot leak into a run.
#   bash docs/card7_evidence/snapshot_submit.sh <label> <job script> [args...]
# The snapshot records HEAD and the SHA-256 of `git diff HEAD` (the uncommitted part, if any).
set -euo pipefail
LABEL="${1:?label}"; JOB="${2:?job script}"; shift 2
WT=/mnt/data/peeradon/h2snn-model
SNAP=/mnt/data/peeradon/h2snn-model-snap/card9_${LABEL}
rm -rf "${SNAP}"; mkdir -p "${SNAP}/H2S-NN"
rsync -a --exclude /vendor --exclude /data --exclude /logs --exclude '__pycache__' --exclude .git \
    "${WT}/H2S-NN/" "${SNAP}/H2S-NN/"
ln -s "${WT}/H2S-NN/vendor" "${SNAP}/H2S-NN/vendor"
ln -s "${WT}/H2S-NN/data" "${SNAP}/H2S-NN/data"
mkdir -p "${SNAP}/H2S-NN/logs"
rsync -a --exclude '__pycache__' "${WT}/docs/" "${SNAP}/docs/"
# the worktree's gitdir pointer, so `git rev-parse HEAD` inside the snapshot reports the branch head
cp "${WT}/.git" "${SNAP}/.git"
( cd "${WT}" && echo "head=$(git rev-parse HEAD) diff_sha256=$(git diff HEAD -- . ':!docs/card9_evidence' | sha256sum | cut -c1-16) untracked=$(git status --short | grep -c '^??' || true) snapshot_at=$(date -Is)" ) > "${SNAP}/SNAPSHOT.txt"
cat "${SNAP}/SNAPSHOT.txt"
SNAP="${SNAP}" sbatch --export=ALL,SNAP="${SNAP}" "${JOB}" "$@"
