#!/bin/bash
# Part M3b B2 card mutation: re-enable the download path (the offline subclass's `_download` hook removed) ->
# tests.test_h2gb_nc ...test_oag_with_empty_raw_never_downloads fails; restored inside this job.
#   srun --jobid=<allocation> --overlap bash docs/partM3b_evidence/b2_mutation_job.sh
set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"
S=src/data/h2gb_nc.py
cp "$S" "${TMPDIR:-/tmp}/h2gb_nc.orig.$$"
python - <<'PY'
from pathlib import Path
p = Path("src/data/h2gb_nc.py"); t = p.read_text()
old = "        def _download(self) -> None:\n            self._require_processed()\n\n"
assert t.count(old) == 1
p.write_text(t.replace(old, ""))
PY
T=tests.test_h2gb_nc.H2GBLoaderTests.test_oag_with_empty_raw_never_downloads
python -m unittest "$T" 2>&1 | grep -E "^(FAIL|ERROR):|^AssertionError|download called|^Ran|^(OK|FAILED)"
cp "${TMPDIR:-/tmp}/h2gb_nc.orig.$$" "$S"; cmp "$S" "${TMPDIR:-/tmp}/h2gb_nc.orig.$$" && echo "restored: identical"
python -m unittest "$T" 2>&1 | tail -n 1
echo "end=$(date -Is)"
