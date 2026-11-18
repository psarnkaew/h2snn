#!/bin/bash
# Model change TD-1, M0: test count at the base (collection only, no test run, no GPU).
#   sbatch docs/model_change_td1_evidence/m0_collect_job.sh <git-ref>
# Collects in a detached checkout of <git-ref> (a temporary worktree), so the count is the base's.
#SBATCH --job-name=td1-m0
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=0:20:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/model_change_td1_evidence/m0_collect_%j.txt
set -uo pipefail
REF="${1:?usage: m0_collect_job.sh <git-ref>}"
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
cd /mnt/data/peeradon/h2snn-model || exit 2
echo "git=$(git rev-parse HEAD) collecting at ref=${REF} ($(git rev-parse ${REF}))"
TMP=$(mktemp -d /tmp/td1-m0-XXXX)
git archive "${REF}" H2S-NN/src H2S-NN/tests H2S-NN/scripts H2S-NN/configs | tar -x -C "${TMP}"
cd "${TMP}/H2S-NN" || exit 2
python - <<'PY'
import unittest
suite = unittest.defaultTestLoader.discover("tests")
def walk(s):
    for t in s:
        if isinstance(t, unittest.TestSuite):
            yield from walk(t)
        else:
            yield t
tests = list(walk(suite))
failed = [t.id() for t in tests if type(t).__name__ == "_FailedTest"]
from collections import Counter
per_file = Counter(t.id().split(".")[0] if not t.id().startswith("tests.") else t.id().split(".")[1] for t in tests)
print("unittest_discover_count", len(tests), "import_failures", failed)
for name, n in sorted(per_file.items()):
    print(f"  {name} {n}")
PY
python -m pytest --version 2>&1 | head -1
python -m pytest --collect-only -q tests 2>&1 | tail -3
rm -rf "${TMP}"
echo "end=$(date -Is)"
