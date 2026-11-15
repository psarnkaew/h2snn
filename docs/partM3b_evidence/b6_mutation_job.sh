#!/bin/bash
# Part M3b B6 card mutations, each against its target test only, restored inside this job:
#   1. each endpoint's scope includes the other endpoints' cells -> tests.test_large_lp StepScopeTests (i) fails;
#   2. held-out removal restricted to one orientation -> tests.test_large_lp HeldOutTests (ii) fails.
#   srun --jobid=<allocation> --overlap bash docs/partM3b_evidence/b6_mutation_job.sh
set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"
F='^(FAIL|ERROR):|^AssertionError|^Ran|^(OK|FAILED)'

A=src/trainers/minibatch_lp.py
cp "$A" "${TMPDIR:-/tmp}/mlp.orig.$$"
echo "===== mutation 1: each endpoint's scope includes the other endpoints' cells"
python - <<'PY'
from pathlib import Path
p = Path("src/trainers/minibatch_lp.py"); t = p.read_text()
old = "batch = build_scope_batch(self.graph, part, [cells.of(int(t)) for t in part.tolist()])"
new = ("batch = build_scope_batch(self.graph, part, [tuple(g for s in part.tolist() for g in cells.of(int(s)))"
       " for t in part.tolist()])")
assert t.count(old) == 1
p.write_text(t.replace(old, new))
PY
python -m unittest tests.test_large_lp.StepScopeTests.test_i_shared_endpoint_computed_once_and_equal_to_its_own_scope 2>&1 | grep -E "$F"
cp "${TMPDIR:-/tmp}/mlp.orig.$$" "$A"; cmp "$A" "${TMPDIR:-/tmp}/mlp.orig.$$" && echo "restored: identical"

B=src/data/large_lp.py
cp "$B" "${TMPDIR:-/tmp}/llp.orig.$$"
echo "===== mutation 2: held-out removal restricted to one orientation"
python - <<'PY'
from pathlib import Path
p = Path("src/data/large_lp.py"); t = p.read_text()
old = ("    e_lo = torch.minimum(edge_index[0], edge_index[1]).long()\n"
       "    e_hi = torch.maximum(edge_index[0], edge_index[1]).long()\n")
assert t.count(old) == 1
p.write_text(t.replace(old, "    e_lo = edge_index[0].long()\n    e_hi = edge_index[1].long()\n"))
PY
python -m unittest tests.test_large_lp.HeldOutTests 2>&1 | grep -E "$F"
cp "${TMPDIR:-/tmp}/llp.orig.$$" "$B"; cmp "$B" "${TMPDIR:-/tmp}/llp.orig.$$" && echo "restored: identical"
python -m unittest tests.test_large_lp 2>&1 | tail -n 1
echo "end=$(date -Is)"
