#!/bin/bash
# one stream of K3 tasks, sequential: bash k3_stream.sh <task> <task> ...
for t in "$@"; do
  bash /mnt/data/peeradon/h2snn-ablprep/docs/card4_evidence/jobs/run_py.sh scripts/known_answer/s1pb.py --task "$t" --out_dir ../docs/card4_evidence/k3 > /mnt/data/peeradon/h2snn-ablprep/H2S-NN/logs/k3_task$t.log 2>&1
  echo "task $t exit $?"
done
