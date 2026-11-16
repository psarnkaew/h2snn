#!/bin/bash
# Card 4 K4/K5 runs inside allocation 16741 (array 16869 was queued with no start time): the same
# per-task script as known_answer_array.sbatch, run sequentially for the given task ids.
for t in "$@"; do
  SLURM_ARRAY_TASK_ID=$t bash /mnt/data/peeradon/h2snn-ablprep/docs/card4_evidence/jobs/known_answer_array.sbatch \
      > /mnt/data/peeradon/h2snn-ablprep-run/H2S-NN/logs/ka-alloc-$t.out 2>&1
  echo "task $t exit $?"
done
