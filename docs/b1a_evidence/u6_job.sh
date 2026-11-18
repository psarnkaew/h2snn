#!/bin/bash
# Card 1f U6.5: the U5 set + the U6 tests + test_baseline_final_seeds, test_wandb_and_runs, test_failed_trials, then the
# in-process mutations (U5 and U6). CPU job, no --gres, 16G. From repo root:
#   PYTHONPATH=$PWD/H2S-NN sbatch --mem=16G --output=docs/b1a_evidence/u6_tests.txt docs/baselines_h2gb_evidence/job.sh b1a-u6 bash ../docs/b1a_evidence/u6_job.sh
set -u
python -m unittest -v tests.test_b1a_validation_subset tests.test_baselines_minibatch \
    tests.test_baselines.RegistryTests.test_campaign_matrix_matches_planned_cell_counts \
    tests.test_baselines.LaneNotebookTests.test_committed_notebooks_match_the_generator \
    tests.test_baseline_final_seeds tests.test_wandb_and_runs tests.test_failed_trials
status=$?
python ../docs/b1a_evidence/u5_mutations.py
mutation=$?
echo "tests_exit=${status} mutations_exit=${mutation}"
exit $(( status || mutation ))
