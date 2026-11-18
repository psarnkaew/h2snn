#!/bin/bash
# Card 1f U5.6: only the new/changed tests and the two tests that failed in U4, then the U5 mutations (CPU job). From repo root:
#   PYTHONPATH=$PWD/H2S-NN sbatch --mem=16G --output=docs/b1a_evidence/u5_tests.txt docs/baselines_h2gb_evidence/job.sh b1a-u5 bash ../docs/b1a_evidence/u5_job.sh
set -u
python -m unittest -v tests.test_b1a_validation_subset tests.test_baselines_minibatch \
    tests.test_baselines.RegistryTests.test_campaign_matrix_matches_planned_cell_counts \
    tests.test_baselines.LaneNotebookTests.test_committed_notebooks_match_the_generator \
    tests.test_wandb_and_runs tests.test_failed_trials
status=$?
python ../docs/b1a_evidence/u5_mutations.py
mutation=$?
echo "tests_exit=${status} mutations_exit=${mutation}"
exit $(( status || mutation ))
