#!/bin/bash
# Card 1f U2/U3: the B1a tests, the existing loop tests, the F3.1 test, then Mutation 1 (CPU job). Run from repo root:
#   PYTHONPATH=$PWD/H2S-NN sbatch --mem=16G --output=docs/b1a_evidence/u3_tests.txt docs/baselines_h2gb_evidence/job.sh b1a-u3 bash ../docs/b1a_evidence/u3_job.sh
set -u
python -m unittest -v tests.test_b1a_validation_subset tests.test_baselines_minibatch \
    tests.test_baselines_blocked_spmm
status=$?
python ../docs/b1a_evidence/u3_mutation1.py
mutation=$?
echo "tests_exit=${status} mutation1_exit=${mutation}"
exit $(( status || mutation ))
