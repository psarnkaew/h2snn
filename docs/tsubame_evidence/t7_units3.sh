#!/bin/bash
# TSUBAME card T7.2, second rerun after the fixture fixes (run from H2S-NN/).
E=../docs/tsubame_evidence
source <(sed -n '/^run() {/,/^}/p' ${E}/t7_units.sh)
run t7_2_tests "tests.test_tsubame_collect_h2snn tests.test_tsubame_collect" t7_2_mutation.py
grep -hE "^(head|tests exit|mutation exit|Ran|MUTATION)" ${E}/t7_2_tests.txt
