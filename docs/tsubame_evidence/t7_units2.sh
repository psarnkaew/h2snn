#!/bin/bash
# TSUBAME card T7.2 (rerun after the fixture fix) and T7.5 (run from H2S-NN/).
E=../docs/tsubame_evidence
source <(sed -n '/^run() {/,/^}/p' ${E}/t7_units.sh)
run t7_2_tests "tests.test_tsubame_collect_h2snn tests.test_tsubame_collect" t7_2_mutation.py
run t7_5_tests "tests.test_tsubame_host_memory"
python scripts/tsubame/host_memory.py > ${E}/t7_5_host_memory.md 2>&1
grep -hE "^(head|tests exit|mutation exit|Ran|MUTATION)" ${E}/t7_2_tests.txt ${E}/t7_5_tests.txt
cat ${E}/t7_5_host_memory.md
