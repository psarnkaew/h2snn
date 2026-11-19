#!/bin/bash
# TSUBAME card T7.1-T7.4: each item's targeted tests and mutation into its own evidence file (run from H2S-NN/).
E=../docs/tsubame_evidence
run() {  # run <evidence name> <test modules> [mutation script]
    local name="$1" mods="$2" mut="${3:-}"
    {
        echo "head=$(git rev-parse HEAD) host=$(hostname) job=${SLURM_JOB_ID}"
        echo "tests: ${mods}"
        python -m unittest -v ${mods} > ${E}/.${name}.tmp 2>&1; local c=$?
        grep -E '\.\.\. |^Ran|^OK|^FAILED|Error' ${E}/.${name}.tmp | grep -vE '^  (node|link|graph)_|Deprecat' | cut -c1-220
        echo "tests exit=${c}"
        if [ -n "${mut}" ]; then
            echo "--- mutation ${mut}"
            python ${E}/${mut} > ${E}/.${name}.mut.tmp 2>&1; local m=$?
            grep -E '^(MUTATION|Ran|FAILED|OK)|^AssertionError' ${E}/.${name}.mut.tmp | head -8 | cut -c1-220
            echo "mutation exit=${m}"
        fi
        rm -f ${E}/.${name}.tmp ${E}/.${name}.mut.tmp
    } > ${E}/${name}.txt 2>&1
}
run t7_1_tests "tests.test_tsubame_h2snn_lines tests.test_tsubame_collect"
run t7_2_tests "tests.test_tsubame_collect_h2snn tests.test_tsubame_collect" t7_2_mutation.py
run t7_3_tests "tests.test_tsubame_final_seeds_gate tests.test_tsubame_cell_list" t7_3_mutation.py
run t7_4_tests "tests.test_tsubame_phases tests.test_tsubame_cell_list" t7_4_mutation.py
grep -hE "^(head|tests exit|mutation exit|Ran|MUTATION)" ${E}/t7_[1-4]_tests.txt
