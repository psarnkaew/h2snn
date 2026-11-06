#!/usr/bin/env sh
# usage: sh suite_compare.sh <full_suite.txt> <out_fail_names.txt>
# Extracts FAIL/ERROR lines and compares them with the pre-D7 baseline (u0_fail_names.txt).
grep -E "^(FAIL|ERROR):" "$1" | tr -d '\r' | sort > "$2"
sed 's/^\xEF\xBB\xBF//' u0_fail_names.txt | tr -d '\r' | sort > .baseline_sorted
grep -E "^Ran [0-9]+ tests|^(OK|FAILED)" "$1" | tr -d '\r'
if diff .baseline_sorted "$2" > /dev/null; then echo "failure names identical to the pre-D7 baseline ($(wc -l < "$2") entries)"; status=0; else echo "FAILURE SET CHANGED:"; diff .baseline_sorted "$2"; status=1; fi
rm -f .baseline_sorted
exit $status
