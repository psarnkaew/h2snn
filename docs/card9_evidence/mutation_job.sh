#!/bin/bash
# Card 9 W9: the five mutations of the card, each against its target test only (environment h2snn-b, CPU), in a
# snapshot (snapshot_submit.sh); the snapshot's file is restored and compared after each mutation.
#   bash docs/card9_evidence/snapshot_submit.sh mut docs/card9_evidence/mutation_job.sh
#SBATCH --job-name=c9-mut
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card9_evidence/mutation_%j.txt
set -uo pipefail
cd "${SNAP:?run through snapshot_submit.sh}/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b MPLBACKEND=Agg
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "snapshot=${SNAP} $(cat ${SNAP}/SNAPSHOT.txt)"
echo "git=$(git -C /mnt/data/peeradon/h2snn-model rev-parse HEAD)"
mutate() {  # $1 label, $2 file, $3 old, $4 new, $5 test id
    cp "$2" /tmp/c9mut.orig.$$
    python - "$2" "$3" "$4" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]); t = p.read_text()
old, new = sys.argv[2], sys.argv[3]
assert t.count(old) == 1, old
p.write_text(t.replace(old, new))
PY
    echo "===== mutation $1 -> target $5"
    python -m unittest -v "$5" 2>&1 | grep -E "^(test_|AssertionError|Ran |OK|FAILED)" | cut -c1-300
    cp /tmp/c9mut.orig.$$ "$2" && cmp "$2" /tmp/c9mut.orig.$$ && echo "restored: identical"
}
echo "===== baseline: the five targets unmutated"
python -m unittest -v tests.test_card9_v3.RuleTests.test_T1_rule_edge_cases \
    tests.test_card9_v3.ScaleFreeMarginTests.test_T4_scaling_x_star_leaves_the_scale_free_margin_unchanged \
    tests.test_card9_v3.NormalisedEnergyTests.test_T5_scaling_leaves_e_tilde_unchanged_and_r2_reads_it \
    tests.test_card9_v3.PerturbationRuleTests.test_T6_e1_needs_both_datasets_e2_and_absent 2>&1 \
    | grep -E "^(test_|Ran |OK|FAILED)" | cut -c1-200
mutate "m1: wins >= 3" scripts/ablation/stats.py \
    "WINS_NEEDED = 4 " "WINS_NEEDED = 3 " \
    tests.test_card9_v3.RuleTests.test_T1_rule_edge_cases
mutate "m2: drop mean_difference > 0" scripts/ablation/stats.py \
    "    return bool(mean_difference > 0 and wins >= WINS_NEEDED)" "    return bool(wins >= WINS_NEEDED)" \
    tests.test_card9_v3.RuleTests.test_T1_rule_edge_cases
mutate "m3: margin not divided by D" scripts/analysis/mechanism_common.py \
    "    return m / centroid_scale(centres)" "    return m" \
    tests.test_card9_v3.ScaleFreeMarginTests.test_T4_scaling_x_star_leaves_the_scale_free_margin_unchanged
mutate "m4: e_tilde_v = raw e_v" scripts/analysis/two_regime.py \
    "    return np.asarray(e_v, dtype=np.float64) / np.square(x).sum(axis=-1)" \
    "    return np.asarray(e_v, dtype=np.float64)" \
    tests.test_card9_v3.NormalisedEnergyTests.test_T5_scaling_leaves_e_tilde_unchanged_and_r2_reads_it
mutate "m5: E1 on one dataset only" scripts/perturbation/rules.py \
    '            out["E1"][method] = {"per_dataset": per, "holds": both({d: per[d]["holds"] for d in DATASETS})}' \
    '            out["E1"][method] = {"per_dataset": per, "holds": per["acm"]["holds"]}' \
    tests.test_card9_v3.PerturbationRuleTests.test_T6_e1_needs_both_datasets_e2_and_absent
echo "end=$(date -Is)"
