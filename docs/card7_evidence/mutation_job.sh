#!/bin/bash
# Card 7: the two mutations of the card, each against its target test only (environment E, CPU), in a snapshot
# (snapshot_submit.sh); the snapshot's file is restored and compared after each mutation.
#   bash docs/card7_evidence/snapshot_submit.sh mut docs/card7_evidence/mutation_job.sh
#SBATCH --job-name=c7-mut
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/mutation_%j.txt
set -uo pipefail
cd "${SNAP:?run through snapshot_submit.sh}/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 MPLBACKEND=Agg
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "snapshot=${SNAP} $(cat ${SNAP}/SNAPSHOT.txt)"
echo "git=$(git -C /mnt/data/peeradon/h2snn-model rev-parse HEAD)"
mutate() {  # $1 label, $2 file, $3 old, $4 new, $5 test id
    cp "$2" /tmp/c7mut.orig.$$
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
    cp /tmp/c7mut.orig.$$ "$2" && cmp "$2" /tmp/c7mut.orig.$$ && echo "restored: identical"
}
# 1 (T3): raising h adds same-class edges (the partner is drawn from the node's own class)
if [ -f scripts/perturbation/make_graphs.py ]; then
mutate "1: raising h adds same-class edges" scripts/perturbation/make_graphs.py \
    "partner_class = other_classes[int(rng.integers(len(other_classes)))]" \
    "partner_class = own_class" \
    tests.test_card7_perturbation.TargetedTests.test_raising_adds_only_typed_new_different_class_facts
fi
# 2 (T2): Holm over all claims instead of per claim
mutate "2: Holm over all claims" scripts/ablation/stats.py \
    "    adjusted = holm({d: (r[\"p_raw\"] if r else None) for d, r in per_dataset.items()})" \
    "    adjusted = {d: v for (c, d), v in holm({(c.name, d): (paired(table, d, (\"full\", \"a\"), (c.row, configuration)) or {}).get(\"p_raw\") for c in CLAIMS for d in DATASETS}).items() if c == claim.name}" \
    tests.test_card7_stats.WilcoxonAndHolmTests.test_holm_is_per_claim_over_four_datasets
echo "end=$(date -Is)"
