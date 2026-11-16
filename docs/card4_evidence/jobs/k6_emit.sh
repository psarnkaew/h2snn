#!/bin/bash
# Card 4 K6: emit configurations (a) and (b) and the ACM manifest (placeholder full config), validate
# every (b) base config with the tuning launcher (--validate_only, no W&B), then the one-epoch dry run.
set -uo pipefail
cd /mnt/data/peeradon/h2snn-ablprep/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) git=$(git rev-parse HEAD)"
echo "dirty: $(git status --short -- src tests scripts | tr '\n' ' ')"
sha256sum src/configs/config.py scripts/ablation/*.py
E=../docs/card4_evidence/k6
status=0
python scripts/ablation/seed_runs.py --dataset acm --full_config configs/experiments/acm_h2snn_v3b.yaml \
    --tuning_base configs/experiments/acm_h2snn_v3b.yaml --out_dir $E/emitted || status=1
for f in $E/emitted/acm/tune_*.yaml; do
    echo "=== validate_only $f"
    python scripts/run_h2snn_wandb_tuning.py --config "$f" --n_trials 100 --validate_only \
        --state_dir experiments/card4_k6_validate/$(basename "$f" .yaml) > experiments/card4_k6_validate_$(basename "$f" .yaml).txt 2>&1 \
        && echo "ok: $(python -c "import json,sys; t=open(sys.argv[1]).read(); s=t[t.index('{'):]; print(sorted(json.loads(s)['parameter_names']))" experiments/card4_k6_validate_$(basename "$f" .yaml).txt)" \
        || { echo "FAILED"; tail -3 experiments/card4_k6_validate_$(basename "$f" .yaml).txt; status=1; }
done
python scripts/ablation/dry_run.py --full_config configs/experiments/acm_h2snn_v3b.yaml --out $E/dry_run_acm.json || status=1
echo "end=$(date -Is) exit=${status}"
exit $status
