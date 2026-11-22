#!/bin/bash
# Card 15 option (b), 2026-11-22: acm_pp split by unordered pair. Rebuild acm_pp (+ F17, HGB negatives); rebuild the
# other derived tasks into job scratch and compare md5 with the files on disk (must be unchanged); rerun the degree
# check, the heterophily values, the two smoke epochs; then the targeted test modules. One GPU.
#SBATCH --job-name=c15-bypair
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-card15/docs/card15_evidence/slurm-%j-bypair.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card15/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b TQDM_MININTERVAL=300 WANDB_MODE=disabled
DATA=/mnt/data/peeradon/h2snn/H2S-NN/data
LP=$DATA/LP/lp_derived
EV=../docs/card15_evidence
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=$(git status --porcelain -- src scripts tests | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== 1 build acm_pp by unordered pair"
python scripts/lp/build_derived_lp.py --data-root "$DATA" --out-root "$LP" --tasks acm_pp || exit 3
python scripts/lp/apply_feature_rule.py --lp-root "$LP" --tasks acm_pp || exit 3
python scripts/lp/write_hgb_negatives.py --data-root "$DATA" --tasks acm_pp || exit 3
python - "$LP/acm_pp" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts.lp import build_derived_lp as bd
d = Path(sys.argv[1]); meta = json.loads((d / "split_meta.json").read_text())
print({k: meta[k] for k in ("target_relation", "reverse_relation", "split_unit", "facts", "train", "valid", "test",
                            "rows", "link_rows_removed")})
s = bd.read_split(d, 0)
print("rows: train", len(s.train), "valid", len(s.valid), "test", len(s.test),
      "self-loops in train set", sum(u == v for u, v in s.train), "in valid/test", sum(u == v for u, v in s.valid | s.test))
print("leaks (test in link.dat, rel 0 either way or rel 1 reversed):",
      bd.test_fact_leaks(d, 0, 1), sum((v, u) in s.link_pairs_by_relation[0] for u, v in s.test))
PY

echo "== 2 other derived tasks unchanged (rebuilt into scratch, md5 vs disk)"
SCR=${TMPDIR:-/tmp}/c15_$SLURM_JOB_ID; mkdir -p "$SCR"
python scripts/lp/build_derived_lp.py --data-root "$DATA" --out-root "$SCR" --tasks acm_pa dblp_ap imdb_ma freebase_bb > /dev/null || exit 3
for t in acm_pa dblp_ap imdb_ma freebase_bb; do
  for f in info.dat link.dat link.dat.test link.dat.valid; do
    a=$(md5sum < "$SCR/$t/$f" | cut -c1-32); b=$(md5sum < "$LP/$t/$f" | cut -c1-32)
    echo "$t $f scratch=$a disk=$b $([ "$a" = "$b" ] && echo SAME || echo DIFFERENT)"
  done
done
rm -rf "$SCR"

echo "== 3 degree check (card 12 L scripts, unchanged)"
OUT=$EV/L
python scripts/diagnostics/card15/a1_data_checks.py --out "$OUT" \
  --task acm_pp=configs/experiments/acm_pp_h2snn_v3b.yaml \
  --task dblp_ap=../docs/card11_inputs/h2snn_link_prediction_dblp_ap_tune_run0_daily-sweep-73/config.yaml > "$EV/a1_bypair.log" 2>&1 || echo "a1 exit $?"
python scripts/diagnostics/card15/l_flags.py --in "$OUT" --out "$OUT"

echo "== 4 heterophily"
for m in task_h_edge mde; do
  python scripts/heterophily_stats.py --dataset acm_pp --task lp --measure "$m" --deadline-h 0.4 2>&1 | grep -v Warn | grep -v "C = torch" || echo "measure $m exit"
done
python scripts/heterophily_stats.py --collect > /dev/null

echo "== 5 smoke epochs"
for m in h2snn gcn; do
  python scripts/diagnostics/card15/smoke_one_epoch.py --run "$m:lp:acm_pp" --device cuda --out "$EV/smoke/${m}_acm_pp.json" 2>&1 | grep -v -i warn | tail -16
  echo "smoke $m exit=${PIPESTATUS[0]}"
done

echo "== 6 targeted tests"
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
python -m unittest tests.test_lp_derived tests.test_hgb_oracle tests.test_lp_valid_files tests.test_heart_negatives \
  tests.test_data_layout tests.test_partP_derived_lp 2>&1 | grep -E "^(FAIL|ERROR):|^Ran|^OK|^FAILED|Error:" 
echo "end=$(date -Is)"
