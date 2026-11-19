#!/bin/bash
# Card 7 Phase B smoke on real ACM at a tiny budget (no metric is reported): `full`, `no_path_cells` and S2
# (configuration (a) from the experiment file as a placeholder for `full`'s tuned file), seed 1, one epoch
# (`--dry_run`), CPU; then face_effect.py (F1, F2, F3 on 20 test nodes, the case-study rule) and two_regime.py (R1-R3)
# on those three runs. Checks that the analysis runs end to end at real size (eta_v on the real graph, chunked R1).
# --mem: no recorded peak for these scripts; the Phase B manifest job (real ACM graph analysis) peaked at 4.1 GB.
#SBATCH --job-name=c7-smokeB
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/phaseb_smoke_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 MPLBACKEND=Agg
E=/mnt/data/peeradon/h2snn-model/docs/card7_evidence
W=/mnt/data/peeradon/h2snn-model-snap/card7_smokeB; rm -rf $W; mkdir -p $W
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=$(git status --short -- src scripts tests | wc -l)"
F='GBWarning|FutureWarning|warnings.warn|UserWarning|DeprecationWarning|import torch_geometric|Processing|Done!'
python scripts/ablation/arms.py --full_config configs/experiments/acm_h2snn_v3b.yaml --dataset acm --out_dir $W/arms 2>&1 | grep -vE "$F" | tail -n 3
for row in full no_path_cells S2; do
  t0=$(date +%s)
  python scripts/run_h2snn_runs.py --config $W/arms/acm/$row.yaml --base_seed 1 --num_runs 1 --exp_name smoke_$row \
      --output_dir $W/runs --dry_run --device cpu > $W/train_$row.log 2>&1
  echo "train $row: exit=$? $(( $(date +%s) - t0 )) s; $(ls $W/runs/smoke_${row}_run0/checkpoints 2>/dev/null | tr '\n' ' ')"
done
RUNS="$W/runs/smoke_full_run0 $W/runs/smoke_no_path_cells_run0 $W/runs/smoke_S2_run0"
t0=$(date +%s)
python scripts/analysis/face_effect.py --runs $RUNS --out_dir $W/face --f3_seeds 1 --f3_max_nodes 20 2>&1 | grep -vE "$F" | tail -n 20
echo "face_effect: exit=${PIPESTATUS[0]} $(( $(date +%s) - t0 )) s"
t0=$(date +%s)
python scripts/analysis/two_regime.py --runs $RUNS --out_dir $W/regime 2>&1 | grep -vE "$F" | tail -n 20
echo "two_regime: exit=${PIPESTATUS[0]} $(( $(date +%s) - t0 )) s"
python - "$W" <<'EOF'
import json, sys
from pathlib import Path
w = Path(sys.argv[1])
f = json.loads((w / "face" / "face_effect.json").read_text())
d = f["datasets"]["acm"]
print("eta quartile edges", d["eta_quartile_edges"], "test nodes without eta", d["test_nodes_without_eta"])
s = d["seeds"][0]
print("F1 nodes per bin", {k: v["nodes"] for k, v in s["F1"].items()}, "F2 arms", sorted(s["F2"]), "F3 nodes", s["F3"]["nodes"])
case = f.get("case_study", {})
print("case study:", case.get("status"), {k: case.get(k) for k in ("node", "candidates")}, "cells", len(case.get("cells", [])))
for p in sorted((w / "regime" / "runs").glob("*.json")):
    r = json.loads(p.read_text())
    r1 = r.get("R1", {}).get("cells") or {}
    print(p.name, "tau", r["tau"], "R1 cells", r1.get("same"), r1.get("different"), "R1 facts",
          (r.get("R1", {}).get("facts") or {}).get("facts"), "R2 nodes", (r.get("R2") or {}).get("nodes"),
          "R3 keys", sorted(r.get("R3", {}))[:4])
EOF
echo "end=$(date -Is)"
