#!/bin/bash
# Card 15 step 1: build acm_pp (ACM paper-cite-paper, rel 0), apply F17, write the HGB test negatives.
# Writes into the shared data root (the worktree's data/ is a symlink to it); acm_pa and the other tasks are not touched.
#SBATCH --job-name=c15-build
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --exclude=g2tn01,g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-card15/docs/card15_evidence/slurm-%j-build.log
set -euo pipefail
cd /mnt/data/peeradon/h2snn-card15/H2S-NN
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
DATA=/mnt/data/peeradon/h2snn/H2S-NN/data
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=$(git status --porcelain -- src scripts | wc -l)"
python scripts/lp/build_derived_lp.py --data-root "$DATA" --out-root "$DATA/LP/lp_derived" --tasks acm_pp
python scripts/lp/apply_feature_rule.py --lp-root "$DATA/LP/lp_derived" --tasks acm_pp
python scripts/lp/write_hgb_negatives.py --data-root "$DATA" --tasks acm_pp
python - "$DATA/LP/lp_derived/acm_pp" <<'PY'
import sys
from pathlib import Path
d = Path(sys.argv[1])
def pairs(name, rel=None):
    out = []
    for line in (d / name).read_text().splitlines():
        s, t, r, _ = line.split("\t")
        if rel is None or int(r) == rel:
            out.append((int(s), int(t)))
    return out
test, valid = pairs("link.dat.test"), pairs("link.dat.valid")
link0, link1 = pairs("link.dat", 0), pairs("link.dat", 1)
print("self-loops: test", sum(u == v for u, v in test), "valid", sum(u == v for u, v in valid),
      "link.dat rel0", sum(u == v for u, v in link0), "rel1", sum(u == v for u, v in link1))
print("rows: test", len(test), "valid", len(valid), "link.dat rel0", len(link0), "rel1", len(link1))
print("leak check (test in link.dat either way):", sum((u, v) in set(link0) or (v, u) in set(link1) for u, v in test))
PY
cat "$DATA/LP/lp_derived/acm_pp/feature_rule.json"
echo "end=$(date -Is)"
