#!/bin/bash
# CS4: compare H2S-NN/data/ with the laptop manifest (path and size). A difference stops CS4.
set -u
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn || exit 2
echo "job=$SLURM_JOB_ID node=$(hostname) date=$(date -Is)"
python -I - <<'PY'
import csv, os
from pathlib import Path
want = {r["path"]: int(r["bytes"]) for r in csv.DictReader(open("../docs/cluster_evidence/laptop_data_manifest.tsv"), delimiter="\t")}
have = {}
for root, _, files in os.walk("data"):
    for f in files:
        p = Path(root, f).as_posix(); have[p] = os.path.getsize(p)
missing = sorted(set(want) - set(have)); extra = sorted(set(have) - set(want))
size = sorted(p for p in set(want) & set(have) if want[p] != have[p])
print(f"manifest: {len(want)} files, {sum(want.values())} bytes; server: {len(have)} files, {sum(have.values())} bytes")
print(f"missing {len(missing)}, extra {len(extra)}, size-different {len(size)}, equal {len(set(want)&set(have))-len(size)}")
for p in missing: print("MISSING ", p, want[p])
for p in extra: print("EXTRA   ", p, have[p])
for p in size: print("SIZE    ", p, "laptop", want[p], "server", have[p])
PY
