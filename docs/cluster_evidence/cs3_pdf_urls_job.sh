#!/bin/bash
# CS3: extract repository URLs from the reference PDFs (the methods' papers) with the standard
# library only: raw bytes plus every FlateDecode stream (link annotations /URI and literal text).
set -u
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn || exit 2
echo "job=$SLURM_JOB_ID node=$(hostname)"
python -I - <<'PY'
import re, zlib, pathlib
pat = re.compile(rb"(?:https?://)?(?:www\.)?(?:github\.com|gitee\.com|gitlab\.com|bitbucket\.org)/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)?", re.I)
streams = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.S)
for p in sorted(pathlib.Path("references").rglob("*.pdf")):
    data = p.read_bytes(); chunks = [data]
    for m in streams.finditer(data):
        try: chunks.append(zlib.decompress(m.group(1)))
        except Exception: pass
    found = set()
    for c in chunks:
        for m in pat.finditer(c):
            found.add(m.group(0).decode("latin-1").rstrip(".)"))
    if found: print(f"{p}: " + " ".join(sorted(found)))
PY
