"""Card 22 C evidence: peak host RSS of one SeHGNN IEEE-CIS-G H2GB NC trial (`SeHGNNAdapter._fit_h2gb`, CPU), the
plan cut to 1 epoch (propagation and the input tensors, the memory peak, are built in full). Run once with
H2SNN_LOADER_WORKERS=6 and once with 0. Usage (from H2S-NN/): python ../docs/card22_evidence/c_sehgnn_memory.py
'<params json>'
"""
import dataclasses
import json
import os
import resource
import sys
import time

import torch

sys.path.insert(0, ".")
import src.baselines.minibatch as mb  # noqa: E402
from src.baselines.adapters.sehgnn import ADAPTER  # noqa: E402
from src.baselines.bundles import build_bundle  # noqa: E402
from src.baselines.protocol import BaselineParams  # noqa: E402

values = json.loads(sys.argv[1])
print(f"H2SNN_LOADER_WORKERS={os.environ.get('H2SNN_LOADER_WORKERS')} LOADER_WORKERS={mb.LOADER_WORKERS} "
      f"params={values}", flush=True)
real_plan = mb.minibatch_plan
mb.minibatch_plan = lambda dataset, batch_size: dataclasses.replace(real_plan(dataset, batch_size), epochs=1)
t = time.time()
bundle = build_bundle("node_classification", "ieee-cis-g", run_index=0, seed=0,
                      overrides={"data": {"batch_size": int(values.get("batch_size", 256))}})
record = ADAPTER._fit_h2gb(bundle, BaselineParams(values=values), seed=0, device=torch.device("cpu"))
print(f"seconds={time.time() - t:.0f} num_metapaths={record.get('num_metapaths')}", flush=True)
for who in ("SELF", "CHILDREN"):
    usage = resource.getrusage(getattr(resource, f"RUSAGE_{who}"))
    print(f"maxrss_{who}_GB={usage.ru_maxrss / 2**20:.1f}")
