"""Card 1f U1: B1a subset fingerprints from the real data/NC splits (CPU job; no model, no metric).

Run from H2S-NN/: python ../docs/b1a_evidence/u1_fingerprints.py ../docs/b1a_evidence/subset_fingerprints.json
"""
import gc
import json
import sys

from src.data.h2gb_nc import ALL_H2GB_NC_SPECS, RAW_ONLY, load_h2gb_nc
from src.evaluation.validation_subset import REFUSED, SEED, SUBSET_SIZES, subset_fingerprint, validation_subset

out: dict = {"seed": SEED, "datasets": {}}
for dataset in SUBSET_SIZES:
    if dataset in REFUSED:
        out["datasets"][dataset] = {"status": "refused", "reason": REFUSED[dataset]}
        print(dataset, "refused", flush=True)
        continue
    assert dataset not in RAW_ONLY
    data = load_h2gb_nc(dataset, "data")
    target = ALL_H2GB_NC_SPECS[dataset].target_node_type
    val_ids = data[target].val_mask.nonzero().view(-1)
    subset = validation_subset(dataset, val_ids)
    again = validation_subset(dataset, val_ids.flip(0))  # input order does not matter
    assert bool((subset == again).all())
    entry = {"target_type": target, "split_size": int(val_ids.numel()), "subset_size": int(subset.numel()),
             "split_fingerprint": subset_fingerprint(val_ids.sort().values),
             "subset_fingerprint": subset_fingerprint(subset)}
    out["datasets"][dataset] = entry
    print(dataset, entry, flush=True)
    del data
    gc.collect()

with open(sys.argv[1], "w") as handle:
    json.dump(out, handle, indent=2)
    handle.write("\n")
