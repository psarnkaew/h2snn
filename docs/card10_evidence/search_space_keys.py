import sys, json, collections
sys.path.insert(0, "/mnt/data/peeradon/h2snn-card10/H2S-NN"); sys.path.insert(0, "/mnt/data/peeradon/h2snn-card10/H2S-NN/scripts")
import run_baseline_tuning as rbt
from src.baselines.campaign import build_sweep_config
keys = collections.defaultdict(set)
for task in sorted(rbt.DATASETS):
    try: cells = rbt.resolve_cells(task, ["all"], ["all"])
    except SystemExit as e: print("skip", task, e); continue
    for c in cells:
        try: p = build_sweep_config(c, monitor="val_x", n_trials=1, epochs=1)["parameters"]
        except Exception as e: print("ERR", c, type(e).__name__, e); continue
        for k, s in p.items(): keys[k].add(json.dumps(s, sort_keys=True))
for k in sorted(keys): print(k, "|", " ; ".join(sorted(keys[k]))[:300])
