"""Card 23 fingerprint check: base-config and sweep fingerprints of the shipped H2S-NN configs on a tree.

    python fingerprints.py <tree>      (tree = a worktree root; run with PYTHONDONTWRITEBYTECODE=1)

Prints one JSON line per config: base (the tuning script's base fingerprint path) and sweep (the sweep
config `_build_sweep_for_run` builds, fingerprinted as `run_unit_sweep` does). Run on b0f13fb and card23;
the two outputs must be equal. With `--pack`, also the ACM config with eval_draws 8 / checkpoint_smoothing 5.
"""
import argparse
import json
import os
import sys
from pathlib import Path

tree = Path(sys.argv[1]).resolve() / "H2S-NN"
os.chdir(tree)
sys.path.insert(0, str(tree))

from scripts.run_h2snn_wandb_tuning import _build_sweep_for_run, load_raw_config  # noqa: E402
import scripts.run_h2snn_wandb_tuning as tuning  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402
from src.utils.misc import dataclass_to_dict  # noqa: E402
from src.utils.wandb_campaign import canonical_fingerprint as campaign_fingerprint  # noqa: E402

NAMES = ["acm", "dblp", "imdb", "freebase", "lastfm", "pubmed", "acm_pp", "dblp_ap", "imdb_ma", "freebase_bb",
         "mutag", "nci1", "ogbn-mag", "pdns", "h-pokec", "oag-cs", "ieee-cis-g"]


def line(name, raw):
    cfg = TrainConfig.from_dict(raw)
    base = tuning.canonical_fingerprint(dataclass_to_dict(cfg))  # what run_seed_sweep / the GC script hash
    sweep = _build_sweep_for_run(args=argparse.Namespace(d_values=None, grid_point=None), base_cfg=cfg,
                                 base_exp_name="fp", monitor="val_macro_f1", run_index=0)
    return {"config": name, "base": base, "sweep": campaign_fingerprint(sweep),
            "params": sorted(sweep["parameters"])}


for name in NAMES:
    path = Path(f"configs/experiments/{name}_h2snn_v3b.yaml")
    try:
        print(json.dumps(line(name, load_raw_config(path))), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"config": name, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
if "--pack" in sys.argv:
    raw = load_raw_config(Path("configs/experiments/acm_h2snn_v3b.yaml"))
    raw["data"]["eval_draws"] = 8
    raw["checkpoint_smoothing"] = 5
    out = line("acm+pack", raw)
    cfg = TrainConfig.from_dict(raw)
    sweep = _build_sweep_for_run(args=argparse.Namespace(d_values=None, grid_point=None), base_cfg=cfg,
                                 base_exp_name="fp", monitor="val_macro_f1", run_index=0)
    out["space"] = sweep["parameters"]
    print(json.dumps(out), flush=True)
