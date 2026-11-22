"""Card 18 test (c): one epoch on acm_pp for MAGNN and for H2G-former with metapath_pe on, through the adapter's own
LP fit and the shared loop. Records every training loss (`check_finite_loss`, partn_smoke's Recorder) and whether the
model and head parameters move (snapshot around `train_link_prediction`). Defaults otherwise; no tuning.

    python scripts/diagnostics/card18/smoke_epoch.py --model magnn --out <json>
    python scripts/diagnostics/card18/smoke_epoch.py --model h2gformer --out <json>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

# C18_PROJECT_ROOT: run against another checkout (the pre-card-18 code at card15 2e4881c) with this script unchanged.
PROJECT_ROOT = Path(os.environ.get("C18_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from partn_smoke import Recorder, _moved, _snapshot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["magnn", "h2gformer"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import src.baselines.adapters.h2gformer as h2gformer
    import src.baselines.training as training
    from src.baselines.bundles import build_bundle
    from src.baselines.protocol import LINK_PREDICTION, BaselineParams
    from src.baselines.registry import get_adapter

    rec = Recorder()
    rec.install()
    moved: dict = {}
    original = training.train_link_prediction

    def wrapped(**kwargs):
        modules = [kwargs["model"], kwargs["head"]]
        before = _snapshot(modules)
        try:
            return original(**kwargs)
        finally:
            moved.update(_moved(before, modules))

    training.train_link_prediction = wrapped      # MAGNN imports it inside its fit
    h2gformer.train_link_prediction = wrapped     # H2G-former imports it at module load
    values = {"epochs": 1, "early_stopping": 1}
    if args.model == "h2gformer":
        values["metapath_pe"] = True
    device = torch.device(args.device)
    bundle = build_bundle(LINK_PREDICTION, "acm_pp", overrides={"device": args.device})
    start, error, record = time.perf_counter(), None, {}
    try:
        record = get_adapter(args.model).fit_task(bundle, BaselineParams(values), task=LINK_PREDICTION, seed=0,
                                                  device=device)
    except Exception as exc:  # recorded, exit 1
        error = f"{type(exc).__name__}: {exc}"[:2000]
    losses = [v for _, v in rec.train_losses]
    result = {"model": args.model, "dataset": "acm_pp", "params": values, "error": error,
              "seconds": round(time.perf_counter() - start, 1),
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "train_losses": losses, "losses_finite": bool(losses) and all(math.isfinite(v) for v in losses),
              **moved, "metapath_pe_in_record": record.get("metapath_pe"),
              "note": "card 18 smoke: function only; no metric is a result"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("model", "error", "losses_finite", "parameters_moved",
                                              "parameters_total", "seconds")}, indent=1))
    ok = error is None and result["losses_finite"] and (result.get("parameters_moved") or 0) > 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
