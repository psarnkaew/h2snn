"""Card 11 A3.3: GCN on acm_pa through the shared baseline LP loop at its central search point.

The fit is the registry's own: `src.baselines.campaign.fit_cell` (what a sweep trial and the
published runner call) on the cell (gcn, link_prediction, acm_pa), seed 0, run index 0, the protocol
budget (no `--epochs` / `--patience` override). The parameter point is the centre of the adapter's
own `search_space("link_prediction", "acm_pa")`: a `values` list -> its element at index len // 2
(sorted as listed), a `log_uniform_values` range -> the geometric midpoint, a `uniform` range -> the
arithmetic midpoint. `--point lower` takes index (len - 1) // 2 instead (differs only for the
two-valued keys `num_layers` and `decoder_type`).

Per epoch, validation metrics come from the loop's own `report`; the training AUROC (training
positives vs that epoch's negatives, training mode, before the step) is read by a run-time wrapper of
`SharedLinkHead.forward` that records the scores of the training-mode calls (no `src` change).
Writes `<out>/a3_gcn_<point>_epochs.tsv` and prints the record.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.misc import compute_link_prediction_metrics  # noqa: E402


def centre(space: dict, which: str) -> dict:
    point = {}
    for key, spec in space.items():
        if "values" in spec:
            vals = list(spec["values"])
            point[key] = vals[len(vals) // 2 if which == "central" else (len(vals) - 1) // 2]
        elif spec.get("distribution") == "log_uniform_values":
            point[key] = math.sqrt(float(spec["min"]) * float(spec["max"]))
        elif spec.get("distribution") == "uniform":
            point[key] = (float(spec["min"]) + float(spec["max"])) / 2
        else:
            raise ValueError(f"unknown search-space entry {key}: {spec}")
    return point


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--point", choices=["central", "lower"], default="central")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from src.baselines import heads as heads_module
    from src.baselines.campaign import fit_cell
    from src.baselines.registry import get_adapter

    space = get_adapter("gcn").search_space("link_prediction", "acm_pa")
    point = centre(space, args.point)
    print(f"search space: {json.dumps(space, default=str)}", flush=True)
    print(f"point ({args.point}): {point}", flush=True)

    captured: list[torch.Tensor] = []
    original = heads_module.SharedLinkHead.forward

    def forward(self, *a, **k):
        out = original(self, *a, **k)
        if self.training:
            captured.append(out.detach().float().cpu())
        return out

    heads_module.SharedLinkHead.forward = forward
    rows = []

    def report(*, epoch, monitor, running_best, metrics):
        train_auroc = float("nan")
        if len(captured) >= 2:
            pos, neg = captured[-2], captured[-1]
            train_auroc = float(compute_link_prediction_metrics(pos, neg)["auroc"])
        captured.clear()
        rows.append({"epoch": epoch, "train_auroc": f"{train_auroc:.4f}",
                     "val_auroc": f"{float(metrics.get('auroc', float('nan'))):.4f}",
                     "val_loss": f"{float(metrics.get('loss', float('nan'))):.4f}",
                     "running_best_val_auroc": f"{running_best:.4f}"})
        print(rows[-1], flush=True)

    raw_base = {"model": "gcn", "task": "link_prediction", "dataset": "acm_pa", "seed": 0,
                "run_index": 0}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    record = fit_cell(raw_base, point, source="tuned", device=device, report=report)
    heads_module.SharedLinkHead.forward = original
    path = args.out / f"a3_gcn_{args.point}_epochs.tsv"
    with path.open("w", encoding="utf-8") as h:
        h.write("\t".join(rows[0]) + "\n")
        for r in rows:
            h.write("\t".join(str(v) for v in r.values()) + "\n")
    (args.out / f"a3_gcn_{args.point}_record.json").write_text(
        json.dumps({"point": point, "record": record}, indent=2, default=str))
    print(json.dumps({"point": point, "record": record}, default=str), flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"card11 peak_gpu_gib={peak:.3f}", flush=True)


if __name__ == "__main__":
    main()
