"""Card 17 smoke: one published IEEE-CIS-G cell (GCN or H2G-former), seed 0, a few steps, on one GPU.

    python ../docs/card17_evidence/smoke_ieee.py --model gcn --steps 4 --eval-cap 512 --out <json>

The cell runs through the adapter's published path unchanged (published file, H2GB's cfg/model/optimizer/scheduler,
`h2gb_criterion`), with three smoke-only cuts: `published_plan` is called as is (its 256 x 32 epoch check runs) and its
result shortened to 1 epoch of `--steps` steps (eval period 1); the target's val/test masks keep their first
`--eval-cap` targets (the whole-split scoring then covers those). Checks: the model builds (featureless types), every
training loss is finite, the criterion is the weighted one, parameters move, the record has Macro-/Micro-F1. Peaks:
host max RSS and CUDA max allocated.
"""
import argparse
import dataclasses
import json
import math
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")


def cap_masks(data, target: str, cap: int) -> None:
    import torch

    for split in ("val", "test"):
        mask = data[target][f"{split}_mask"]
        keep = mask.nonzero().view(-1)[:cap]
        capped = torch.zeros_like(mask)
        capped[keep] = True
        data[target][f"{split}_mask"] = capped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("gcn", "h2gformer"))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--eval-cap", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    import src.baselines.h2gb_published as hp
    import src.baselines.minibatch as mb
    import src.data.h2gb_nc as nc
    from src.baselines.adapters._h2gb_network import DatasetShim
    from src.baselines.bundles import build_bundle
    from src.baselines.campaign import DEFAULT_MINIBATCH_BATCH_SIZE
    from src.baselines.protocol import NODE_CLASSIFICATION
    from src.baselines.published import load_published_params
    from src.baselines.registry import get_adapter

    dataset, target = "ieee-cis-g", nc.ALL_H2GB_NC_SPECS["ieee-cis-g"].target_node_type
    device = torch.device("cuda")
    info: dict = {"model": args.model, "dataset": dataset, "steps": args.steps, "eval_cap": args.eval_cap}

    original_plan = hp.published_plan

    def short_plan(ds, settings):
        plan = original_plan(ds, settings)  # the yaml's 256 x 32 = one epoch check runs unchanged
        info["yaml_plan"] = {"batch_size": plan.batch_size, "steps_per_epoch": plan.steps_per_epoch,
                             "epochs": plan.epochs, "eval_period": plan.eval_period}
        return dataclasses.replace(plan, epochs=1, steps_per_epoch=args.steps, eval_period=1)

    hp.published_plan = short_plan
    original_graph, original_load = mb.h2gb_graph, nc.load_h2gb_dataset

    def graph(bundle):  # GCN path
        data = original_graph(bundle)
        cap_masks(data, target, args.eval_cap)
        return data

    def load(key, root, seed=None):  # H2G-former path
        data = original_load(key, root, seed)[0]
        cap_masks(data, target, args.eval_cap)
        return DatasetShim(data, key)

    mb.h2gb_graph, nc.load_h2gb_dataset = graph, load
    original_train = mb.train_minibatch_node_classification

    def train(**kw):
        model, criterion = kw["model"], kw["criterion"]
        info["criterion"] = getattr(criterion, "__name__", repr(criterion))
        info["featureless_types"] = sorted(t for t, d in __import__("H2GB.graphgym.config", fromlist=["cfg"])
                                           .cfg.share["dim_in"].items() if d is None)
        info["num_parameters"] = sum(p.numel() for p in model.parameters())
        before = [p.detach().clone() for p in model.parameters()]
        losses = []

        def logged(logits, y):
            loss = criterion(logits, y)
            losses.append(float(loss))
            return loss

        record = original_train(**{**kw, "criterion": logged})
        info["losses"] = losses
        info["params_moved"] = sum(int(not torch.equal(b, p.detach())) for b, p in zip(before, model.parameters()))
        return record

    mb.train_minibatch_node_classification = train

    params = load_published_params(args.model, NODE_CLASSIFICATION, dataset)
    bundle = build_bundle(NODE_CLASSIFICATION, dataset, overrides={
        "device": "cuda", "data": {"batch_size": int(DEFAULT_MINIBATCH_BATCH_SIZE)}})
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    record = get_adapter(args.model).fit_task(bundle, params, task=NODE_CLASSIFICATION, seed=0, device=device)
    info["seconds"] = round(time.time() - started, 1)
    info["record"] = {k: v for k, v in record.items() if isinstance(v, (int, float, str))}
    info["cuda_max_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    info["host_max_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3)
    keys = {k.lower() for k in info["record"]}
    checks = {
        "finite_loss": bool(info["losses"]) and all(math.isfinite(x) for x in info["losses"]),
        "weighted_criterion": info["criterion"] == "weighted_cross_entropy",
        "params_moved": info["params_moved"] > 0,
        "macro_micro_f1": any("macro" in k for k in keys) and any("micro" in k for k in keys),
    }
    info["checks"] = checks
    Path(args.out).write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: info[k] for k in ("model", "criterion", "featureless_types", "params_moved", "seconds",
                                           "cuda_max_allocated_gib", "host_max_rss_gib", "checks")}, indent=1))
    print("losses", info["losses"])
    print("record", {k: v for k, v in info["record"].items() if "f1" in k.lower() or "acc" in k.lower()})
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
