"""Card 11 B1 check: the solver probe does not change the computation.

One ACM full-batch epoch (`configs/experiments/acm_h2snn_v3b.yaml`, `epochs: 1`), three times in one
process, each from `seed_everything(cfg.seed)`: plain, plain again, with the probe installed. Compares
the epoch's train and validation loss (float bits) and every parameter after the epoch
(`torch.equal`). plain-vs-plain says whether the device is deterministic at all; plain-vs-probe is
the check. Usage: `b1_identity_check.py --device cuda|cpu --out DIR`.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.configs.config import TrainConfig  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402


def bits(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


def one_run(device: str, out: Path, name: str, probe_path: Path | None):
    raw = yaml.safe_load((PROJECT_ROOT / "configs/experiments/acm_h2snn_v3b.yaml").read_text())
    raw["epochs"] = 1
    raw["device"] = device
    raw["output_dir"] = str(out)
    raw["exp_name"] = name
    raw["use_wandb"] = False
    cfg = TrainConfig.from_dict(raw)
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed, cfg.device)
    probe = None
    if probe_path is not None:
        from solver_probe import install
        probe = install(probe_path)
    from src.builders import build_experiment_components

    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    seen = {}

    def cb(epoch, tr, va, te):
        seen["train_loss"], seen["val_loss"] = tr["loss"], va["loss"]

    try:
        trainer.fit(data, epoch_callback=cb)
    finally:
        logger.close()
        if probe is not None:
            rows = probe.rows_written
            probe.close()
            seen["probe_rows"] = rows
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return seen, state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=1,
                    help="plain/probe pairs (GPU: how often the train loss differs run to run)")
    args = ap.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)  # with the pinned kernel environment E (job --export)
    print(f"threads={torch.get_num_threads()} cpu_capability={torch.backends.cpu.get_cpu_capability()}",
          flush=True)
    runs = {}
    plan = [("plain_a", None), ("plain_b", None),
            ("probe", args.out / f"b1_identity_{args.device}_probe.tsv")]
    for r in range(1, args.repeats):
        plan += [(f"plain_r{r}", None), (f"probe_r{r}", args.out / f"b1_identity_{args.device}_probe_r{r}.tsv")]
    for name, probe in plan:
        runs[name] = one_run(args.device, args.out, f"b1_identity_{args.device}_{name}", probe)
        s = runs[name][0]
        print(f"{name}: train_loss={s['train_loss']!r} ({bits(s['train_loss'])}) "
              f"val_loss={s['val_loss']!r} ({bits(s['val_loss'])}) probe_rows={s.get('probe_rows', 0)}",
              flush=True)
    plain = sorted({bits(runs[n][0]["train_loss"]) for n in runs if n.startswith("plain")})
    probe = sorted({bits(runs[n][0]["train_loss"]) for n in runs if n.startswith("probe")})
    print(f"distinct train-loss bit patterns: plain {plain} probe {probe}", flush=True)

    def same(a, b):
        sa, pa = runs[a]
        sb, pb = runs[b]
        loss = bits(sa["train_loss"]) == bits(sb["train_loss"]) and bits(sa["val_loss"]) == bits(sb["val_loss"])
        params = all(torch.equal(pa[k], pb[k]) for k in pa)
        return loss, params

    for a, b in (("plain_a", "plain_b"), ("plain_a", "probe")):
        loss, params = same(a, b)
        print(f"compare {a} vs {b} on {args.device}: losses bit-identical={loss} "
              f"parameters equal={params}", flush=True)


if __name__ == "__main__":
    main()
