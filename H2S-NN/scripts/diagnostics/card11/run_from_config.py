"""Card 11: train one reference trial from its resolved config, as the sweep's `train_trial` does.

`scripts/run_h2snn_wandb_tuning.py::train_trial` (and the GC launcher) resolve the trial config,
save it as `<exp_dir>/config.yaml`, then `seed_everything(cfg.seed)`, `build_experiment_components`
and `trainer.fit(data, epoch_callback=...)`. The configs in `docs/card11_inputs/` are those saved
files, so this script loads one, changes only the keys given with `--set` (plus the bookkeeping keys
`output_dir`, `exp_name`, `use_wandb: false`, which do not reach the model) and runs the same three
calls. No `src/` change.

Options:
  --set a.b=value    a config key the card names (value parsed as YAML); repeatable
  --probe PATH       install the B1 solver probe (`solver_probe.py`), one row per PCG call
  --wall_limit S     stop cleanly after S seconds (diagnosis wall limit, not a model stop)
  --save_epochs L    comma list of epochs whose model state is saved (`init` = before training)

Writes `<out>/<exp_name>/` (config, run.log, metrics.jsonl, checkpoints of the trainer) and
`<out>/<exp_name>/card11_epochs.tsv` (epoch, wall s, peak GPU memory). Prints the outcome
(`finished` / `stop condition ...` / `PCGConvergenceError` / `wall limit`) on the last line.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.configs.config import TrainConfig  # noqa: E402
from src.utils.misc import save_config, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--set", action="append", default=[], dest="sets")
    p.add_argument("--probe", type=Path, default=None)
    p.add_argument("--wall_limit", type=float, default=None)
    p.add_argument("--save_epochs", type=str, default="")
    return p.parse_args()


def apply_set(raw: dict, item: str) -> tuple[str, object, object]:
    key, _, value = item.partition("=")
    parts = key.split(".")
    node = raw
    for part in parts[:-1]:
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"--set {key}: not a key of the reference config")
    old = node[parts[-1]]
    new = yaml.safe_load(value)
    node[parts[-1]] = new
    return key, old, new


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reference = copy.deepcopy(raw)
    changes = [apply_set(raw, item) for item in args.sets]
    raw["output_dir"] = str(args.out)
    raw["exp_name"] = args.exp_name
    raw["use_wandb"] = False
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=PROJECT_ROOT).stdout.strip()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"card11 run: config={args.config} head={head} host={socket.gethostname()} "
          f"job={os.environ.get('SLURM_JOB_ID', 'none')} gpu={gpu}", flush=True)
    for key, old, new in changes:
        print(f"card11 change: {key}: {old!r} -> {new!r}", flush=True)
    diff = [k for k in raw if k not in ("output_dir", "exp_name", "use_wandb")
            and raw[k] != reference.get(k) and not any(c[0].split('.')[0] == k for c in changes)]
    assert not diff, f"unexpected config changes: {diff}"

    cfg = TrainConfig.from_dict(raw)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.exp_dir / "config.yaml")
    (cfg.exp_dir / "card11_changes.json").write_text(json.dumps(
        {"reference": str(args.config), "changes": [[k, o, n] for k, o, n in changes],
         "head": head, "gpu": gpu, "job": os.environ.get("SLURM_JOB_ID")}, indent=2, default=str))
    seed_everything(cfg.seed, cfg.device)

    probe = None
    if args.probe is not None:
        from solver_probe import install
        probe = install(args.probe, args.wall_limit)

    from src.builders import build_experiment_components

    t0 = time.time()
    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    build_s = time.time() - t0
    save_epochs = {s.strip() for s in args.save_epochs.split(",") if s.strip()}
    if "init" in save_epochs:
        torch.save(model.state_dict(), cfg.exp_dir / "card11_state_init.pt")
    epochs_tsv = (cfg.exp_dir / "card11_epochs.tsv").open("w", encoding="utf-8")
    epochs_tsv.write("epoch\twall_s\tpeak_gpu_gib\n")
    fit_t0 = time.time()

    class _Wall(RuntimeError):
        pass

    def callback(epoch, train_metrics, val_metrics, test_metrics) -> None:
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        epochs_tsv.write(f"{epoch}\t{time.time() - fit_t0:.1f}\t{peak:.3f}\n")
        epochs_tsv.flush()
        if str(epoch) in save_epochs:
            torch.save(model.state_dict(), cfg.exp_dir / f"card11_state_epoch{epoch}.pt")
        if args.wall_limit is not None and time.time() - t0 > args.wall_limit:
            raise _Wall(f"wall limit {args.wall_limit:.0f} s after epoch {epoch}")

    outcome = "finished"
    try:
        logger.info("card11: build %.1f s; changes %s", build_s, [c[:3] for c in changes])
        trainer.fit(data, epoch_callback=callback)
    except _Wall as exc:
        outcome = f"wall limit ({exc})"
    except Exception as exc:  # noqa: BLE001 - report the outcome, keep the traceback
        name = type(exc).__name__
        if name == "WallLimitReached":
            outcome = f"wall limit ({exc})"
        else:
            traceback.print_exc()
            outcome = f"{name}: {exc}"
    finally:
        try:
            logger.close()
        except Exception:  # noqa: BLE001
            pass
        epochs_tsv.close()
        if probe is not None:
            probe.close()
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"card11 peak_gpu_gib={peak:.3f} fit_s={time.time() - fit_t0:.1f} build_s={build_s:.1f}",
          flush=True)
    print(f"card11 outcome: {outcome}", flush=True)


if __name__ == "__main__":
    main()
