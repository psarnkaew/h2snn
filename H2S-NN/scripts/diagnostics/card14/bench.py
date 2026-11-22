"""Card 14 speed table: 1 warm-up step + 20 training steps + one evaluation pass, old vs new code.

The same script times either tree: `--root` is the H2S-NN directory whose `src/` is imported (the
card14 worktree or the base worktree at 66e9349). It loads a saved trial config, applies `--set`
overrides, builds the components as the sweep does and runs the trainer's own `fit`, with timing
wrappers on the trainer methods (no `src/` change):

* mini-batch NC (mag-year, PDNS): `epoch_targets = 21 * B`, one epoch; per-step wall from the
  optimizer's `step` (CUDA-synchronised); the evaluation pass = the first `evaluate_split` (the B1a
  validation subset, which includes the run's one evaluation draw, timed apart), then stop;
* GC (NCI1): the training loader cut to 21 graphs (one per step); the evaluation pass = the whole
  validation loader, then stop;
* full-batch NC (ACM): 21 epochs of one step; step = `training_draw` + `_train_epoch`; the
  evaluation pass = one `_eval_epoch` (validation).

GPU utilisation: `nvidia-smi` sampled every 200 ms over the 20 timed steps. Prints one `BENCH` JSON
line.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import resource
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import yaml


class _Done(Exception):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, default=Path("/mnt/data/peeradon/card14_runs/bench"))
    p.add_argument("--set", action="append", default=[], dest="sets")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--profile", action="store_true",
                   help="run under torch.profiler (CPU + CUDA) and print the top functions")
    return p.parse_args()


class GpuSampler:
    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
             "-lms", "200"], stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            try:
                self.samples.append((time.time(), float(line.strip().split(",")[0])))
            except ValueError:
                pass

    def mean(self, start: float, end: float) -> float | None:
        vals = [u for t, u in self.samples if start <= t <= end]
        return sum(vals) / len(vals) if vals else None

    def stop(self) -> None:
        self.proc.terminate()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    import torch

    from src.configs.config import TrainConfig
    from src.utils.misc import seed_everything

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for item in args.sets:
        key, _, value = item.partition("=")
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    raw["output_dir"] = str(args.out)
    raw["exp_name"] = args.name
    raw["use_wandb"] = False
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=root).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, cwd=root).stdout.count("\n")
    gpu = torch.cuda.get_device_name(0)
    print(f"bench {args.name}: root={root} head={head} dirty={dirty} host={socket.gethostname()} "
          f"job={os.environ.get('SLURM_JOB_ID')} gpu={gpu}", flush=True)
    cfg = TrainConfig.from_dict(raw)
    steps = int(args.steps)
    task = cfg.task_type
    minibatch = task == "node_classification" and cfg.data.training_mode == "minibatch"
    if minibatch:
        cfg = replace(cfg, data=replace(cfg.data, epoch_targets=(steps + 1) * cfg.data.batch_size),
                      epochs=1)
    elif task == "node_classification":
        cfg = replace(cfg, epochs=steps + 1)
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed, cfg.device)

    from src.builders import build_experiment_components

    t0 = time.time()
    data, model, optimizer, _, logger, _, trainer = build_experiment_components(cfg)
    build_s = time.time() - t0
    sampler = GpuSampler()
    sync = torch.cuda.synchronize
    step_end: list[float] = []
    record: dict[str, float] = {}

    original_step = optimizer.step

    def timed_step(*a, **k):
        out = original_step(*a, **k)
        sync()
        step_end.append(time.time())
        return out

    optimizer.step = timed_step  # type: ignore[method-assign]
    fit_start = [0.0]

    if minibatch:
        evaluate_split = trainer.evaluate_split
        evaluation_cells = trainer.evaluation_cells

        def timed_cells(*a, **k):
            s = time.time()
            out = evaluation_cells(*a, **k)
            sync()
            record.setdefault("eval_draw_s", time.time() - s)
            return out

        def timed_eval(*a, **k):
            sync()
            s = time.time()
            evaluate_split(*a, **k)
            sync()
            record["eval_s"] = time.time() - s
            raise _Done

        trainer.evaluation_cells = timed_cells
        trainer.evaluate_split = timed_eval
        fit = lambda: trainer.fit(data)  # noqa: E731
    elif task == "graph_classification":
        data = replace(data, train_loader=list(itertools.islice(iter(data.train_loader), steps + 1)))
        run_epoch = trainer._run_epoch

        def timed_run_epoch(loader, epoch, train):
            if train:
                return run_epoch(loader, epoch, train)
            if args.profile:  # profile the training steps only
                raise _Done
            sync()
            s = time.time()
            run_epoch(loader, epoch, train)
            sync()
            record["eval_s"] = time.time() - s
            raise _Done

        trainer._run_epoch = timed_run_epoch
        fit = lambda: trainer.fit(data)  # noqa: E731
    else:  # full-batch NC
        training_draw, train_epoch, eval_epoch = (trainer.training_draw, trainer._train_epoch,
                                                  trainer._eval_epoch)
        step_s: list[float] = []
        evals = [0]

        def timed_draw(*a, **k):
            sync()
            fit_start[0] = time.time()
            return training_draw(*a, **k)

        def timed_train(*a, **k):
            out = train_epoch(*a, **k)
            sync()
            step_s.append(time.time() - fit_start[0])
            return out

        def timed_eval_epoch(*a, **k):
            if len(step_s) < steps + 1:
                return eval_epoch(*a, **k)
            sync()
            s = time.time()
            out = eval_epoch(*a, **k)
            sync()
            record["eval_s"] = time.time() - s
            evals[0] += 1
            raise _Done

        trainer.training_draw = timed_draw
        trainer._train_epoch = timed_train
        trainer._eval_epoch = timed_eval_epoch
        fit = lambda: trainer.fit(data)  # noqa: E731

    torch.cuda.reset_peak_memory_stats()
    sync()
    start = time.time()
    outcome = "done"
    profiler = None
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           with_stack=False)
        profiler.__enter__()
    try:
        fit()
    except _Done:
        pass
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        outcome = f"{type(exc).__name__}: {exc}"
    sampler.stop()
    if profiler is not None:
        profiler.__exit__(None, None, None)
        averages = profiler.key_averages()
        print(averages.table(sort_by="self_cpu_time_total", row_limit=30), flush=True)
        print(averages.table(sort_by="self_cuda_time_total", row_limit=15), flush=True)
    if task == "node_classification" and not minibatch:
        per_step = step_s[1:]
        window = (start, start + sum(step_s))
    else:
        ends = [start] + step_end
        per_step = [b - a for a, b in zip(ends[1:], ends[2:])]
        window = (step_end[0], step_end[-1]) if len(step_end) > 1 else (start, start)
    result = {
        "name": args.name, "head": head, "dirty": dirty, "gpu": gpu,
        "job": os.environ.get("SLURM_JOB_ID"), "outcome": outcome,
        "build_s": round(build_s, 1),
        "first_step_s": round((step_end[0] - start) if step_end and minibatch else
                              (step_s[0] if task == "node_classification" and not minibatch and step_s
                               else (step_end[0] - start if step_end else float("nan"))), 3),
        "timed_steps": len(per_step),
        "s_per_step": round(sum(per_step) / len(per_step), 4) if per_step else None,
        "s_per_step_median": round(sorted(per_step)[len(per_step) // 2], 4) if per_step else None,
        "gpu_util_pct": sampler.mean(*window),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "host_peak_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
        **{k: round(v, 3) for k, v in record.items()},
    }
    print("BENCH " + json.dumps(result), flush=True)
    try:
        logger.close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
