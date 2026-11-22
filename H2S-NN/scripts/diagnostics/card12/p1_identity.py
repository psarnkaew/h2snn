"""Card 12 P1: the zero right-hand-side rule is bit-identical when no scope triggers.

One ACM full-batch epoch (`configs/experiments/acm_h2snn_v3b.yaml`, `epochs: 1`, CPU, 1 thread; run
inside the pinned kernel environment E: `CARD12_PINNED=1`), each in its own process from
`seed_everything(cfg.seed)`:

- `new_a`, `new_b`: this worktree (card12: 2d822a4 + the rule);
- `old`: a temporary copy of this worktree's `src/` whose `src/models/anchored_solve.py` is replaced by
  the file at 2d822a4 (`git show`); `git diff --stat 2d822a4 -- H2S-NN/src` lists only that file, so
  the copy is `src/` at 2d822a4.

Compares the epoch's train and validation loss (float bits) and every parameter after the epoch
(`torch.equal`). In the new runs, every call of `count_zero_rhs_scopes` (one per forward and per
backward solve) is wrapped to sum the triggered scopes (`pcg_zero_rhs_scopes`); must be 0.
Usage: `p1_identity.py --out DIR`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # H2S-NN/
BASE = "2d822a4eb48867de4476b97159430a64d8468808"
TARGET = "src/models/anchored_solve.py"


def bits(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


def worker(root: Path, out: Path, name: str, result: Path) -> None:
    os.chdir(root)
    sys.path.insert(0, str(root))
    import torch
    import yaml

    torch.set_num_threads(1)
    from src.configs.config import TrainConfig
    from src.utils.misc import seed_everything
    import src.models.anchored_solve as AS

    print(f"[{name}] root={root} anchored_solve={AS.__file__} "
          f"md5={hashlib.md5(Path(AS.__file__).read_bytes()).hexdigest()} "
          f"threads={torch.get_num_threads()} cpu_capability={torch.backends.cpu.get_cpu_capability()}",
          flush=True)
    counter = {"calls": 0, "triggered": 0}
    if hasattr(AS, "count_zero_rhs_scopes"):
        original = AS.count_zero_rhs_scopes

        def counting(*args, **kwargs):
            n = original(*args, **kwargs)
            counter["calls"] += 1
            counter["triggered"] += n
            return n
        AS.count_zero_rhs_scopes = counting

    raw = yaml.safe_load((root / "configs/experiments/acm_h2snn_v3b.yaml").read_text())
    raw.update(epochs=1, device="cpu", output_dir=str(out), exp_name=f"p1_{name}", use_wandb=False)
    cfg = TrainConfig.from_dict(raw)
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed, cfg.device)
    from src.builders import build_experiment_components

    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    seen: dict = {}

    def cb(epoch, tr, va, te):
        seen["train_loss"], seen["val_loss"] = tr["loss"], va["loss"]
        seen["train_stats"] = {k: v for k, v in tr.items() if k.startswith("pcg_")}

    try:
        trainer.fit(data, epoch_callback=cb)
    finally:
        logger.close()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"seen": seen, "state": state, "counter": counter}, result)
    print(f"[{name}] train_loss={seen['train_loss']!r} ({bits(seen['train_loss'])}) "
          f"val_loss={seen['val_loss']!r} ({bits(seen['val_loss'])}) "
          f"zero_rhs_calls={counter['calls']} zero_rhs_triggered={counter['triggered']} "
          f"logged={seen.get('train_stats')}", flush=True)


def old_tree(tmp: Path) -> Path:
    tree = tmp / "old_tree"
    tree.mkdir()
    shutil.copytree(ROOT / "src", tree / "src", symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    old = subprocess.run(["git", "show", f"{BASE}:H2S-NN/{TARGET}"], cwd=ROOT, check=True,
                         capture_output=True).stdout
    (tree / TARGET).write_bytes(old)
    for link in ("configs", "data", "vendor"):
        os.symlink(ROOT / link, tree / link)
    return tree


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--worker", nargs=3, metavar=("ROOT", "NAME", "RESULT"))
    args = ap.parse_args()
    if args.worker:
        worker(Path(args.worker[0]), args.out, args.worker[1], Path(args.worker[2]))
        return 0
    import torch

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix="c12p1_") as tmp_s:
        tmp = Path(tmp_s)
        trees = {"new_a": ROOT, "old": old_tree(tmp), "new_b": ROOT}
        for name, root in trees.items():
            result = tmp / f"{name}.pt"
            subprocess.run([sys.executable, "-u", __file__, "--out", str(args.out),
                            "--worker", str(root), name, str(result)], check=True)
            results[name] = torch.load(result, weights_only=False)

    def same(a: str, b: str) -> tuple[bool, bool]:
        sa, sb = results[a]["seen"], results[b]["seen"]
        loss = (bits(sa["train_loss"]) == bits(sb["train_loss"])
                and bits(sa["val_loss"]) == bits(sb["val_loss"]))
        pa, pb = results[a]["state"], results[b]["state"]
        params = pa.keys() == pb.keys() and all(torch.equal(pa[k], pb[k]) for k in pa)
        return loss, params

    ok = True
    for a, b in (("new_a", "new_b"), ("old", "new_a")):
        loss, params = same(a, b)
        print(f"compare {a} vs {b}: losses bit-identical={loss} parameters equal={params} "
              f"({len(results[a]['state'])} tensors)", flush=True)
        ok = ok and loss and params
    triggered = sum(results[n]["counter"]["triggered"] for n in ("new_a", "new_b"))
    calls = results["new_a"]["counter"]["calls"]
    print(f"pcg_zero_rhs_scopes over all solves of new_a/new_b: {triggered} ({calls} solves per run)")
    ok = ok and triggered == 0 and calls > 0
    print("P1 identity: PASS" if ok else "P1 identity: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
