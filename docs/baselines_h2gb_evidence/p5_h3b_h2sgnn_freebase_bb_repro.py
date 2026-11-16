"""Card 1d H3b: try to reproduce the dry run's H2SGNN LP freebase_bb error (CUDA illegal memory access,
dry-run job 16795) off real data, with CUDA_LAUNCH_BLOCKING=1 so the failing kernel is named.

A Freebase-like release (`tests.test_lp_derived.write_freebase_like`: book-book relation 0 without a stored
reverse, self-loops, book->film relation 1, no features) is built into the derived task freebase_bb with the
project's pipeline (`build_task_by_pair`, `apply_rule`), at the test's size and at a larger size, then
H2SGNN's `fit_task` runs on the GPU with the dry run's parameters. No real data are read.

    CUDA_LAUNCH_BLOCKING=1 python ../docs/baselines_h2gb_evidence/p5_h3b_h2sgnn_freebase_bb_repro.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())  # run from H2S-NN/

from scripts.lp import build_derived_lp as bd  # noqa: E402
from scripts.lp.apply_feature_rule import apply_rule  # noqa: E402
from src.data.data_layout import lp_derived_root  # noqa: E402

# dry-run cells.csv, h2sgnn|link_prediction|freebase_bb, column `point`
DRY_RUN_PARAMS = {"K": 6, "alpha": 0.1, "alpha_list": 0.1, "beta_lr": 1e-05, "beta_wd": 0.0, "decoder_type": "dot",
                  "dropout": 0.45, "emb_dim": 256, "embed_dim": 128, "hidden": 256, "input_drop": 0.45, "lr": 0.001,
                  "poly": "GPR_GPR", "temp_lr": 0.001, "temp_wd": 0.1, "weight_decay": 0.0}


def write_release(folder: Path, n_book: int, n_film: int, n_pairs: int, n_loops: int, seed: int = 0) -> None:
    """`tests.test_lp_derived.write_freebase_like` with its sizes as arguments (pairs drawn without
    enumerating all n_book^2 candidates)."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    nodes = [f"{i}\tbook{i}\t0\n" for i in range(n_book)]
    nodes += [f"{n_book + i}\tfilm{i}\t1\n" for i in range(n_film)]
    (folder / "node.dat").write_bytes("".join(nodes).encode())
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n_pairs:
        a, b = rng.randrange(n_book), rng.randrange(n_book)
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    links = []
    for a, b in sorted(pairs):
        r = rng.random()
        if r < 0.6:
            links += [f"{a}\t{b}\t0\t1.0\n", f"{b}\t{a}\t0\t1.0\n"]
        else:
            links.append(f"{a}\t{b}\t0\t1.0\n" if r < 0.8 else f"{b}\t{a}\t0\t1.0\n")
    links += [f"{i}\t{i}\t0\t1.0\n" for i in range(n_loops)]
    links += [f"{a}\t{n_book + a % n_film}\t1\t1.0\n" for a in range(n_book)]
    (folder / "link.dat").write_bytes("".join(links).encode())
    (folder / "info.dat").write_text(json.dumps({"synthetic": True}), encoding="utf-8")


def run(label: str, sizes: dict[str, int], epochs: int) -> bool:
    from src.baselines.bundles import build_bundle
    from src.baselines.protocol import LINK_PREDICTION, BaselineParams
    from src.baselines.registry import get_adapter

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_release(root / "release", **sizes)
        task_dir = lp_derived_root(root) / "freebase_bb"
        bd.build_task_by_pair(root / "release", task_dir, 0, seed=42, val_frac=0.09, test_frac=0.10)
        apply_rule("freebase_bb", task_dir)
        bundle = build_bundle(LINK_PREDICTION, "freebase_bb", overrides={"data": {"root": str(root)}})
        params = BaselineParams({**DRY_RUN_PARAMS, "epochs": epochs, "early_stopping": epochs})
        print(f"[{label}] sizes={sizes} nodes={int(bundle.data.x.size(0))} "
              f"message_edges={int(bundle.data.edge_index.size(1))}", flush=True)
        try:
            record = get_adapter("h2sgnn").fit_task(bundle, params, task=LINK_PREDICTION, seed=0,
                                                    device=torch.device("cuda"))
            torch.cuda.synchronize()
        except Exception:  # report the kernel; the reproduction is the output
            traceback.print_exc()
            print(f"[{label}] REPRODUCED", flush=True)
            return True
        print(f"[{label}] no error; test_random_auroc={record.get('test_random_auroc')}", flush=True)
        return False


def main() -> int:
    print("CUDA_LAUNCH_BLOCKING", os.environ.get("CUDA_LAUNCH_BLOCKING"), "device", torch.cuda.get_device_name(0))
    reproduced = run("test-size", {"n_book": 50, "n_film": 10, "n_pairs": 120, "n_loops": 5}, epochs=3)
    if not reproduced:
        reproduced = run("large", {"n_book": 20000, "n_film": 500, "n_pairs": 120000, "n_loops": 200}, epochs=2)
    print("RESULT", "reproduced" if reproduced else "not reproduced off real data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
