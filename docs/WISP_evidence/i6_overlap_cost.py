"""I6 (Part I): exposed sampling share of epoch time with overlap sampling (pre-registered gate
<= 25 %, followup_preregistration.md, I6). Run from H2S-NN/:
    python ../docs/WISP_evidence/i6_overlap_cost.py
ACM, DBLP, IMDB at the v1.4 defaults (path_source wisp, r = 42 / 65 / 12, `full`), seed 42,
log_every 1000, overlap on and off. Per epoch:
  wait_s   : Trainer.draw_wait_seconds (the main thread's wait for the draw + the device copy)
  draw_s   : the draw's own duration (in the worker when overlapped), timed around _draw_groups
  train_s  : Trainer._train_epoch with cuda synchronisation
  epoch_s  : whole epoch of Trainer.fit, from consecutive epoch callbacks
Gate: wait_s / epoch_s <= 25 % in steady state, epochs 2-3 (4 epochs run). Epoch 0 always runs
the logged operator diagnostics (6-19 s), so epoch 1's prefetch hides behind an atypical epoch;
epoch 1 (the carded 2-epoch number) is reported too. Accuracy is logged by the trainer and not
copied here. Writes docs/WISP_evidence/i6_overlap_cost.json."""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.builders import build_experiment_components  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

OUT = ROOT.parent / "docs" / "WISP_evidence" / "i6_overlap_cost.json"
EPOCHS = 4


def run(dataset: str, overlap: bool) -> dict:
    raw = yaml.safe_load((ROOT / "configs" / "experiments" / f"{dataset}_h2snn.yaml")
                         .read_text(encoding="utf-8"))
    raw["data"] = {**raw["data"], "overlap_sampling": overlap}
    raw.update(epochs=EPOCHS, seed=42, device="cuda", log_every=1000,
               exp_name=f"i6_{dataset}_overlap_{overlap}", output_dir="experiments/wisp_i6")
    cfg = TrainConfig.from_dict(raw)
    assert cfg.data.path_source == "wisp", cfg.data.path_source
    if (cfg.exp_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{cfg.exp_dir} already has metrics.jsonl (the logger appends).")
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed, cfg.device)
    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    record = {"r": cfg.data.wisp_walks_per_anchor, "draw_s": {}, "train_s": [], "epoch_s": []}
    draw, train = trainer._draw_groups, trainer._train_epoch

    def timed_draw(data_, epoch, proposal):
        t = time.perf_counter()
        out = draw(data_, epoch, proposal)
        record["draw_s"][epoch] = time.perf_counter() - t
        return out

    def timed_train(batch, epoch):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = train(batch, epoch)
        torch.cuda.synchronize()
        record["train_s"].append(time.perf_counter() - t)
        return out

    trainer._draw_groups, trainer._train_epoch = timed_draw, timed_train
    stamps = []

    def callback(*_):
        torch.cuda.synchronize()
        stamps.append(time.perf_counter())

    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        trainer.fit(data, epoch_callback=callback)
    finally:
        logger.close()
    edges = [start] + stamps
    record["epoch_s"] = [b - a for a, b in zip(edges, edges[1:])]
    record["wait_s"] = list(trainer.draw_wait_seconds)
    record["draw_s"] = [record["draw_s"][e] for e in sorted(record["draw_s"])]
    share = [w / e for w, e in zip(record["wait_s"], record["epoch_s"])]
    record["exposed_share"] = share
    steady = range(2, EPOCHS)
    record["steady_exposed_share"] = sum(share[e] for e in steady) / len(steady)
    record["epoch1_exposed_share"] = share[1]
    rows = [json.loads(line) for line in
            (cfg.exp_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    record["train_loss"] = [r["train"]["loss"] for r in rows if "train" in r]
    record["path_log_w_max"] = [r["train"].get("path_log_w_max") for r in rows if "train" in r]
    return record


def main() -> None:
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for dataset in ("acm", "dblp", "imdb"):
        for overlap in (True, False):
            key = f"{dataset}/overlap_{overlap}"
            results[key] = run(dataset, overlap)
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
            r = results[key]
            print("I6", key, "r", r["r"], "steady share", round(r["steady_exposed_share"], 3),
                  "epoch1 share", round(r["epoch1_exposed_share"], 3),
                  "wait", [round(x, 2) for x in r["wait_s"]], "draw", [round(x, 2) for x in r["draw_s"]],
                  "train", [round(x, 2) for x in r["train_s"]], "epoch", [round(x, 2) for x in r["epoch_s"]],
                  flush=True)


if __name__ == "__main__":
    main()
