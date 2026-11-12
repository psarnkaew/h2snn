"""Part N smoke (card N7): correctness checks only, never performance.

One run per invocation, 2 epochs, v3-B anchored, arm `full`, d = 32, l_max = 4, r = 8, default
protocol otherwise; or the GCN baseline at its default protocol for 2 epochs. Checks recorded:
  (a) every training and validation loss finite;
  (b) no stop condition fired (if one fires: its condition, value and epoch; nothing changed);
  (c) no collapse: NC/GC -- validation predictions use >= 2 classes and the mean over coordinates
      of the std over nodes of x* > 1e-6; LP -- std of the scores > 1e-6 on the validation set
      and on every non-empty test set;
  (d) parameters move;
  (e) last-epoch training loss vs first (reported, not a gate).
No accuracy, F1, ROC-AUC or MRR is written or printed.

Exit code: 0 = checks recorded, no non-finite loss and no collapse; 2 = a non-finite loss or a
collapse (card N7: stop N7 and report); 1 = the run raised something that is not a stop condition.

    python scripts/partn_smoke.py --run h2snn:nc:acm --out ../docs/partN_evidence/n7/h2snn_acm.json
    python scripts/partn_smoke.py --run gcn:gc:mutag --out ../docs/partN_evidence/n7/gcn_mutag.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.anchored_solve import (NonFiniteLossError, PCGConvergenceError,  # noqa: E402
                                       StopConditionError)

EPOCHS = 2
WISP = {"wisp_max_length": 4, "wisp_walks_per_anchor": 8}
COLLAPSE_EPS = 1e-6
LP_TARGETS: dict[str, tuple[str, str, str]] = {
    "acm_pa": ("paper", "rel_2", "author"),
    "dblp_ap": ("author", "rel_0", "paper"),
    "imdb_ma": ("movie", "rel_2", "actor"),
    "freebase_bb": ("book", "rel_0", "book"),
    "pubmed": ("disease", "rel_2", "disease"),
    "lastfm": ("user", "rel_0", "artist"),
}
TEMPLATES = {"nc": "configs/experiments/acm_h2snn_v3b.yaml",
             "lp": "configs/experiments/acm_pa_h2snn_v3b.yaml",
             "gc": "configs/experiments/mutag_h2snn_v3b.yaml"}
TASKS = {"nc": "node_classification", "lp": "link_prediction", "gc": "graph_classification"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="<model>:<nc|lp|gc>:<dataset>")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


# ------------------------------------------------------------------------------- recorders


class Recorder:
    """Every training loss the loops check (`check_finite_loss`), per epoch."""

    def __init__(self) -> None:
        self.train_losses: list[tuple[int, float]] = []
        self.val_losses: list[tuple[int, float]] = []
        self.val_predictions: list[torch.Tensor] = []   # last evaluation's, per epoch
        self.x_star: list[torch.Tensor] = []
        self.lp_score_std: list[dict[str, float | None]] = []
        self.class_counts: list[dict[str, Any]] = []
        self.x_star_std: list[float] = []

    def install(self) -> None:
        import src.baselines.training as baseline_training
        import src.trainers.graph_classification_trainer as gc_trainer
        import src.trainers.link_prediction_trainer as lp_trainer
        import src.trainers.trainer as nc_trainer

        for module in (baseline_training, gc_trainer, lp_trainer, nc_trainer):
            original = module.check_finite_loss

            def checked(loss: torch.Tensor, epoch: int, _original=original) -> None:
                self.train_losses.append((int(epoch), float(loss.detach())))
                _original(loss, epoch)

            module.check_finite_loss = checked

    def close_epoch(self, epoch: int, predictions: torch.Tensor, x_star: torch.Tensor,
                    multilabel: bool = False) -> None:
        if multilabel:  # per genre: predicted present and absent both occur
            pred = predictions.bool()
            per_genre = [int(pred[:, g].any() and (~pred[:, g]).any()) for g in range(pred.size(1))]
            self.class_counts.append({"epoch": epoch, "genres_with_both_values": sum(per_genre),
                                      "genres": len(per_genre)})
        else:
            values = torch.unique(predictions).tolist()
            self.class_counts.append({"epoch": epoch, "classes_predicted": len(values)})
        self.x_star_std.append(float(x_star.float().std(dim=0, unbiased=False).mean()))


def _snapshot(modules: list[torch.nn.Module]) -> dict[str, torch.Tensor]:
    return {f"{i}.{name}": p.detach().clone().cpu()
            for i, module in enumerate(modules) for name, p in module.named_parameters()}


def _moved(before: dict[str, torch.Tensor], modules: list[torch.nn.Module]) -> dict[str, Any]:
    try:
        after = _snapshot(modules)
    except RuntimeError as exc:  # e.g. the device is unusable after a CUDA OOM: keep the run's error
        return {"parameters_moved": None, "parameters_total": len(before),
                "parameters_snapshot_error": f"{type(exc).__name__}: {exc}"[:300]}
    delta = {name: float((after[name] - before[name]).abs().max()) for name in before}
    return {"parameters_moved": sum(v > 0 for v in delta.values()),
            "parameters_total": len(delta),
            "parameters_not_moved": sorted(name for name, v in delta.items() if v == 0),
            "max_abs_change": max(delta.values()) if delta else 0.0}


def _stop_record(exc: BaseException) -> dict[str, Any]:
    return {"condition": getattr(exc, "condition", type(exc).__name__),
            "value": getattr(exc, "value", None), "limit": getattr(exc, "limit", None),
            "epoch": getattr(exc, "epoch", None), "message": str(exc)[:500]}


# ------------------------------------------------------------------------------- H2S-NN


def h2snn_config(kind: str, dataset: str, device: str):
    from src.configs.config import TrainConfig

    payload = yaml.safe_load((PROJECT_ROOT / TEMPLATES[kind]).read_text(encoding="utf-8"))
    payload["data"].update(WISP)
    if kind == "lp":
        payload["data"]["dataset_name"] = dataset
        payload["data"]["task_type"] = "link_prediction"
        payload["data"]["target_relation"] = list(LP_TARGETS[dataset])
    payload.update({"device": device, "epochs": EPOCHS, "log_every": 1,
                    "exp_name": f"partn_smoke_h2snn_{dataset}",
                    "output_dir": "experiments/partn_smoke"})
    return TrainConfig.from_dict(payload), payload


def run_h2snn(kind: str, dataset: str, device: str, rec: Recorder) -> dict[str, Any]:
    from src.builders import build_experiment_components
    from src.utils.misc import lp_test_sets, seed_everything

    cfg, payload = h2snn_config(kind, dataset, device)
    seed_everything(cfg.seed, cfg.device)
    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    before = _snapshot([model])
    epochs: list[dict[str, Any]] = []
    state: dict[str, Any] = {"in_val": False}

    if kind == "nc":
        original_eval = trainer._eval_epoch

        def eval_epoch(data_, mask, epoch=-1):
            metrics = original_eval(data_, mask, epoch)
            if mask is data_.target_val_mask:
                with torch.no_grad():
                    out = model(data_, epoch=-1)
                logits = out.logits[mask]
                preds = (logits > 0).long() if data_.is_multilabel else logits.argmax(-1)
                state["val"] = (preds.cpu(), out.x_star.detach().cpu(), bool(data_.is_multilabel))
            return metrics

        trainer._eval_epoch = eval_epoch
    elif kind == "gc":
        original_run, original_encode = trainer._run_epoch, model.encode
        logits_parts: list[torch.Tensor] = []
        x_parts: list[torch.Tensor] = []

        def encode(*args, **kwargs):
            encoded = original_encode(*args, **kwargs)
            if state["in_val"]:
                x_parts.append(encoded.x_star.detach().cpu())
            return encoded

        original_forward = trainer.forward_batch

        def forward_batch(batch, seed=None, epoch=-1):
            logits = original_forward(batch, seed=seed, epoch=epoch)
            if state["in_val"]:
                logits_parts.append(logits.detach().cpu())
            return logits

        def run_epoch(loader, epoch, train):
            state["in_val"] = (not train) and loader is data.val_loader
            if state["in_val"]:
                logits_parts.clear()
                x_parts.clear()
            try:
                return original_run(loader, epoch, train)
            finally:
                if state["in_val"]:
                    state["val"] = (torch.cat(logits_parts).argmax(-1), torch.cat(x_parts), False)
                state["in_val"] = False

        model.encode, trainer.forward_batch, trainer._run_epoch = encode, forward_batch, run_epoch

    def callback(epoch: int, train: dict, val: dict, test: dict) -> None:
        epochs.append({"epoch": epoch, "train_loss": train.get("loss"),
                       "val_loss": val.get("loss"), "tau": train.get("tau"),
                       "path_log_w_range": train.get("path_log_w_range"),
                       "pcg_iters": {k: v for k, v in train.items() if k.endswith("_iters")}})
        rec.val_losses.append((epoch, float(val.get("loss", float("nan")))))
        if kind in ("nc", "gc"):
            preds, x_star, multilabel = state.pop("val")
            rec.close_epoch(epoch, preds, x_star, multilabel)
            return
        prepared = data.to(trainer.device)
        model.eval()
        with torch.no_grad():
            encoded = model.encode(x=prepared.x, node_type=prepared.node_type,
                                   edge_index=prepared.edge_index, edge_type=prepared.edge_type,
                                   face_groups=prepared.face_groups,
                                   num_nodes=prepared.num_nodes, epoch=-1)

            def score(edge_index, relation):
                return model.link_readout(node_representation=encoded.representation,
                                          edge_label_index=edge_index, edge_relation=relation)

            val_scores = torch.cat([score(prepared.val_pos_edge_index,
                                          prepared.val_pos_edge_relation),
                                    score(prepared.val_neg_edge_index,
                                          prepared.val_neg_edge_relation)])
            stds: dict[str, float | None] = {"val": float(val_scores.std(unbiased=False))}
            pos = score(prepared.test_pos_edge_index, prepared.test_pos_edge_relation)
            for name, (neg, neg_rel, _reason) in lp_test_sets(prepared).items():
                stds[f"test_{name}"] = (None if neg.size(1) == 0 else
                                        float(torch.cat([pos, score(neg, neg_rel)])
                                              .std(unbiased=False)))
        rec.lp_score_std.append({"epoch": epoch, **stds})

    stop = None
    error = None
    start = time.perf_counter()
    try:
        trainer.fit(data, epoch_callback=callback)
    except (StopConditionError, PCGConvergenceError) as exc:
        stop = _stop_record(exc)
    except Exception as exc:  # recorded; exit code 1 below
        error = f"{type(exc).__name__}: {exc}"[:1000]
    finally:
        logger.close()
    return {"model": "h2snn", "config": {"template": TEMPLATES[kind], "data": payload["data"],
                                          "model": payload["model"], "epochs": EPOCHS,
                                          "seed": payload["seed"]},
            "epochs": epochs, "stop_condition": stop, "error": error,
            "seconds": time.perf_counter() - start, **_moved(before, [model])}


# ------------------------------------------------------------------------------- GCN baseline


def run_gcn(kind: str, dataset: str, device: str, rec: Recorder) -> dict[str, Any]:
    import src.baselines.adapters.hgb_gnn as hgb_gnn
    from src.baselines.campaign import fit_cell
    from src.baselines.registry import get_adapter, unsupported_reason

    task = TASKS[kind]
    reason = unsupported_reason("gcn", task, dataset)
    if reason:
        return {"model": "gcn", "not_runnable": reason, "epochs": [], "stop_condition": None,
                "error": None, "parameters_moved": None, "parameters_total": None}
    published = get_adapter("gcn").published_params(task, dataset)
    params = dict(published.values) if published is not None else {}
    params["epochs"] = EPOCHS
    moved: dict[str, Any] = {}
    criterion_holder: dict[str, Any] = {}

    if kind == "nc":
        original = hgb_gnn.train_full_batch_node_classification

        def train_nc(**kwargs):
            model, forward = kwargs["model"], kwargs["forward"]
            val_idx, labels = kwargs["val_idx"], kwargs["labels"]
            num_target, criterion = kwargs["num_target"], kwargs["criterion"]
            before = _snapshot([model])
            epoch_box = [0]

            def recorded_forward():
                logits = forward()
                if not model.training:
                    target = logits[:num_target]
                    val_logits = target[val_idx]
                    rec.val_losses.append((epoch_box[0],
                                           float(criterion(val_logits, labels[val_idx]))))
                    preds = ((val_logits > 0).long() if kwargs["is_multilabel"]
                             else val_logits.argmax(-1))
                    # x* for GCN: its last layer's output over the target nodes (the vendor GCN
                    # has no separate embedding; recorded as a deviation)
                    rec.close_epoch(epoch_box[0], preds.cpu(), target.detach().cpu(),
                                    kwargs["is_multilabel"])
                    epoch_box[0] += 1
                return logits

            kwargs["forward"] = recorded_forward
            try:
                return original(**kwargs)
            finally:
                moved.update(_moved(before, [model]))

        hgb_gnn.train_full_batch_node_classification = train_nc
    else:
        original = hgb_gnn.train_graph_classification

        def train_gc(**kwargs):
            model, head, embed = kwargs["model"], kwargs["head"], kwargs["embed_batch"]
            _, val_loader, _ = kwargs["loaders"]
            before = _snapshot([model, head])
            state = {"in_val": False, "x": [], "logits": []}

            class ValLoader:
                def __iter__(self_inner):
                    state["in_val"] = True
                    state["x"], state["logits"] = [], []
                    try:
                        yield from val_loader
                    finally:
                        state["in_val"] = False

            def recorded_embed(batch):
                out = embed(batch)
                if state["in_val"]:
                    state["x"].append(out.detach().cpu())
                return out

            def head_hook(_module, _inputs, output):
                if state["in_val"]:
                    state["logits"].append(output.detach().cpu())

            handle = head.register_forward_hook(head_hook)

            def report(*, epoch, monitor, running_best, metrics):
                rec.val_losses.append((epoch, float(metrics["loss"])))
                rec.close_epoch(epoch, torch.cat(state["logits"]).argmax(-1),
                                torch.cat(state["x"]))

            loaders = kwargs["loaders"]
            kwargs.update(embed_batch=recorded_embed, report=report,
                          loaders=(loaders[0], ValLoader(), loaders[2]))
            try:
                return original(**kwargs)
            finally:
                handle.remove()
                moved.update(_moved(before, [model, head]))

        hgb_gnn.train_graph_classification = train_gc

    raw_base = {"model": "gcn", "task": task, "dataset": dataset, "seed": 42, "run_index": 0}
    stop = error = None
    start = time.perf_counter()
    try:
        fit_cell(raw_base, params, source="smoke", device=torch.device(device))
    except (StopConditionError, PCGConvergenceError) as exc:
        stop = _stop_record(exc)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:1000]
    per_epoch: dict[int, list[float]] = {}
    for epoch, value in rec.train_losses:
        per_epoch.setdefault(epoch, []).append(value)
    epochs = [{"epoch": e, "train_loss": sum(v) / len(v)} for e, v in sorted(per_epoch.items())]
    for entry in epochs:
        entry["val_loss"] = next((v for e, v in rec.val_losses if e == entry["epoch"]), None)
    return {"model": "gcn", "config": {"params": {k: v for k, v in params.items()},
                                       "params_source": "published" if published else "defaults",
                                       "seed": 42, "run_index": 0},
            "epochs": epochs, "stop_condition": stop, "error": error,
            "seconds": time.perf_counter() - start, **moved}


# ------------------------------------------------------------------------------- checks


def checks(kind: str, rec: Recorder, result: dict[str, Any]) -> dict[str, Any]:
    if result.get("not_runnable"):
        return {"runnable": False}
    losses = [v for _, v in rec.train_losses] + [v for _, v in rec.val_losses]
    out: dict[str, Any] = {
        "runnable": True,
        "a_losses_finite": bool(losses) and all(math.isfinite(v) for v in losses),
        "train_loss_steps": len(rec.train_losses),
        "non_finite_train_losses": [(e, v) for e, v in rec.train_losses if not math.isfinite(v)],
        "val_losses": rec.val_losses,
        "b_no_stop_condition": result["stop_condition"] is None,
    }
    if kind == "lp":
        flat = [v for entry in rec.lp_score_std for k, v in entry.items()
                if k != "epoch" and v is not None]
        out["c_score_std"] = rec.lp_score_std
        out["c_no_collapse"] = bool(flat) and all(v > COLLAPSE_EPS for v in flat)
    else:
        out["c_val_prediction_classes"] = rec.class_counts
        out["c_x_star_std_mean"] = rec.x_star_std
        classes_ok = all(c.get("classes_predicted", 2) >= 2 and
                         c.get("genres_with_both_values", 1) >= 1 for c in rec.class_counts)
        out["c_no_collapse"] = (bool(rec.class_counts) and classes_ok and
                                all(v > COLLAPSE_EPS for v in rec.x_star_std))
    out["d_parameters_moved"] = (result.get("parameters_moved") or 0) > 0
    epochs = result["epochs"]
    out["e_train_loss_first_last"] = ((epochs[0]["train_loss"], epochs[-1]["train_loss"])
                                      if epochs else None)
    return out


def main() -> None:
    args = parse_args()
    model_name, kind, dataset = args.run.split(":")
    rec = Recorder()
    rec.install()
    runner = {"h2snn": run_h2snn, "gcn": run_gcn}[model_name]
    result = runner(kind, dataset, args.device, rec)
    result.update({"run": args.run, "task": TASKS[kind], "dataset": dataset,
                   "device": args.device,
                   "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None})
    result["checks"] = checks(kind, rec, result)
    result["note"] = ("Part N smoke: correctness checks only; no accuracy, F1, ROC-AUC or MRR "
                      "is recorded (card N7)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    c = result["checks"]
    print(json.dumps({"run": args.run, "error": result["error"],
                      "not_runnable": result.get("not_runnable"),
                      "stop_condition": result["stop_condition"],
                      **{k: c[k] for k in c if k in ("a_losses_finite", "b_no_stop_condition",
                                                      "c_no_collapse", "d_parameters_moved",
                                                      "e_train_loss_first_last")}},
                     indent=1, default=str))
    if not c.get("runnable", True):
        sys.exit(0)
    stop = result["stop_condition"] or {}
    if (not c["a_losses_finite"] or stop.get("condition") == "non_finite_loss"
            or not c["c_no_collapse"]):
        sys.exit(2)
    if result["error"] is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
