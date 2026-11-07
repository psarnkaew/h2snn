"""G0 (b): reference baseline R-GCN (HetSheaf's vendored implementation and committed config) on
our pipeline (arm O) and on HetSheaf's pipeline (arm P, control). See g0_preregistration.md.

Run from H2S-NN/ with the h2snn environment:
  python ../docs/D12_evidence/g0_rgcn.py [--datasets acm dblp imdb] [--arms O P] [--seed 42]
Writes docs/D12_evidence/g0_rgcn_<dataset>_<arm>.json (per-epoch curves + summary).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
HETSHEAF = ROOT / "vendor" / "Heterogeneous-Sheaf-Neural-Networks" / "hetsheaf"
OUT = ROOT.parent / "docs" / "D12_evidence"
TARGET = {"acm": "paper", "dblp": "author", "imdb": "movie"}
# Committed defaults: configs/model/rgcn.yaml, nc_config.yaml, trainer/trainer.yaml, callbacks/sheaf_nc.yaml.
CONFIG = {"layers": 2, "hidden_channels": 32, "dropout": 0.2, "input_dropout": 0.0, "initial_dropout": 0.0,
          "output_dim": 32, "lr": 1e-3, "weight_decay": 5e-4, "t_max": 1000, "eta_min": 1e-6,
          "max_epochs": 200, "patience": 100}


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_torch(seed: int) -> None:
    """hetsheaf/utils/instantiators.py:61-76."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def arm_p(dataset: str, drop: tuple[str, ...] = ()) -> dict:
    """HetSheaf's HGBBaseDataModule.prepare_data (hetsheaf/datasets/hgb.py:49-77), homogeneous=True.
    `drop` removes named transforms (diagnostic swaps only)."""
    import torch_geometric.transforms as T
    from torch_geometric.datasets import HGBDataset

    utils = load_file("hetsheaf_utils", HETSHEAF / "datasets" / "utils.py")
    chain = {"constant": T.Constant(node_types=None), "to_undirected": T.ToUndirected(),
             "split": utils.SubsetTrainSplit(val_ratio=0.1), "remove_self_loops": utils.RemoveSelfLoops(),
             "normalize": T.NormalizeFeatures(), "remove_duplicates": T.RemoveDuplicatedEdges()}
    unknown = set(drop) - set(chain)
    if unknown:
        raise ValueError(f"unknown transforms {unknown}")
    transform = T.Compose([t for name, t in chain.items() if name not in drop])
    data = HGBDataset(root=str(ROOT / "data" / dataset), name=dataset.upper(), transform=transform)[0].coalesce()
    num_relations = len(data.metadata()[1])
    hom = data.to_homogeneous()
    multilabel = hom.y.dim() == 2
    valid = torch.any(~hom.y.isnan(), dim=1) if multilabel else hom.y != -1
    return {"edge_index": hom.edge_index, "edge_type": hom.edge_type, "num_nodes": hom.num_nodes,
            "num_relations": num_relations, "y": hom.y, "multilabel": multilabel,
            "train": hom.train_mask & valid, "val": hom.val_mask & valid, "test": hom.test_mask & valid,
            "num_classes": int(hom.y.size(1)) if multilabel else int(hom.y.max()) + 1,
            "relations": [list(r) for r in data.metadata()[1]]}


def arm_o(dataset: str) -> dict:
    """Our build_dataset; each fact (u, r, v) -> relation r, its reverse -> relation r + R."""
    from src.configs.config import TrainConfig
    from src.data.dataset import _load_hetero_data, build_dataset
    from src.data.hetero_flatten import flatten_edges, stack_node_features

    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / f"{dataset}_h2snn.yaml").read_text())
    cfg["device"] = "cpu"
    config = TrainConfig.from_dict(cfg)
    ours = build_dataset(config)
    layer = ours.fact_layer
    R = len(layer.kept)
    ei, et = ours.edge_index, ours.edge_type
    edge_index = torch.cat([ei, ei.flip(0)], dim=1)
    edge_type = torch.cat([et, et + R])

    # Faithfulness: facts + reversal (r + R -> rho(r)) reproduce the stored HGB edge multiset.
    raw = _load_hetero_data(config)
    _, _, _, slices, _ = stack_node_features(raw)
    stored_ei, stored_et, _, _ = flatten_edges(raw, slices)
    kept = torch.tensor(layer.kept)
    rho = torch.tensor([r if layer.rho[r] is None else layer.rho[r] for r in range(len(layer.rho))])
    recon_type = torch.cat([kept[et], rho[kept[et]]])
    unmatched = torch.tensor([layer.rho[int(r)] is None for r in kept])
    keep_rev = torch.cat([torch.ones_like(et, dtype=torch.bool), ~unmatched[et]])

    def keys(e, t):
        n = int(ours.num_nodes)
        return torch.sort((t.long() * n + e[0].long()) * n + e[1].long()).values

    faithful = bool(torch.equal(keys(edge_index[:, keep_rev], recon_type[keep_rev]), keys(stored_ei, stored_et)))
    return {"edge_index": edge_index, "edge_type": edge_type, "num_nodes": int(ours.num_nodes),
            "num_relations": 2 * R, "y": ours.y, "multilabel": ours.is_multilabel,
            "train": ours.train_mask, "val": ours.val_mask, "test": ours.test_mask,
            "num_classes": ours.num_classes, "relations": list(ours.relation_names),
            "fact_layer": {"mode": layer.mode, "kept": list(layer.kept), "rho": list(layer.rho),
                           "self_inverse": list(layer.self_inverse), "facts_before": layer.facts_before,
                           "facts_after": layer.facts_after},
            "reconstructs_stored_edges": faithful, "stored_edges": int(stored_ei.size(1))}


def metrics_ours(logits, y, multilabel):
    from src.utils.misc import compute_classification_metrics, compute_multilabel_classification_metrics
    if multilabel:
        return compute_multilabel_classification_metrics(logits, y)
    return compute_classification_metrics(logits, y)


def metrics_torchmetrics(logits, y, multilabel, num_classes):
    """NodeClassifier's metric code: torchmetrics F1Score on act_fn(y_hat) (node_classifier.py:44-56, 92-97)."""
    from torchmetrics.classification import F1Score
    task = "multilabel" if multilabel else "multiclass"
    kw = {"task": task, "num_labels": num_classes, "num_classes": num_classes}
    probs = torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=-1)
    target = y.to(torch.int)
    return {"macro_f1": float(F1Score(average="macro", **kw).to(logits.device)(probs, target)),
            "micro_f1": float(F1Score(average="micro", **kw).to(logits.device)(probs, target))}


def swap(dataset: str, graph_from: str, split_from: str, p_drop: tuple[str, ...]) -> dict:
    """Graph (edges, relations) of one arm with the split of the other. Node order and labels must agree."""
    o, p = arm_o(dataset), arm_p(dataset, p_drop if "split" not in p_drop else tuple(d for d in p_drop if d != "split"))
    if o["num_nodes"] != p["num_nodes"] or not torch.equal(o["test"], p["test"]):
        raise RuntimeError("arms disagree on node order or test set; swap undefined")
    labelled = o["train"] | o["val"] | o["test"]
    if not torch.equal(o["y"][labelled].float(), p["y"][labelled].float()):
        raise RuntimeError("arms disagree on labels; swap undefined")
    g, s = (o if graph_from == "O" else p), (o if split_from == "O" else p)
    return {**g, "train": s["train"], "val": s["val"], "test": s["test"], "y": o["y"]}


def run(dataset: str, arm: str, seed: int, device: torch.device, graph_from: str | None = None,
        split_from: str | None = None, p_drop: tuple[str, ...] = ()) -> dict:
    setup_torch(seed)
    if graph_from or split_from:
        data = swap(dataset, graph_from or arm, split_from or arm, p_drop)
        arm = f"graph{graph_from or arm}_split{split_from or arm}" + (f"_drop-{'-'.join(p_drop)}" if p_drop else "")
    else:
        data = arm_p(dataset, p_drop) if arm == "P" else arm_o(dataset)
        if p_drop:
            arm = f"P_drop-{'-'.join(p_drop)}"
    rgcn = load_file("hetsheaf_rgcn", HETSHEAF / "models" / "gnn_baselines" / "RGCN.py")
    from torch_geometric.data import Data
    from torch_geometric.nn import Linear

    model = rgcn.RGCN(hidden_channels=CONFIG["hidden_channels"], num_nodes=data["num_nodes"],
                      num_relations=data["num_relations"], layers=CONFIG["layers"], dropout=CONFIG["dropout"],
                      input_dropout=CONFIG["input_dropout"], initial_dropout=CONFIG["initial_dropout"],
                      output_dim=CONFIG["output_dim"]).to(device)
    decoder = Linear(CONFIG["output_dim"], data["num_classes"]).to(device)
    params = list(model.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["t_max"], eta_min=CONFIG["eta_min"])
    graph = Data(edge_index=data["edge_index"].to(device), edge_type=data["edge_type"].to(device),
                 num_nodes=data["num_nodes"])
    y = data["y"].to(device)
    masks = {k: data[k].to(device) for k in ("train", "val", "test")}
    multilabel = data["multilabel"]
    loss_fn = F.multilabel_soft_margin_loss if multilabel else F.cross_entropy

    curve, best_loss, wait, start = [], float("inf"), 0, time.time()
    for epoch in range(CONFIG["max_epochs"]):
        model.train(); decoder.train()
        optimizer.zero_grad()
        logits = decoder(model(graph))
        loss = loss_fn(logits[masks["train"]], y[masks["train"]])
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.eval(); decoder.eval()
        with torch.no_grad():
            logits = decoder(model(graph))
            row = {"epoch": epoch, "train_loss": float(loss),
                   "val_loss": float(loss_fn(logits[masks["val"]], y[masks["val"]]))}
            for split in ("val", "test"):
                m = metrics_ours(logits[masks[split]], y[masks[split]], multilabel)
                row[f"{split}_macro_f1"], row[f"{split}_micro_f1"] = m["macro_f1"], m["micro_f1"]
            tm = metrics_torchmetrics(logits[masks["test"]], y[masks["test"]], multilabel, data["num_classes"])
            row["test_macro_f1_torchmetrics"], row["test_micro_f1_torchmetrics"] = tm["macro_f1"], tm["micro_f1"]
        curve.append(row)
        # Lightning EarlyStopping(monitor=valid/loss, mode=min, min_delta=0, patience=100).
        if row["val_loss"] < best_loss:
            best_loss, wait = row["val_loss"], 0
        else:
            wait += 1
            if wait >= CONFIG["patience"]:
                break
    best_val = max(curve, key=lambda r: r["val_macro_f1"])  # first max
    final = curve[-1]
    summary = {
        "dataset": dataset, "arm": arm, "seed": seed, "device": str(device), "config": CONFIG,
        "epochs_run": len(curve), "minutes": (time.time() - start) / 60,
        "sizes": {k: int(masks[k].sum()) for k in masks}, "num_relations": data["num_relations"],
        "num_edges": int(data["edge_index"].size(1)), "num_nodes": data["num_nodes"],
        "relations": data["relations"],
        "test_at_best_val_macro_f1": {"epoch": best_val["epoch"], "macro_f1": best_val["test_macro_f1"],
                                      "micro_f1": best_val["test_micro_f1"]},
        "test_at_final_epoch": {"epoch": final["epoch"], "macro_f1": final["test_macro_f1"],
                                "micro_f1": final["test_micro_f1"],
                                "macro_f1_torchmetrics": final["test_macro_f1_torchmetrics"],
                                "micro_f1_torchmetrics": final["test_micro_f1_torchmetrics"]},
        "params": sum(p.numel() for p in params),
    }
    for key in ("fact_layer", "reconstructs_stored_edges", "stored_edges"):
        if key in data:
            summary[key] = data[key]
    (OUT / f"g0_rgcn_{dataset}_{arm}.json").write_text(json.dumps({"summary": summary, "curve": curve}, indent=1),
                                                       encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp", "imdb"])
    parser.add_argument("--arms", nargs="+", default=["O", "P"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph_from", choices=["O", "P"], default=None, help="diagnostic swap")
    parser.add_argument("--split_from", choices=["O", "P"], default=None, help="diagnostic swap")
    parser.add_argument("--p_drop", nargs="*", default=[], help="drop named transforms from arm P's chain")
    args = parser.parse_args()
    for dataset in args.datasets:
        for arm in args.arms:
            s = run(dataset, arm, args.seed, torch.device(args.device), args.graph_from, args.split_from,
                    tuple(args.p_drop))
            print(json.dumps({k: s[k] for k in ("dataset", "arm", "epochs_run", "sizes", "num_relations",
                                                "num_edges", "test_at_best_val_macro_f1", "test_at_final_epoch")
                              } | {k: s[k] for k in ("reconstructs_stored_edges", "fact_layer") if k in s}),
                  flush=True)


if __name__ == "__main__":
    main()
