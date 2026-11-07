"""G3b (b): synthetic known-answer task (pre-registered in g3b_preregistration.md).
Run from H2S-NN/:  python ../docs/D12_evidence/g3b_synthetic.py [--arms full h0_only first_order no_path_cells]
                                                              [--seeds 42 43 44] [--K 4]
The production pipeline end to end (fact layer, walk sampler, model, trainer); only the dataset
loader is replaced by the synthetic HeteroData:
  600 target papers t_i (noise features), authors a_i (noise), carrier papers q_i (noise), 16
  featureless venues; facts (a_i writes t_i), (a_i writes q_i), (q_i published_in V_i);
  label(t_i) = class(V_i) = V_i mod 4. t_i has no venue edge: the label is 3 hops away (t-a-q-V).
Writes docs/D12_evidence/g3b_synthetic.json (appends per run)."""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import src.data.dataset as dataset_module  # noqa: E402
from src.builders import build_experiment_components  # noqa: E402
from src.configs.config import MetaPathConfig, TrainConfig  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

OUT = ROOT.parent / "docs" / "D12_evidence" / "g3b_synthetic.json"
N, VENUES, CLASSES, FEATURES = 600, 16, 4, 16
ARMS = {"full": ({}, {}), "h0_only": ({}, {"h0_only_readout": True}),
        "first_order": ({"encoder": "first_order"}, {}), "no_path_cells": ({}, {"no_path_cells": True})}


def synthetic_graph():
    from torch_geometric.data import HeteroData

    g = torch.Generator().manual_seed(0)
    venue = torch.randint(VENUES, (N,), generator=g)
    data = HeteroData()
    data["paper"].x = torch.randn(2 * N, FEATURES, generator=g)  # 0..N-1 targets, N..2N-1 carriers
    data["author"].x = torch.randn(N, FEATURES, generator=g)
    data["venue"].num_nodes = VENUES
    authors, targets, carriers = torch.arange(N), torch.arange(N), torch.arange(N, 2 * N)
    writes = torch.stack([torch.cat([authors, authors]), torch.cat([targets, carriers])])
    published = torch.stack([carriers, venue])
    data["author", "writes", "paper"].edge_index = writes
    data["paper", "written_by", "author"].edge_index = writes.flip(0)
    data["paper", "published_in", "venue"].edge_index = published
    data["venue", "publishes", "paper"].edge_index = published.flip(0)
    y = torch.full((2 * N,), -1, dtype=torch.long)
    y[:N] = venue % CLASSES
    order = torch.randperm(N, generator=g)
    masks = {name: torch.zeros(2 * N, dtype=torch.bool) for name in ("train", "val", "test")}
    masks["train"][order[: int(0.6 * N)]] = True
    masks["val"][order[int(0.6 * N): int(0.8 * N)]] = True
    masks["test"][order[int(0.8 * N):]] = True
    data["paper"].y = y
    for name, mask in masks.items():
        data["paper"][f"{name}_mask"] = mask
    return data


def config(arm: str, seed: int, K: int) -> TrainConfig:
    raw = yaml.safe_load((ROOT / "configs" / "experiments" / "acm_h2snn.yaml").read_text(encoding="utf-8"))
    model_extra, ablation = ARMS[arm]
    raw["model"] = {**raw["model"], **model_extra, "K": K}
    raw["data"] = {**raw["data"], "resample_paths": False}
    raw.update(seed=seed, epochs=200, device="cuda", ablation=ablation, checkpoint_metric="acc",
               checkpoint_mode="max", exp_name=f"g3b_synth_{arm}_K{K}_s{seed}",
               output_dir="experiments/d12_g3b")
    cfg = TrainConfig.from_dict(raw)
    templates = (MetaPathConfig(name="PAP", node_types=("paper", "author", "paper")),
                 MetaPathConfig(name="PAPV", node_types=("paper", "author", "paper", "venue")))
    cfg.data = dataclasses.replace(cfg.data, metapath_templates=templates, target_node_type="paper")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--K", type=int, default=4)
    args = parser.parse_args()
    graph = synthetic_graph()
    dataset_module._load_hetero_data = lambda _config: graph.clone()  # the only replaced piece
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for arm in args.arms:
        for seed in args.seeds:
            cfg = config(arm, seed, args.K)
            cfg.exp_dir.mkdir(parents=True, exist_ok=True)
            seed_everything(seed, cfg.device)
            data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
            try:
                trainer.fit(data)
            finally:
                logger.close()
            best = json.loads((cfg.exp_dir / "best_test_metrics.json").read_text(encoding="utf-8"))
            rows = [json.loads(l) for l in (cfg.exp_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            test_targets = set(data.target_indices[data.target_test_mask].tolist())
            groups = {grp.template_name: grp for grp in data.face_groups}
            papv = groups.get("PAPV")
            covered = set() if papv is None else set(papv.face_node_index[:, 0].tolist()) | set(
                papv.face_node_index[:, -1].tolist())
            results[f"{arm}/K{args.K}/s{seed}"] = {
                "test_acc": best["test_acc"], "val_acc": best["val_acc"], "best_epoch": best["best_epoch"],
                "epochs_run": len(rows), "train_acc_at_best_trainmode": rows[best["best_epoch"]]["train"]["acc"],
                "face_groups": {name: grp.num_faces for name, grp in groups.items()},
                "test_targets_on_a_PAPV_cell": len(test_targets & covered) / len(test_targets),
            }
            print(arm, f"K{args.K}", seed, json.dumps(results[f"{arm}/K{args.K}/s{seed}"]), flush=True)
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
