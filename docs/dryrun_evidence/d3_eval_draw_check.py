"""Card 2 D3.3: the evaluation draw reproduces on real ACM (spec §6 schema read-out requirement).

    python ../docs/dryrun_evidence/d3_eval_draw_check.py <trial run dir> <work dir>          (from H2S-NN/)

1. End of training: the launcher trial's resolved `config.yaml` is trained as `train_trial` does (same config,
   seed, epochs = its dry-run budget) in a new run directory; after the last epoch's checkpoint the evaluation
   cells (node, fact, sign tensors, relation and signature ids, m_f) and w_f = exp(g(xi_f)) are recorded
   (`evaluation_cell_weights`, the model's own code).
2. Reload in a fresh process: `load_trained_model(run_dir)` (config.yaml + checkpoints/best.pt; `build_dataset`
   rebuilds the evaluation draw from the saved base seed) -> cells and w_f; `torch.equal` with step 1 (same dtype,
   same device type).
3. The launcher trial's own run directory reloaded the same way rebuilds the same cells as step 1 (the draw depends on
   the base seed and the graph only).
Negative control: step 2 with the base seed + 1 in a copy of config.yaml must differ.
No metric is read or printed (the trainer's own console lines go to the job log, outside the evidence).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch  # noqa: E402
import yaml  # noqa: E402

KEYS = ("face_node_index", "face_edge_index", "face_edge_sign", "relation_index", "path_weight", "signature_index")


def cells_of(groups) -> list[dict[str, torch.Tensor]]:
    return [{k: getattr(g, k).detach().cpu().clone() for k in KEYS} for g in groups]


def reload(run_dir: Path, out: Path) -> None:
    """Fresh-process half: the read-out's own loader."""
    from src.analysis.model_loading import evaluation_cell_weights, load_trained_model, nonempty_groups

    model, data, cfg = load_trained_model(run_dir, device="cuda")
    weights = evaluation_cell_weights(model, data)
    torch.save({"cells": cells_of(nonempty_groups(data.face_groups)),
                "weights": [w.detach().cpu().clone() for w in weights],
                "device": str(weights[0].device) if weights else "", "seed": cfg.seed}, out)


def first_difference(a: dict, b: dict, with_weights: bool = True) -> str | None:
    if len(a["cells"]) != len(b["cells"]):
        return f"number of groups {len(a['cells'])} vs {len(b['cells'])}"
    for i, (x, y) in enumerate(zip(a["cells"], b["cells"])):
        for k in KEYS:
            if x[k].dtype != y[k].dtype or x[k].shape != y[k].shape or not torch.equal(x[k], y[k]):
                return f"group {i} {k}"
    if with_weights:
        for i, (x, y) in enumerate(zip(a["weights"], b["weights"])):
            if x.dtype != y.dtype or not torch.equal(x, y):
                return f"group {i} w_f"
    return None


def fresh(run_dir: Path, out: Path) -> dict:
    subprocess.run([sys.executable, __file__, "--reload", str(run_dir), str(out)], check=True)
    return torch.load(out, weights_only=False)


def main(trial_dir: Path, work: Path) -> None:
    from src.analysis.model_loading import evaluation_cell_weights, nonempty_groups
    from src.builders import build_experiment_components
    from src.configs.config import TrainConfig
    from src.utils.misc import save_config, seed_everything

    work.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load((trial_dir / "config.yaml").read_text())
    raw["exp_name"] = "eval_draw_check"
    raw["output_dir"] = str(work)
    raw["use_wandb"] = False
    cfg = TrainConfig.from_dict(raw)
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.exp_dir / "config.yaml")
    seed_everything(cfg.seed, cfg.device)
    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    snapshots: dict[int, dict] = {}
    eval_data = data.to(cfg.device)  # the trainer evaluates on the device copy (trainer.py: prepared_data)

    def snapshot(epoch, *_metrics) -> None:  # after the epoch's checkpoint save; metrics are not read
        snapshots[epoch] = {"cells": cells_of(nonempty_groups(eval_data.face_groups)),
                            "weights": [w.detach().cpu().clone() for w in evaluation_cell_weights(model, eval_data)]}

    try:
        trainer.fit(data, epoch_callback=snapshot)
    finally:
        logger.close()
    best_epoch = int(torch.load(cfg.exp_dir / "checkpoints" / "best.pt", weights_only=False)["epoch"])
    expected = snapshots[best_epoch]
    result = {"trial_dir": str(trial_dir), "seed": cfg.seed, "epochs": cfg.epochs, "best_epoch": best_epoch,
              "device": cfg.device, "groups": len(expected["cells"]),
              "cells": int(sum(c["path_weight"].numel() for c in expected["cells"]))}

    actual = fresh(cfg.exp_dir, work / "reload.pt")
    result["reload_equal"] = first_difference(expected, actual) is None
    result["reload_first_difference"] = first_difference(expected, actual)

    trial = fresh(trial_dir, work / "trial_reload.pt")
    result["trial_reload_cells_equal"] = first_difference(expected, trial, with_weights=False) is None

    mutated = work / "eval_draw_check_other_seed"
    if mutated.exists():
        shutil.rmtree(mutated)
    shutil.copytree(cfg.exp_dir, mutated)
    saved = yaml.safe_load((mutated / "config.yaml").read_text())
    saved["seed"] = cfg.seed + 1
    (mutated / "config.yaml").write_text(yaml.safe_dump(saved, sort_keys=False))
    other = fresh(mutated, work / "other.pt")
    result["negative_control_detected"] = first_difference(expected, other) is not None
    print("EVAL_DRAW " + json.dumps(result), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--reload":
        reload(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
