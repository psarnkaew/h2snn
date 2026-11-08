"""G3c diagnosis D2 (addendum in g3c_preregistration.md): the v2 full arm with w_f replaced by the
oracle indicator (1 on the relevant cell (P, L_P, Q*_P) of every target, 0 on every other cell).
Everything else is the full arm of g3c_binding.py (same config, budget and seeds).
Run from H2S-NN/:  python ../docs/D12_evidence/g3c_v2_oracle.py [--seeds 42 43 44]
Writes docs/D12_evidence/g3c_oracle_results.json. A diagnostic, not a G3c result."""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "docs" / "D12_evidence"))
import g3c_binding as g3c  # noqa: E402
import src.data.dataset as dataset_module  # noqa: E402
from src.builders import build_experiment_components  # noqa: E402
from src.models.h2snn_model import H2SNN  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

OUT = ROOT.parent / "docs" / "D12_evidence" / "g3c_oracle_results.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    graph, truth, _ = g3c.graph_v2()
    dataset_module._load_hetero_data = lambda _config: graph.clone()
    original = H2SNN._path_weights
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for seed in args.seeds:
        cfg = dataclasses.replace(g3c.config("v2", "full", seed), exp_name=f"g3c_v2_oracle_s{seed}")
        if (cfg.exp_dir / "metrics.jsonl").exists():
            raise SystemExit(f"{cfg.exp_dir} already has metrics.jsonl (the logger appends).")
        cfg.exp_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(seed, cfg.device)
        start = time.time()
        data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
        group, relevant, _ = g3c.cell_index(data, truth, list(range(g3c.NT)))
        assert bool((relevant >= 0).all()), "every target's relevant cell must be in the draw"
        oracle = torch.zeros(group.num_faces)
        oracle[relevant] = 1.0

        def oracle_weights(self, latent, grp, like):
            if grp.num_faces != oracle.numel():
                return original(self, latent, grp, like)
            return oracle.to(device=like.device, dtype=like.dtype)

        H2SNN._path_weights = oracle_weights
        try:
            trainer.fit(data)
        finally:
            logger.close()
            H2SNN._path_weights = original
        best = json.loads((cfg.exp_dir / "best_test_metrics.json").read_text(encoding="utf-8"))
        rows = [json.loads(l) for l in (cfg.exp_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        entry = {"test_acc": best["test_acc"], "val_acc": best["val_acc"], "best_epoch": best["best_epoch"],
                 "epochs_run": len(rows), "train_acc_at_best_trainmode": rows[best["best_epoch"]]["train"]["acc"],
                 "oracle_cells_on": int(oracle.sum()), "cells_PAP": group.num_faces,
                 "seconds": round(time.time() - start, 1)}
        results[f"v2/oracle_gate/s{seed}"] = entry
        print("v2 oracle_gate", seed, json.dumps(entry), flush=True)
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
