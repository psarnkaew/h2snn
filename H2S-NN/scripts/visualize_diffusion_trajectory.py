from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.diffusion_trajectory import compute_trajectory, plot_trajectory
from src.analysis.model_loading import load_trained_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="best")
    parser.add_argument("--num_steps", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, default=Path("figures/acm_diffusion_trajectory.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    model, data, cfg = load_trained_model(args.exp_dir, device=device, tag=args.tag)
    print(f"Trained K={cfg.model.K}; rendering illustrative rollout of {args.num_steps} steps.")

    snapshots = compute_trajectory(model, data, num_steps=args.num_steps)
    plot_trajectory(snapshots, labels=data.target_y, output_path=args.output)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
