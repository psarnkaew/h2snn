from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.embeddings import plot_representations
from src.analysis.model_loading import load_trained_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="best")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, default=Path("figures/acm_representations.png"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    model, data, _ = load_trained_model(args.exp_dir, device=device, tag=args.tag)

    with torch.no_grad():
        encoded = model.encode(
            x=data.x,
            node_type=data.node_type,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            face_groups=data.face_groups,
            closure_edge_index=data.closure_edge_index,
            num_nodes=data.num_nodes,
            num_edges=data.num_edges,
            discovery=data.discovery,
        )

    scores = plot_representations(
        h0=encoded.h0[data.target_indices],
        x_star=encoded.x_star[data.target_indices],
        labels=data.target_y,
        method=args.method,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Saved figure to {args.output}")
    for key, value in scores.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
