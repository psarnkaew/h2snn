from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.training_separability import (
    build_fake_hetero_data,
    plot_all_results,
    plot_result,
    save_result,
    seed_everything,
    train_and_capture,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the current H2SNN on PyG FakeHeteroDataset and visualize "
            "target-node embedding separability over training."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("figures/fakehetero_embedding_separability.png"))
    parser.add_argument(
        "--method",
        choices=["pca", "tsne", "umap", "all"],
        default="all",
        help="Projection to render; 'all' saves separate PCA, t-SNE, and UMAP figures.",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu; cuda falls back to cpu when unavailable")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument(
        "--avg-num-nodes",
        type=int,
        default=400,
        help="Average nodes per fake type; use 1000 for the larger manuscript-style graph.",
    )
    parser.add_argument("--avg-degree", type=float, default=5.0)
    parser.add_argument("--faces-per-anchor", type=int, default=2)
    parser.add_argument("--max-total-faces", type=int, default=4000)
    parser.add_argument("--max-total-incidences", type=int, default=12000)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--use-orthogonal",
        dest="use_orthogonal",
        action="store_true",
        default=True,
        help="Use stable FastHPP restriction maps (the default).",
    )
    parser.add_argument(
        "--no-orthogonal",
        dest="use_orthogonal",
        action="store_false",
        help="Use unconstrained dense maps; this can be numerically unstable because face transport inverts them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.", flush=True)
        device = "cpu"
    seed_everything(args.seed)
    data = build_fake_hetero_data(
        seed=args.seed,
        num_classes=args.num_classes,
        avg_num_nodes=args.avg_num_nodes,
        avg_degree=args.avg_degree,
        faces_per_anchor=args.faces_per_anchor,
        max_total_faces=args.max_total_faces,
        max_total_incidences=args.max_total_incidences,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(
        f"FakeHeteroDataset: nodes={data.num_nodes} target_nodes={data.target_y.numel()} "
        f"edges={data.num_edges} face_instances={sum(group.num_faces for group in data.face_groups)} "
        f"target_type={data.target_node_type} classes={data.num_classes} "
        f"class_counts={torch.bincount(data.target_y, minlength=data.num_classes).tolist()} "
        f"device={device}",
        flush=True,
    )
    result = train_and_capture(
        data,
        device=device,
        epochs=args.epochs,
        snapshot_every=args.snapshot_every,
        hidden_dim=args.hidden_dim,
        diffusion_steps=args.diffusion_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        use_orthogonal=args.use_orthogonal,
        seed=args.seed,
        log_every=args.log_every,
    )
    if args.method == "all":
        figure_paths = plot_all_results(
            result,
            output_path=args.output,
            seed=args.seed,
            tsne_perplexity=args.tsne_perplexity,
            columns=args.columns,
            point_size=args.point_size,
        )
    else:
        plot_result(
            result,
            output_path=args.output,
            method=args.method,  # type: ignore[arg-type]
            seed=args.seed,
            tsne_perplexity=args.tsne_perplexity,
            columns=args.columns,
            point_size=args.point_size,
        )
        figure_paths = (args.output,)
    save_result(result, args.output)
    for figure_path in figure_paths:
        print(f"Saved figure: {figure_path}")
    print(f"Saved trajectory: {args.output.with_suffix('.npz')}")
    print(f"Saved metrics: {args.output.with_suffix('.json')}")
    for epoch, score in zip(result.epochs, result.silhouettes, strict=True):
        print(f"epoch={epoch} silhouette={score:.4f}")


if __name__ == "__main__":
    main()
