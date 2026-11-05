from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.training_separability import (  # noqa: E402
    SeparabilityResult,
    build_fake_hetero_data,
    plot_all_results,
    save_result,
    seed_everything,
    train_and_capture,
)


def _target_edges(data) -> torch.Tensor:
    """Return a connected, local target-node graph for the A control plot."""
    start, end = data.node_slices[data.target_node_type]
    edge_index = data.edge_index.detach().cpu()
    mask = (
        (edge_index[0] >= start)
        & (edge_index[0] < end)
        & (edge_index[1] >= start)
        & (edge_index[1] < end)
    )
    pairs = {
        tuple(sorted((int(u - start), int(v - start))))
        for u, v in edge_index[:, mask].t().tolist()
        if u != v
    }
    n = int(data.target_y.numel())
    # A ring makes the oracle control connected even when FakeHeteroDataset has
    # no target-target relation. Existing target-target edges are retained.
    if n > 1:
        pairs.update((i, (i + 1) % n) for i in range(n))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()


def _rotation(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c))))


def oracle_o2_trajectory(
    data,
    *,
    seed: int,
    steps: int,
    step_size: float,
    initialization: str,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """Class-aligned O(2) connection diffusion, used only as a positive control.

    Each class is assigned a phase on the circle. For an edge u--v, the fixed
    transport is R(phi_u - phi_v). The compatible sections therefore satisfy
    x_u = R(phi_u) z, which creates class-dependent directions by construction.
    """
    labels = data.target_y.detach().cpu()
    n = int(labels.numel())
    num_classes = int(data.num_classes)
    edges = _target_edges(data)
    generator = torch.Generator().manual_seed(seed)
    phases = 2.0 * math.pi * labels.float() / max(num_classes, 1)
    class_centers = torch.stack((torch.cos(phases), torch.sin(phases)), dim=1)
    if initialization == "random":
        x = torch.randn((n, 2), generator=generator)
    elif initialization == "shared":
        x = torch.tensor([1.0, 0.0]).repeat(n, 1)
        x = x + 0.10 * torch.randn((n, 2), generator=generator)
    elif initialization == "class-aligned":
        # Positive control: intentionally label-aware initial coordinates.
        x = class_centers + 0.35 * torch.randn((n, 2), generator=generator)
    else:
        raise ValueError(f"Unknown oracle initialization: {initialization}")
    degrees = torch.zeros(n, dtype=torch.float32)
    degrees.scatter_add_(0, edges[0], torch.ones(edges.size(1)))
    degrees.scatter_add_(0, edges[1], torch.ones(edges.size(1)))
    degrees.clamp_min_(1.0)
    trajectory = [x.clone()]
    for _ in range(steps):
        grad = torch.zeros_like(x)
        for u, v in edges.t().tolist():
            u, v = int(u), int(v)
            transport = _rotation(phases[u] - phases[v])
            residual_u = x[u] - transport @ x[v]
            residual_v = x[v] - transport.t() @ x[u]
            grad[u] += residual_u / degrees[u]
            grad[v] += residual_v / degrees[v]
        x = x - step_size * grad
        trajectory.append(x.clone())
    return tuple(trajectory), labels, edges


def _direct_oracle_plot(
    trajectory: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    edges: torch.Tensor,
    *,
    output_path: Path,
    columns: int,
    point_size: float,
    initialization: str,
) -> tuple[float, ...]:
    import matplotlib.pyplot as plt

    silhouettes = tuple(
        float(silhouette_score(x.numpy(), labels.numpy())) for x in trajectory
    )
    rows = (len(trajectory) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.4 * rows), squeeze=False)
    flat = list(axes.flat)
    labels_np = labels.numpy()
    cmap = plt.get_cmap("tab10", max(int(labels.max()) + 1, 1))
    all_xy = torch.cat(trajectory).numpy()
    pad = 0.08 * max(float(np.ptp(all_xy[:, 0])), float(np.ptp(all_xy[:, 1])), 1.0)
    xmin, xmax = float(all_xy[:, 0].min()) - pad, float(all_xy[:, 0].max()) + pad
    ymin, ymax = float(all_xy[:, 1].min()) - pad, float(all_xy[:, 1].max()) + pad
    for step, (ax, coords, score) in enumerate(
        zip(flat[: len(trajectory)], trajectory, silhouettes, strict=True)
    ):
        xy = coords.numpy()
        for u, v in edges.t().tolist():
            ax.plot(xy[[u, v], 0], xy[[u, v], 1], color="0.75", lw=0.25, alpha=0.35, zorder=0)
        ax.scatter(xy[:, 0], xy[:, 1], c=labels_np, cmap=cmap, s=point_size, alpha=0.85, linewidths=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Diffusion time {step}\nSilhouette {score:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in flat[len(trajectory):]:
        ax.axis("off")
    fig.suptitle(f"A: oracle O(2) diffusion, init={initialization} (FakeHeteroDataset)", y=1.0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return silhouettes


def _b2_result(training: SeparabilityResult, *, steps: int) -> SeparabilityResult:
    if training.final_model is None or training.prepared_data is None:
        raise RuntimeError("B2 requires the final model and data from B1.")
    model = training.final_model
    data = training.prepared_data
    model.eval()
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
            epoch=-1,
            discovery=data.discovery,
            diffusion_overrides={"num_steps": steps, "record_trajectory": True},
        )
    if encoded.x_trajectory is None:
        raise RuntimeError("The model did not return a diffusion trajectory for B2.")
    raw_norms = tuple(float(state.norm().item()) for state in encoded.x_trajectory)
    embeddings = tuple(
        model.node_norm(state)[data.target_indices].detach().cpu()
        for state in encoded.x_trajectory
    )
    labels = data.target_y.detach().cpu()
    silhouettes = tuple(
        float(silhouette_score(embedding.numpy(), labels.numpy())) for embedding in embeddings
    )
    return SeparabilityResult(
        epochs=tuple(range(len(embeddings))),
        silhouettes=silhouettes,
        labels=labels,
        embeddings=embeddings,
        final_model=model,
        prepared_data=data,
        raw_trajectory_norms=raw_norms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A, B1, B2 embedding separability experiments.")
    parser.add_argument("--experiment", choices=["a", "b1", "b2", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/embedding_separability"))
    parser.add_argument("--method", choices=["pca", "tsne", "umap", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--oracle-steps", type=int, default=14)
    parser.add_argument("--oracle-step-size", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--avg-num-nodes", type=int, default=400)
    parser.add_argument("--avg-degree", type=float, default=5.0)
    parser.add_argument("--faces-per-anchor", type=int, default=2)
    parser.add_argument("--max-total-faces", type=int, default=4000)
    parser.add_argument("--max-total-incidences", type=int, default=12000)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument(
        "--oracle-init",
        choices=["random", "shared", "class-aligned"],
        default="random",
        help="A initialization; random is unbiased, class-aligned is a positive control.",
    )
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--no-orthogonal", action="store_true")
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
        f"edges={data.num_edges} classes={data.num_classes} "
        f"class_counts={torch.bincount(data.target_y, minlength=data.num_classes).tolist()} "
        f"device={device}",
        flush=True,
    )
    run_all = args.experiment == "all"
    if args.experiment == "a" or run_all:
        trajectory, labels, edges = oracle_o2_trajectory(
            data,
            seed=args.seed,
            steps=args.oracle_steps,
            step_size=args.oracle_step_size,
            initialization=args.oracle_init,
        )
        path = args.output_dir / "A_oracle_o2_diffusion.png"
        scores = _direct_oracle_plot(
            trajectory,
            labels,
            edges,
            output_path=path,
            columns=args.columns,
            point_size=args.point_size,
            initialization=args.oracle_init,
        )
        np.savez_compressed(path.with_suffix(".npz"), steps=np.arange(len(trajectory)), embeddings=torch.stack(trajectory).numpy(), labels=labels.numpy(), edges=edges.numpy())
        path.with_suffix(".json").write_text(json.dumps({"experiment": "A", "initialization": args.oracle_init, "silhouette": list(scores)}, indent=2) + "\n", encoding="utf-8")
        print(f"Saved A figure: {path}")
    training: SeparabilityResult | None = None
    if args.experiment in {"b1", "b2"} or run_all:
        training = train_and_capture(
            data,
            device=device,
            epochs=args.epochs,
            snapshot_every=args.snapshot_every,
            hidden_dim=args.hidden_dim,
            diffusion_steps=args.diffusion_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            use_orthogonal=not args.no_orthogonal,
            seed=args.seed,
            log_every=args.log_every,
        )
    if args.experiment in {"b1", "all"}:
        path = args.output_dir / "B1_training_separability.png"
        paths = plot_all_results(training, output_path=path, seed=args.seed, tsne_perplexity=args.tsne_perplexity, columns=args.columns, point_size=args.point_size) if args.method == "all" else (path,)
        if args.method != "all":
            from src.analysis.training_separability import plot_result
            plot_result(training, output_path=path, method=args.method, seed=args.seed, tsne_perplexity=args.tsne_perplexity, columns=args.columns, point_size=args.point_size)
        save_result(training, path)
        for figure in paths:
            print(f"Saved B1 figure: {figure}")
    if args.experiment == "b2" and training is None:
        raise RuntimeError("B2 training state was not created.")
    if args.experiment in {"b2", "all"}:
        b2 = _b2_result(training, steps=args.rollout_steps)
        path = args.output_dir / "B2_learned_diffusion_time.png"
        paths = plot_all_results(b2, output_path=path, seed=args.seed, tsne_perplexity=args.tsne_perplexity, columns=args.columns, panel_label="Diffusion step", figure_title="B2: learned H2S-NN diffusion-time rollout (FakeHeteroDataset)", point_size=args.point_size) if args.method == "all" else (path,)
        if args.method != "all":
            from src.analysis.training_separability import plot_result
            plot_result(b2, output_path=path, method=args.method, seed=args.seed, tsne_perplexity=args.tsne_perplexity, columns=args.columns, panel_label="Diffusion step", figure_title="B2: learned H2S-NN diffusion-time rollout (FakeHeteroDataset)", point_size=args.point_size)
        save_result(b2, path)
        for figure in paths:
            print(f"Saved B2 figure: {figure}")


if __name__ == "__main__":
    main()
