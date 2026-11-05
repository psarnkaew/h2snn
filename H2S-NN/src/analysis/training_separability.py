from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from src.configs.config import TrainConfig
from src.data.graph_faces import build_typed_path_face_groups
from src.data.hetero_flatten import flatten_edges, stack_node_features
from src.data.structures import PreparedH2SData
from src.models.h2snn_model import H2SNN, build_model


ProjectionMethod = Literal["pca", "tsne", "umap"]
ProjectionChoice = Literal["pca", "tsne", "umap", "all"]


@dataclass(frozen=True)
class SeparabilityResult:
    epochs: tuple[int, ...]
    silhouettes: tuple[float, ...]
    labels: torch.Tensor
    embeddings: tuple[torch.Tensor, ...]
    final_model: H2SNN | None = None
    prepared_data: PreparedH2SData | None = None
    raw_trajectory_norms: tuple[float, ...] | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stratified_masks(
    labels: torch.Tensor,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic train/validation/test masks for FakeHeteroDataset."""
    if labels.dim() != 1:
        raise ValueError("FakeHeteroDataset must provide single-label targets.")
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0:
        raise ValueError("train_ratio and val_ratio must be in (0, 1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    train = torch.zeros(labels.numel(), dtype=torch.bool)
    val = torch.zeros_like(train)
    test = torch.zeros_like(train)
    generator = torch.Generator().manual_seed(seed)
    for class_id in labels.unique(sorted=True).tolist():
        class_indices = (labels == int(class_id)).nonzero(as_tuple=False).view(-1)
        class_indices = class_indices[torch.randperm(class_indices.numel(), generator=generator)]
        num_train = max(1, int(round(train_ratio * class_indices.numel())))
        num_val = max(1, int(round(val_ratio * class_indices.numel())))
        if num_train + num_val >= class_indices.numel():
            num_val = max(0, class_indices.numel() - num_train - 1)
        train[class_indices[:num_train]] = True
        val[class_indices[num_train : num_train + num_val]] = True
        test[class_indices[num_train + num_val :]] = True
    return train, val, test


def build_fake_hetero_data(
    *,
    seed: int,
    num_classes: int,
    avg_num_nodes: int,
    avg_degree: float,
    faces_per_anchor: int,
    max_total_faces: int,
    max_total_incidences: int,
    train_ratio: float,
    val_ratio: float,
) -> PreparedH2SData:
    """Adapt PyG's FakeHeteroDataset to the current H2SNN data contract.

    PyG's fake dataset has labels on one node type and intentionally does not ship
    train/validation/test masks. The split here is only for the training loss; all
    labeled target nodes are used in the visualization, matching the manuscript.
    """
    from torch_geometric.datasets import FakeHeteroDataset

    seed_everything(seed)
    dataset = FakeHeteroDataset(
        num_graphs=1,
        num_classes=num_classes,
        avg_num_nodes=avg_num_nodes,
        avg_degree=avg_degree,
    )
    hetero_data = dataset[0]
    target_types = [
        str(node_type)
        for node_type in hetero_data.node_types
        if getattr(hetero_data[node_type], "y", None) is not None
    ]
    if len(target_types) != 1:
        raise ValueError(
            "Expected FakeHeteroDataset to have exactly one labeled node type; "
            f"found {target_types}."
        )
    target_node_type = target_types[0]

    raw_x, node_type, node_type_names, node_slices, feature_dims = stack_node_features(
        hetero_data
    )
    edge_index, edge_type, relation_names, relation_tuples = flatten_edges(
        hetero_data, node_slices
    )
    target_y = hetero_data[target_node_type].y.long()
    if target_y.numel() == 0 or int(target_y.min()) < 0 or int(target_y.max()) >= num_classes:
        raise ValueError(
            "FakeHeteroDataset labels fall outside the requested class range "
            f"[0, {num_classes}): min={int(target_y.min()) if target_y.numel() else 'empty'} "
            f"max={int(target_y.max()) if target_y.numel() else 'empty'}."
        )
    target_train_mask, target_val_mask, target_test_mask = _stratified_masks(
        target_y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    target_indices = torch.arange(
        node_slices[target_node_type][0],
        node_slices[target_node_type][1],
        dtype=torch.long,
    )
    total_nodes = raw_x.size(0)
    y = torch.full((total_nodes,), -1, dtype=torch.long)
    y[target_indices] = target_y
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    train_mask[target_indices] = target_train_mask
    val_mask[target_indices] = target_val_mask
    test_mask[target_indices] = target_test_mask

    face_groups, closure_edge_index = build_typed_path_face_groups(
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type,
        max_paths_per_anchor=faces_per_anchor,
        max_total_paths=max_total_faces,
        max_total_incidences=max_total_incidences,
        min_path_length=2,
        max_path_length=2,
        allow_repeated_node_types=True,
        exclude_identity_paths=False,
        seed=seed,
        sampling_mode="random_walk",
    )

    return PreparedH2SData(
        x=raw_x,
        y=y,
        node_type=node_type,
        node_type_names=node_type_names,
        node_slices=node_slices,
        feature_dims=feature_dims,
        edge_index=edge_index,
        edge_type=edge_type,
        relation_names=relation_names,
        relation_tuples=relation_tuples,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        target_node_type=target_node_type,
        target_indices=target_indices,
        target_y=target_y,
        target_train_mask=target_train_mask,
        target_val_mask=target_val_mask,
        target_test_mask=target_test_mask,
        # Preserve the requested class space even if one class is absent in a
        # particular random draw; the classifier must still have four outputs.
        num_classes=int(num_classes),
        is_multilabel=False,
        face_groups=face_groups,
        closure_edge_index=closure_edge_index,
        metrics=torch.zeros(0),
    )


def _build_config(
    *,
    data: PreparedH2SData,
    device: str,
    epochs: int,
    hidden_dim: int,
    diffusion_steps: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    use_orthogonal: bool,
) -> TrainConfig:
    config = TrainConfig()
    config.data.dataset_name = "fake_hetero"
    config.data.task_type = "node_classification"
    config.data.target_node_type = data.target_node_type
    config.data.metapath_templates = ()
    config.device = device
    config.epochs = epochs
    config.model.d = hidden_dim
    config.model.d_r = max(4, hidden_dim // 4)
    config.model.d_h = hidden_dim
    config.model.edge_mlp_hidden_dim = max(2 * hidden_dim, 16)
    config.model.edge_init_hidden_dim = max(2 * hidden_dim, 16)
    config.model.face_mlp_hidden_dim = max(2 * hidden_dim, 16)
    config.model.classifier_hidden_dim = max(2 * hidden_dim, 16)
    config.model.K = diffusion_steps
    config.model.encoder_dropout = dropout
    config.model.classifier_dropout = dropout
    config.model.use_orthogonal = use_orthogonal
    config.model.layer_dropout = 0.0
    config.optim.lr = learning_rate
    config.optim.weight_decay = weight_decay
    config.optim.scheduler = "none"
    return config


def train_and_capture(
    data: PreparedH2SData,
    *,
    device: str,
    epochs: int,
    snapshot_every: int,
    hidden_dim: int,
    diffusion_steps: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    use_orthogonal: bool,
    seed: int,
    log_every: int,
) -> SeparabilityResult:
    if snapshot_every <= 0:
        raise ValueError("snapshot_every must be positive.")
    if epochs < 0:
        raise ValueError("epochs must be non-negative.")

    config = _build_config(
        data=data,
        device=device,
        epochs=epochs,
        hidden_dim=hidden_dim,
        diffusion_steps=diffusion_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        use_orthogonal=use_orthogonal,
    )
    device_obj = torch.device(device)
    data = data.to(device_obj)
    model = build_model(config, data).to(device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    snapshots: list[torch.Tensor] = []
    snapshot_epochs: list[int] = []
    silhouettes: list[float] = []

    def capture_snapshot(epoch: int) -> None:
        model.eval()
        with torch.no_grad():
            snapshot_output = model(data, epoch=epoch)
        embedding = snapshot_output.x_star[data.target_indices].detach().cpu()
        labels = data.target_y.detach().cpu()
        if not torch.isfinite(embedding).all():
            raise RuntimeError(
                f"Non-finite node embedding at completed epoch {epoch}. "
                "Use the default FastHPP maps and reduce the model/dataset size if needed."
            )
        snapshots.append(embedding)
        snapshot_epochs.append(epoch)
        silhouettes.append(float(silhouette_score(embedding.numpy(), labels.numpy())))

    # Epoch 0 is the untrained baseline. Subsequent snapshots are captured only
    # after the corresponding optimizer update has completed, so the printed
    # epoch and silhouette always refer to the same model state.
    capture_snapshot(0)
    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(data, epoch=epoch)
        loss = criterion(
            output.logits[data.target_train_mask], data.target_y[data.target_train_mask]
        )
        loss.backward()
        if config.optim.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
        optimizer.step()
        completed_epoch = epoch + 1
        if completed_epoch % snapshot_every == 0 or completed_epoch == epochs:
            capture_snapshot(completed_epoch)
        if completed_epoch % max(1, log_every) == 0 or completed_epoch == epochs:
            snapshot_epoch = snapshot_epochs[-1]
            print(
                f"epoch={completed_epoch}/{epochs} loss={loss.item():.4f} "
                f"snapshot_epoch={snapshot_epoch} "
                f"snapshot_silhouette={silhouettes[-1]:.4f}",
                flush=True,
            )

    return SeparabilityResult(
        epochs=tuple(snapshot_epochs),
        silhouettes=tuple(silhouettes),
        labels=data.target_y.detach().cpu(),
        embeddings=tuple(snapshots),
        final_model=model,
        prepared_data=data,
    )


def _project(
    embedding: np.ndarray,
    *,
    method: ProjectionMethod,
    seed: int,
    pca_projector: PCA | None,
    tsne_perplexity: float,
) -> np.ndarray:
    if method == "pca":
        if pca_projector is None:
            raise ValueError("A shared PCA projector is required for PCA plots.")
        return pca_projector.transform(embedding)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError(
                "UMAP plotting requires `pip install umap-learn`; use --method pca "
                "or --method tsne otherwise."
            ) from exc
        return umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=min(15, max(2, embedding.shape[0] - 1)),
            n_epochs=200,
            n_jobs=1,
        ).fit_transform(embedding)
    perplexity = min(float(tsne_perplexity), max(2.0, (embedding.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        perplexity=perplexity,
    ).fit_transform(embedding)


def save_result(result: SeparabilityResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = np.stack([embedding.numpy() for embedding in result.embeddings], axis=0)
    arrays_to_save: dict[str, np.ndarray] = {
        "epochs": np.asarray(result.epochs, dtype=np.int64),
        "embeddings": arrays,
        "labels": result.labels.numpy(),
    }
    if result.raw_trajectory_norms is not None:
        arrays_to_save["raw_trajectory_norms"] = np.asarray(
            result.raw_trajectory_norms, dtype=np.float64
        )
    np.savez_compressed(output_path.with_suffix(".npz"), **arrays_to_save)
    metrics = {
        "epochs": list(result.epochs),
        "silhouette": list(result.silhouettes),
        "num_nodes": int(result.labels.numel()),
        "num_classes": int(result.labels.unique().numel()),
    }
    if result.raw_trajectory_norms is not None:
        metrics["raw_trajectory_norms"] = list(result.raw_trajectory_norms)
    output_path.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def plot_result(
    result: SeparabilityResult,
    *,
    output_path: Path,
    method: ProjectionMethod,
    seed: int,
    tsne_perplexity: float,
    columns: int = 5,
    panel_label: str = "Epoch",
    figure_title: str | None = None,
    point_size: float = 18.0,
) -> None:
    import matplotlib.pyplot as plt

    if not result.embeddings:
        raise ValueError("Cannot plot an empty trajectory.")
    if columns <= 0:
        raise ValueError("columns must be positive.")
    if point_size <= 0:
        raise ValueError("point_size must be positive.")
    arrays = [embedding.numpy() for embedding in result.embeddings]
    pca_projector = None
    if method == "pca":
        # One shared coordinate system makes movement across epochs comparable.
        pca_projector = PCA(n_components=2, random_state=seed).fit(np.concatenate(arrays, axis=0))
    coordinates = [
        _project(
            array,
            method=method,
            seed=seed,
            pca_projector=pca_projector,
            tsne_perplexity=tsne_perplexity,
        )
        for array in arrays
    ]

    rows = (len(coordinates) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.4 * columns, 3.2 * rows),
        squeeze=False,
        sharex=method == "pca",
        sharey=method == "pca",
    )
    labels = result.labels.numpy()
    num_classes = int(labels.max()) + 1
    cmap = plt.get_cmap("tab10", max(num_classes, 1))
    plot_axes = list(axes.flat)[: len(coordinates)]
    for ax, coords, epoch, score in zip(
        plot_axes, coordinates, result.epochs, result.silhouettes, strict=True
    ):
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, vmin=0, vmax=max(num_classes - 1, 1), s=point_size, alpha=0.75, linewidths=0)
        ax.set_title(f"{panel_label} {epoch}\nSilhouette {score:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(coordinates) :]:
        ax.axis("off")
    fig.suptitle(
        figure_title
        or (
            "Node embedding separability during H2S-NN training "
            f"(FakeHeteroDataset, {method.upper()})"
        ),
        y=1.0,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_all_results(
    result: SeparabilityResult,
    *,
    output_path: Path,
    seed: int,
    tsne_perplexity: float,
    columns: int = 5,
    panel_label: str = "Epoch",
    figure_title: str | None = None,
    point_size: float = 18.0,
) -> tuple[Path, ...]:
    """Save PCA, t-SNE, and UMAP versions of the same training trajectory."""
    output_paths: list[Path] = []
    for method in ("pca", "tsne", "umap"):
        method_output = output_path.with_name(
            f"{output_path.stem}_{method}{output_path.suffix}"
        )
        plot_result(
            result,
            output_path=method_output,
            method=method,  # type: ignore[arg-type]
            seed=seed,
            tsne_perplexity=tsne_perplexity,
            columns=columns,
            panel_label=panel_label,
            figure_title=figure_title,
            point_size=point_size,
        )
        output_paths.append(method_output)
    return tuple(output_paths)
