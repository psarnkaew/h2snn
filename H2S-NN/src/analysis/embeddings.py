from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def project_2d(embeddings: torch.Tensor, method: Literal["tsne", "umap"], seed: int = 42) -> np.ndarray:
    array = embeddings.detach().cpu().numpy()
    if method == "umap":
        try:
            import umap
        except ImportError:
            print("umap-learn not installed - falling back to t-SNE.")
            method = "tsne"
    if method == "umap":
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(array)
    return TSNE(n_components=2, random_state=seed, init="pca").fit_transform(array)


def plot_representations(
    h0: torch.Tensor,
    x_star: torch.Tensor,
    labels: torch.Tensor,
    method: Literal["tsne", "umap"],
    output_path: Path,
    seed: int = 42,
) -> dict[str, float]:
    """Silhouette scores are computed on the raw (pre-projection) embeddings -
    t-SNE/UMAP distances are for visualization only and don't preserve the
    representation's actual cluster geometry."""
    import matplotlib.pyplot as plt

    labels_np = labels.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    scores: dict[str, float] = {}
    for ax, embedding, title, key in (
        (axes[0], h0, "Pre-diffusion (h0)", "pre_diffusion_silhouette"),
        (axes[1], x_star, "Post-diffusion (x*)", "post_diffusion_silhouette"),
    ):
        score = float(silhouette_score(embedding.detach().cpu().numpy(), labels_np))
        scores[key] = score
        projected = project_2d(embedding, method=method, seed=seed)
        ax.scatter(projected[:, 0], projected[:, 1], c=labels_np, cmap="tab10", s=8, alpha=0.8)
        ax.set_title(f"{title} - silhouette={score:.4f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"H2S-NN node representations on ACM ({method})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return scores
