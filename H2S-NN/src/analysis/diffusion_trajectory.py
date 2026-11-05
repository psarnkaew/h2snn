from __future__ import annotations

from pathlib import Path

import torch

from src.data.structures import PreparedH2SData
from src.models.h2snn_model import H2SNN


def compute_trajectory(model: H2SNN, data: PreparedH2SData, num_steps: int) -> list[torch.Tensor]:
    """Runs an illustrative diffusion rollout of `num_steps` (may exceed the
    trained K) and returns the target-node cochain at every step, including
    step 0 (pre-diffusion)."""
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
            diffusion_overrides={"num_steps": num_steps, "record_trajectory": True},
        )
    return [snapshot[data.target_indices] for snapshot in encoded.x_trajectory]


def plot_trajectory(snapshots: list[torch.Tensor], labels: torch.Tensor, output_path: Path) -> None:
    """One row of scatter panels, one per recorded diffusion step. All panels
    share a single PCA projection fit on the final snapshot so a node's
    movement across panels is visually comparable, not re-randomized per
    panel (t-SNE has no consistent out-of-sample transform)."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    labels_np = labels.detach().cpu().numpy()
    arrays = [snapshot.detach().cpu().numpy() for snapshot in snapshots]

    projector = PCA(n_components=2, random_state=42).fit(arrays[-1])
    projected = [projector.transform(array) for array in arrays]

    num_panels = len(projected)
    fig, axes = plt.subplots(1, num_panels, figsize=(2.8 * num_panels, 3.2), sharex=True, sharey=True)
    if num_panels == 1:
        axes = [axes]
    for step, (ax, coords) in enumerate(zip(axes, projected)):
        ax.scatter(coords[:, 0], coords[:, 1], c=labels_np, cmap="tab10", s=6, alpha=0.8)
        ax.set_title(f"t={step}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("H2S-NN diffusion trajectory on ACM (illustrative extended rollout, PCA-projected)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
