"""Kernel of the normalized operator of a trained node-classification model (plan §12.6, U4).

  python scripts/analyze_kernel.py --exp_dir experiments/<run> [--device cpu] [--k 64]
         [--cut 1e-6] [--tag best] [--out kernel.json]

1. `dim ker Delta_hat` (cut 1e-6): the k smallest eigenvalues of the PSD operator
   `Delta_hat = D^-1/2 (Delta_0 + gamma L_P) D^-1/2` by LOBPCG (matrix-free, float64). The count
   below the cut is exact when the k-th smallest eigenvalue is above the cut (`resolved`) AND
   the iteration converged (`converged`: largest residual norm of the k Ritz pairs <= 1e-4);
   otherwise it is only an estimate. LOBPCG Ritz values are upper bounds of true eigenvalues.
2. A linear probe of the labels on the projection of `h0` (and of `B h0`, whose image the fixed
   points constrain: `B x in ker Delta_hat`) onto that kernel: multinomial logistic regression
   fit on the training targets, scored on the test targets (accuracy; macro-F1 for single-label
   tasks). Reported next to a probe on `h0` itself. With an empty kernel the probe is skipped.

Evidence for the core-claim screen only; nothing here feeds training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.model_loading import load_trained_model  # noqa: E402
from src.models.h2snn_model import H2SNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tag", type=str, default="best")
    parser.add_argument("--k", type=int, default=64, help="number of smallest eigenvalues")
    parser.add_argument("--cut", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def kernel_basis(model: H2SNN, data, k: int, cut: float, maxiter: int, seed: int = 0):
    """`(eigenvalues[k], eigenvectors [N d, k], residual norms [k], operator, h0)` of `Delta_hat`
    in float64 (the k smallest, ascending)."""
    from scipy.sparse.linalg import LinearOperator, lobpcg

    model.double().eval()
    x = data.x.double()
    h0 = model.encode_nodes(x, data.node_type)
    operator = model.build_operator(h0, data.edge_index, data.edge_type, data.face_groups,
                                    data.num_nodes)
    n, d = data.num_nodes, model.d
    laplacian = operator.laplacian

    def matvec(block: np.ndarray) -> np.ndarray:
        columns = []
        for column in np.atleast_2d(block.T):
            tensor = torch.from_numpy(np.ascontiguousarray(column)).to(h0.device, h0.dtype)
            columns.append(laplacian.apply_normalized(tensor.view(n, d)).reshape(-1).cpu().numpy())
        return np.stack(columns, axis=1)

    op = LinearOperator((n * d, n * d), matvec=matvec, matmat=matvec, dtype=np.float64)
    generator = np.random.default_rng(seed)
    start = generator.standard_normal((n * d, k))
    values, vectors, residual_history = lobpcg(
        op, start, largest=False, tol=1e-9, maxiter=maxiter, retResidualNormsHistory=True)
    order = np.argsort(values)
    residuals = np.asarray(residual_history[-1])[order]
    return values[order], vectors[:, order], residuals, operator, h0


def probe(features: torch.Tensor, data, seed: int = 0) -> dict[str, float]:
    """Logistic regression on `features[target]`: train on the training targets, test on the
    test targets (single-label tasks only; multilabel is skipped)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    if data.is_multilabel:
        return {"skipped": 1.0}
    target = features[data.target_indices].cpu().numpy()
    labels = data.target_y.cpu().numpy()
    train, test = data.target_train_mask.cpu().numpy(), data.target_test_mask.cpu().numpy()
    if np.abs(target).max() == 0.0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "features_are_zero": 1.0}
    scale = np.abs(target[train]).max() or 1.0
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(target[train] / scale, labels[train])
    prediction = clf.predict(target[test] / scale)
    return {"accuracy": float(accuracy_score(labels[test], prediction)),
            "macro_f1": float(f1_score(labels[test], prediction, average="macro"))}


def main() -> None:
    args = parse_args()
    model, data, cfg = load_trained_model(args.exp_dir, args.device, tag=args.tag)
    if model.h0_only:
        raise SystemExit("h0_only_readout has no diffusion operator: no kernel to analyse.")
    if getattr(model, "coupling", "") == "pair_attention":
        raise SystemExit("pair_attention has no Laplacian operator: no kernel to analyse.")
    data = data.to(args.device)
    n, d = data.num_nodes, model.d
    k = min(args.k, n * d // 5)
    values, vectors, residuals, operator, h0 = kernel_basis(model, data, k, args.cut, args.maxiter)
    in_kernel = int((values < args.cut).sum())
    result: dict[str, object] = {
        "exp_dir": str(args.exp_dir), "num_nodes": n, "d": d, "k": k, "cut": args.cut,
        "smallest_eigenvalues": [float(v) for v in values[:min(k, 8)]],
        "kth_smallest": float(values[-1]), "dim_ker": in_kernel,
        "resolved": bool(values[-1] > args.cut),
        "max_residual": float(residuals.max()), "converged": bool(residuals.max() <= 1e-4),
    }
    result["probe_h0"] = probe(h0, data)
    if in_kernel > 0:
        basis = torch.from_numpy(vectors[:, :in_kernel]).to(h0.device, h0.dtype)

        def project(features: torch.Tensor) -> torch.Tensor:
            flat = features.reshape(-1)
            return (basis @ (basis.T @ flat)).view(n, d)

        broadcast = operator.broadcast
        weighted = h0 if broadcast is None else broadcast.unsqueeze(-1) * h0
        result["probe_projection_h0"] = probe(project(h0), data)
        result["probe_projection_b_h0"] = probe(project(weighted), data)
        result["projection_energy_fraction_h0"] = float(project(h0).square().sum()
                                                         / h0.square().sum())
    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
