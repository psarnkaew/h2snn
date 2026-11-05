"""D4 smoke stability run: ACM, seed 42, 20 epochs, v2 factorized k=2, d=32 (run from H2S-NN/).

Same training step as `Trainer._train_epoch` (full batch, cross-entropy on the train mask,
grad clip, AdamW + cosine from the ACM config with epochs=20). Logs per epoch: train loss,
eta, gamma, mean/max w_f, max c, and eta * lambda_max of the normalized operator
D^{-1/2} L D^{-1/2} (power iteration, eval mode, no grad). No accuracy is computed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_d4 import acm_config, load_data  # noqa: E402
from src.builders import build_optimizer, build_scheduler  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402

EPOCHS = 20
POWER_STEPS = 100


@torch.no_grad()
def lambda_max(model, data) -> float:
    model.eval()
    h0 = model.encode_nodes(data.x, data.node_type)
    operator = model.build_operator(h0, data.edge_index, data.edge_type, data.face_groups,
                                    data.num_nodes)
    generator = torch.Generator(device=h0.device).manual_seed(0)
    x = torch.randn(h0.shape, generator=generator, device=h0.device)
    x = x / x.norm()
    for _ in range(POWER_STEPS):
        y = operator.laplacian.apply_normalized(x)
        x = y / y.norm()
    return float((x * operator.laplacian.apply_normalized(x)).sum())


def main() -> None:
    cfg = acm_config("factorized", 2, 32, True, epochs=EPOCHS)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    data = load_data(cfg).to(device)
    model = build_model(cfg, data).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    targets = data.target_y[data.target_train_mask]
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(data, epoch=epoch)
        loss = torch.nn.functional.cross_entropy(output.logits[data.target_train_mask], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        lam = lambda_max(model, data)
        eta = float(model.eta())
        record = {
            "epoch": epoch, "train_loss": float(loss), "eta": eta, "gamma": float(model.gamma()),
            "w_mean": float(output.path_gate_mean), "w_max": float(output.path_gate_max),
            "max_c": float(output.max_edge_scale), "lambda_max": lam, "eta_lambda_max": eta * lam,
            "finite": bool(math.isfinite(float(loss))),
        }
        print(json.dumps(record), flush=True)
        if not record["finite"] or eta * lam >= 2.0:
            print(json.dumps({"stop": "divergence condition met", "epoch": epoch}))
            break


if __name__ == "__main__":
    main()
