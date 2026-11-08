"""G3b (a1, a3): functional smoke checks of the D12 model (pre-registered in g3b_preregistration.md).
Run from H2S-NN/:  python ../docs/D12_evidence/g3b_smoke.py [--tag f64]
a1: ACM, seed 42, 5 epochs, defaults, production components (build_experiment_components + fit).
a3: ACM, 2 epochs, every V_k at 2x its D11 value (norm cap active), manual loop, production optimizer.
Writes docs/D12_evidence/g3b_smoke[_<tag>].json; experiment dir g3b_smoke_acm[_<tag>] (must be new:
the logger appends to metrics.jsonl). Tag f64 = the float64 norm-sum re-check (addendum)."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.builders import build_experiment_components, build_optimizer  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="")
TAG = parser.parse_args().tag
SUFFIX = f"_{TAG}" if TAG else ""
OUT = ROOT.parent / "docs" / "D12_evidence" / f"g3b_smoke{SUFFIX}.json"
raw = yaml.safe_load((ROOT / "configs" / "experiments" / "acm_h2snn.yaml").read_text(encoding="utf-8"))
raw.update(epochs=5, log_every=1, seed=42, device="cuda", exp_name=f"g3b_smoke_acm{SUFFIX}",
           output_dir="experiments/d12_g3b")
cfg = TrainConfig.from_dict(raw)
if (cfg.exp_dir / "metrics.jsonl").exists():
    raise SystemExit(f"{cfg.exp_dir / 'metrics.jsonl'} exists; the logger would append to it.")
cfg.exp_dir.mkdir(parents=True, exist_ok=True)
seed_everything(cfg.seed, cfg.device)
data, model, optimizer, _, logger, _, trainer = build_experiment_components(cfg)
report: dict = {"a1": {}, "a3": {}}

# ---------------------------------------------------------------- a1
init = {name: p.detach().clone() for name, p in model.named_parameters()}
init_W = model.filter_bank.weights().detach().clone()
init_gamma = float(model.gamma())
captured: dict = {}


def keep_context(_module, _inputs, output):
    if "c" not in captured and torch.is_grad_enabled() and output.requires_grad:
        output.retain_grad()
        captured["c"] = output


hook = model.relational_context.register_forward_hook(keep_context)
try:
    trainer.fit(data)
finally:
    hook.remove()
    logger.close()
rows = [json.loads(line) for line in (cfg.exp_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
losses = [r["train"]["loss"] for r in rows]
all_numbers = [v for r in rows for split in ("train", "val", "test") for v in r[split].values()
               if isinstance(v, (int, float))]
end_W = model.filter_bank.weights().detach().cpu()
w_rel = [float((end_W[k] - init_W[k]).norm() / init_W[k].norm()) for k in range(end_W.size(0))]
heads = {}
for head in ("listen_gate", "broadcast_gate"):
    before = torch.cat([init[f"{head}.weight"].flatten(), init[f"{head}.bias"].flatten()])
    after = torch.cat([getattr(model, head).weight.detach().flatten(), getattr(model, head).bias.detach().flatten()]).cpu()
    heads[head] = float((after - before).norm() / before.norm())
c_grad = captured["c"].grad
# reload: final state -> fresh model, CPU, eval
state_path = cfg.exp_dir / "final_state.pt"
torch.save(model.state_dict(), state_path)
fresh = build_model(cfg, data)
fresh.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
model.eval()
fresh.to("cuda").eval()
with torch.no_grad():
    gpu_a, gpu_b = model(data.to("cuda")).logits, fresh(data.to("cuda")).logits
    gpu_diff = float((gpu_a - gpu_b).abs().max())
model.to("cpu")
fresh.to("cpu")
with torch.no_grad():
    cpu_equal = bool(torch.equal(model(data).logits, fresh(data).logits))
checks = {
    "1_finite_and_loss_decreases": all(map(math.isfinite, all_numbers)) and losses[4] < losses[0],
    "2_filter_diagnostics_every_epoch": all(
        r["train"]["filter_gain"] <= 1 + 1e-5 and r["train"]["filter_norm_sum"] <= 1 + 1e-6
        and r["train"]["lambda_max_A"] <= 2 for r in rows),
    "3_every_W_k_moves": min(w_rel) > 1e-3,
    "4_gates_and_gamma_move": abs(float(model.gamma()) - init_gamma) > 1e-4 and min(heads.values()) > 1e-3,
    "5_gradient_reaches_context": c_grad is not None and bool(torch.isfinite(c_grad).all())
    and float(c_grad.norm()) > 0,
    "6_reload_reproduces_logits_cpu_exact": cpu_equal,
}
report["a1"] = {
    "losses": losses, "filter_gain": [r["train"]["filter_gain"] for r in rows],
    "filter_norm_sum": [r["train"]["filter_norm_sum"] for r in rows],
    "lambda_max_A": [r["train"]["lambda_max_A"] for r in rows],
    "W_relative_change": w_rel, "gamma": [init_gamma, float(model.gamma())],
    "gate_head_relative_change": heads,
    "g_mean": [r["train"].get("g_mean") for r in rows], "b_mean": [r["train"].get("b_mean") for r in rows],
    "context_grad_norm": None if c_grad is None else float(c_grad.norm()),
    "reload_gpu_max_abs_diff": gpu_diff,
    "final_norm_sum_float32_vs_float64": [float(torch.linalg.matrix_norm(end_W, ord=2).sum()),
                                          float(torch.linalg.matrix_norm(end_W.double(), ord=2).sum())],
    "metrics_rows": len(rows),
    "val_macro_f1_logged_only": [r["val"]["macro_f1"] for r in rows],
    "checks": checks, "pass": all(checks.values()),
}
print("a1", json.dumps(report["a1"]["checks"]), flush=True)

# ---------------------------------------------------------------- a3
torch.manual_seed(42)
device = torch.device("cuda")
gpu_data = data.to(device)
capped = build_model(cfg, data).to(device).train()
with torch.no_grad():
    capped.filter_bank.V.mul_(2.0)
opt = build_optimizer(capped, cfg)
mask, labels = gpu_data.target_train_mask, gpu_data.target_y[gpu_data.target_train_mask]
diag_args = (gpu_data.x, gpu_data.node_type, gpu_data.edge_index, gpu_data.edge_type,
             gpu_data.face_groups, gpu_data.num_nodes)
start = capped.operator_diagnostics(*diag_args)
steps = []
for epoch in range(2):
    opt.zero_grad()
    loss = F.cross_entropy(capped(gpu_data).logits[mask], labels)
    loss.backward()
    finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in capped.parameters())
    torch.nn.utils.clip_grad_norm_(capped.parameters(), cfg.optim.grad_clip)
    opt.step()
    stats = capped.operator_diagnostics(*diag_args)
    steps.append({"epoch": epoch, "loss": float(loss), "grads_finite": finite,
                  "V_norm_sum": float(capped.filter_bank.norm_sum()), **stats})
report["a3"] = {
    "start": {**start, "V_norm_sum": 2.0}, "epochs": steps,
    "checks": {"grads_finite": all(s["grads_finite"] for s in steps),
               "gain_le_1": all(s["filter_gain"] <= 1 + 1e-5 for s in [start, *steps]),
               "norm_sum_le_1": all(s["filter_norm_sum"] <= 1 + 1e-6 for s in [start, *steps])},
}
report["a3"]["pass"] = all(report["a3"]["checks"].values())
print("a3", json.dumps(report["a3"]["checks"]), flush=True)
OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
