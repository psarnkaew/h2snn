"""L8a checks (run from H2S-NN/ in the h2snn env): the r guard of v3b_cost_smoke.load_config, the removed
--walks_per_anchor option, and the OOM path (simulated: run() raises OutOfMemoryError) -> JSON + exit 1."""
import json, sys, tempfile
import torch
import yaml
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import v3b_cost_smoke as s
for name in ("imdb", "acm", "dblp"):
    cfg = s.load_config(Path(f"configs/experiments/{name}_h2snn_v3b.yaml"), "cpu")
    print(name, "r =", cfg.data.wisp_walks_per_anchor, "OK")
payload = yaml.safe_load(open("configs/experiments/acm_h2snn_v3b.yaml"))
payload["data"]["wisp_walks_per_anchor"] = 21
bad = Path(tempfile.mkdtemp()) / "bad.yaml"
bad.write_text(yaml.safe_dump(payload))
try:
    s.load_config(bad, "cpu"); print("NOT REJECTED")
except ValueError as exc:
    print("rejected:", exc)
import argparse
sys.argv = ["x", "--config", "c", "--out", "o", "--walks_per_anchor", "21"]
try:
    s.parse_args(); print("option still accepted")
except SystemExit as exc:
    print("option removed (argparse exit", exc.code, ")")

# ---- OOM path (simulated)
calls = []
def fake_run(cfg, epochs, final):
    calls.append(cfg.data.wisp_walks_per_anchor)
    raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
s.run = fake_run
out = Path(tempfile.mkdtemp()) / "oom.json"
sys.argv = ["x", "--config", "configs/experiments/acm_h2snn_v3b.yaml", "--device", "cpu", "--out", str(out)]
code = s.main()
print("exit status", code, "| run called with r =", calls, "| json:", json.loads(out.read_text()))
