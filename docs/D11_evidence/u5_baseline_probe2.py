"""Try to run the only vendored heterogeneous baseline (HINormer) and the sheaf baseline CSNN adapter."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from src.baselines.registry import get_adapter, NODE_CLASSIFICATION
from src.baselines.bundles import build_bundle
bundle = build_bundle(NODE_CLASSIFICATION, "dblp")
for name in ("hinormer", "csnn", "sehgnn"):
    try:
        adapter = get_adapter(name)
        params = adapter.published_params(NODE_CLASSIFICATION, "dblp")
        print(name, "published params:", None if params is None else "present")
        out = adapter.fit_task(bundle, params or {}, task=NODE_CLASSIFICATION, seed=0, device=torch.device("cpu"))
        print(name, "RAN:", out)
    except BaseException as exc:
        print(name, "DID NOT RUN:", type(exc).__name__, str(exc).splitlines()[0][:200] if str(exc) else "")
