"""U5 precondition: can a strong baseline run in this environment? (SeHGNN first, then the others.)"""
import importlib, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.baselines.runtime.vendor_paths import VENDOR_ROOTS
print("vendor clones present:", {k: (v.exists() and any(v.iterdir())) for k, v in VENDOR_ROOTS.items()})
try:
    import dgl
    print("dgl import: ok", dgl.__version__)
except Exception as exc:
    print("dgl import: FAILED -", type(exc).__name__, str(exc)[:160])
from src.baselines import registry
print("registry names:", [n for n in dir(registry) if not n.startswith("_")][:30])
