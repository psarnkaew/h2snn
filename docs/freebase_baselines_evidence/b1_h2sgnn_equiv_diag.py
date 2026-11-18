"""Diagnostic for the H2SGNN ACM-branch operator mismatch: per operator, shape, pattern equality, max |diff|."""
import sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, ".")
from tests.test_freebase_baselines import _write_acm_shaped_release
from src.baselines.adapters._hgb_vendor import load_vendor_dataset
from src.baselines.adapters._hgb_generic import hgb_generic_inputs, read_hgb_raw
with tempfile.TemporaryDirectory() as d:
    root = Path(d); folder = _write_acm_shaped_release(root)
    out = load_vendor_dataset("h2sgnn", "acm", SimpleNamespace(dataset="ACM", device=torch.device("cpu")), root)
    gen = hgb_generic_inputs(read_hgb_raw("h2sgnn", folder), {"0": "P", "1": "A", "2": "C", "3": "K"})
    for name, t, o in zip(out[2], out[9], gen["metapath_operators"]):
        T, O = t.to_dense(), o.to_dense()
        same_shape = T.shape == O.shape
        print(name, tuple(T.shape), tuple(O.shape), "pattern_eq", same_shape and torch.equal(T != 0, O != 0),
              "maxdiff", (T - O).abs().max().item() if same_shape else None, "nnz", t.nnz(), o.nnz(),
              "dtype", T.dtype, O.dtype)
