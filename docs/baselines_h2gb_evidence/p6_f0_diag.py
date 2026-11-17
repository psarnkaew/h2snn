"""Card 1e F0: diagnose the three Freebase sparse failures on real data (no metric is read or printed).

    python -X faulthandler ../docs/baselines_h2gb_evidence/p6_f0_diag.py {han_nc|han_lp|h2sgnn_lp}

One epoch through `campaign.fit_cell` (the dry run's path) at the dry run's point (cells.csv; HAN LP at the step-down
num_heads = 4, where it segfaulted). Before each candidate sparse call the operands are logged (shape, nnz, index
dtypes, device, coalesced/sorted state, the largest index / row-pointer value, nnz x dense width) against 2^31 - 1,
with a flush, so that the last block before a fault is the failing call; `-X faulthandler` prints its Python frame.
No native debugger exists on the cluster (gdb/lldb/eu-stack absent), so native frames are not available.
"""

from __future__ import annotations

import faulthandler
import functools
import math
import os
import sys
import time
from typing import Any

import torch

sys.path.insert(0, os.getcwd())  # run from H2S-NN/
faulthandler.enable(all_threads=True)

INT32_MAX = 2**31 - 1
COUNTS: dict[str, int] = {}
LIMIT_PER_CALL = 40  # log the first calls of each kind in full, then every 50th


def log(*parts: Any) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def over(value: int) -> str:
    return f"{value:,}" + (" > 2^31-1 !" if value > INT32_MAX else "")


def describe(name: str, t: Any) -> str:
    try:
        from torch_sparse import SparseTensor
    except ImportError:  # pragma: no cover
        SparseTensor = ()
    if isinstance(t, SparseTensor):
        s = t.storage
        rowptr = s._rowptr
        parts = [f"{name}: SparseTensor sizes={tuple(t.sizes())} nnz={over(int(t.nnz()))} device={s._col.device}",
                 f"col={s._col.dtype} max={int(s._col.max()) if s._col.numel() else -1}"]
        if rowptr is not None:
            parts.append(f"rowptr={rowptr.dtype} last={over(int(rowptr[-1]))}")
        if s._row is not None:
            parts.append(f"row={s._row.dtype} max={int(s._row.max()) if s._row.numel() else -1}")
        parts.append(f"value={None if s._value is None else (s._value.dtype, tuple(s._value.shape))}")
        parts.append(f"cached colptr={s._colptr is not None} csr2csc={s._csr2csc is not None}")
        return " ".join(parts)
    if isinstance(t, torch.Tensor):
        if t.layout == torch.sparse_coo:
            idx = t._indices()
            return (f"{name}: COO shape={tuple(t.shape)} nnz={over(int(t._nnz()))} indices={idx.dtype} "
                    f"max={int(idx.max()) if idx.numel() else -1} coalesced={t.is_coalesced()} device={t.device} "
                    f"values={t._values().dtype}")
        if t.layout == torch.sparse_csr:
            crow, col = t.crow_indices(), t.col_indices()
            return (f"{name}: CSR shape={tuple(t.shape)} nnz={over(int(t._nnz()))} crow={crow.dtype} "
                    f"last={over(int(crow[-1]))} col={col.dtype} max={int(col.max()) if col.numel() else -1} "
                    f"device={t.device}")
        return f"{name}: dense shape={tuple(t.shape)} dtype={t.dtype} device={t.device} numel={over(t.numel())}"
    return f"{name}: {type(t).__name__}"


def gpu_memory() -> str:
    if not torch.cuda.is_available() or os.environ.get("P6_DEVICE", "cuda") != "cuda":
        return ""
    free, total = torch.cuda.mem_get_info()
    return (f" | gpu allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB "
            f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB free={free / 2**30:.2f} of {total / 2**30:.2f} GiB")


def should_log(kind: str) -> bool:
    COUNTS[kind] = COUNTS.get(kind, 0) + 1
    n = COUNTS[kind]
    return n <= LIMIT_PER_CALL or n % 50 == 0


def instrument() -> None:
    import importlib

    tsm = importlib.import_module("torch_sparse.matmul")  # `torch_sparse.matmul` the attribute is the function

    original_spmm_sum = tsm.spmm_sum

    @functools.wraps(original_spmm_sum)
    def spmm_sum(src, other):
        if should_log("spmm_sum"):
            width = int(other.shape[-1]) if other.dim() > 1 else 1
            log(f"CALL torch_sparse spmm_sum #{COUNTS['spmm_sum']} (matmul.py:27) other.requires_grad="
                f"{other.requires_grad}")
            log("  ", describe("src", src))
            log("  ", describe("other", other))
            log(f"   nnz x width = {over(int(src.nnz()) * width)}; rows x width = {over(int(src.sizes()[0]) * width)}")
        out = original_spmm_sum(src, other)
        if out.requires_grad and should_log("spmm_sum_bw_hook"):
            n = COUNTS["spmm_sum"]
            out.register_hook(lambda g, n=n: log(f"BACKWARD into spmm_sum #{n}: grad {describe('grad', g)}") or None)
        return out

    tsm.spmm_sum = spmm_sum

    original_mm = torch.sparse.mm

    @functools.wraps(original_mm)
    def sparse_mm(a, b, *args, **kwargs):
        if should_log("torch.sparse.mm"):
            log(f"CALL torch.sparse.mm #{COUNTS['torch.sparse.mm']}")
            log("  ", describe("a", a))
            log("  ", describe("b", b))
        out = original_mm(a, b, *args, **kwargs)
        if should_log("torch.sparse.mm.out"):
            log("   ->", describe("out", out))
        return out

    torch.sparse.mm = sparse_mm

    original_linear = torch.nn.functional.linear

    @functools.wraps(original_linear)
    def linear(x, weight, bias=None):
        if x.is_sparse and should_log("linear_sparse"):
            log(f"CALL F.linear on a sparse input #{COUNTS['linear_sparse']}")
            log("  ", describe("input", x))
            log("  ", describe("weight", weight))
        return original_linear(x, weight, bias)

    torch.nn.functional.linear = linear

    try:
        import dgl
        from dgl.nn.pytorch import GATConv

        original_reach = dgl.metapath_reachable_graph

        @functools.wraps(original_reach)
        def reach(g, metapath):
            log(f"CALL dgl.metapath_reachable_graph {metapath} on {g.device}")
            out = original_reach(g, metapath)
            log(f"   -> nodes={out.num_nodes():,} edges={over(out.num_edges())} idtype={out.idtype}")
            return out

        dgl.metapath_reachable_graph = reach
        original_gat = GATConv.forward

        def gat_forward(self, graph, feat, *args, **kwargs):
            logged = should_log("GATConv")
            n = COUNTS["GATConv"]
            if logged:
                heads, width = self._num_heads, self._out_feats
                log(f"CALL GATConv.forward #{n} graph nodes={graph.num_nodes():,} "
                    f"edges={over(graph.num_edges())} idtype={graph.idtype} device={graph.device} heads={heads} "
                    f"out={width}; edges x heads x out = {over(graph.num_edges() * heads * width)}; "
                    f"edges x heads = {over(graph.num_edges() * heads)}")
                log("  ", describe("feat", feat))
            out = original_gat(self, graph, feat, *args, **kwargs)
            if logged:
                log(f"   GATConv.forward #{n} returned", describe("out", out) + gpu_memory())
                if out.requires_grad:
                    out.register_hook(lambda g, n=n: log(f"BACKWARD into GATConv #{n}: {describe('grad', g)}") or None)
            return out

        GATConv.forward = gat_forward

        import dgl.nn.pytorch.conv.gatconv as gatconv_module

        original_softmax = gatconv_module.edge_softmax

        def edge_softmax(graph, logits, *args, **kwargs):
            logged = should_log("edge_softmax")
            if logged:
                log(f"CALL edge_softmax #{COUNTS['edge_softmax']} edges={over(graph.num_edges())}",
                    describe("logits", logits) + gpu_memory())
            out = original_softmax(graph, logits, *args, **kwargs)
            if logged:
                log(f"   edge_softmax #{COUNTS['edge_softmax']} returned")
                if out.requires_grad:
                    n = COUNTS["edge_softmax"]
                    out.register_hook(lambda g, n=n: log(f"BACKWARD into edge_softmax #{n}") or None)
            return out

        gatconv_module.edge_softmax = edge_softmax

        for method in ("update_all", "apply_edges"):
            original_method = getattr(dgl.DGLGraph, method)

            def wrapped(self, *args, _name=method, _orig=original_method, **kwargs):
                logged = should_log(_name)
                if logged:
                    log(f"CALL DGLGraph.{_name} #{COUNTS[_name]} edges={over(self.num_edges())} "
                        f"formats={self.formats()} device={self.device}")
                result = _orig(self, *args, **kwargs)
                if logged:
                    log(f"   DGLGraph.{_name} #{COUNTS[_name]} returned")
                return result

            setattr(dgl.DGLGraph, method, wrapped)
    except ImportError:  # pragma: no cover
        pass


POINTS = {  # docs/dryrun_evidence/cells.csv (branch dry-run), column `point`
    "han_lp": ("han", "link_prediction", "freebase_bb",
               {"decoder_type": "dot", "dropout": 0.45, "embed_dim": 128, "hidden_dim": 256, "lr": 0.001,
                "num_heads": 4, "weight_decay": 0.0}),  # step-down num_heads = 4 (d2_cells_stepdown_24gb.txt:23)
    "h2sgnn_lp": ("h2sgnn", "link_prediction", "freebase_bb",
                  {"K": 6, "alpha": 0.1, "alpha_list": 0.1, "beta_lr": 1e-05, "beta_wd": 0.0, "decoder_type": "dot",
                   "dropout": 0.45, "emb_dim": 256, "embed_dim": 128, "hidden": 256, "input_drop": 0.45,
                   "lr": 0.001, "poly": "GPR_GPR", "temp_lr": 0.001, "temp_wd": 0.1, "weight_decay": 0.0}),
}


def dry_run_point(space: dict[str, Any]) -> dict[str, Any]:
    """`d2_dryrun.baseline_point` (dry-run branch, d2_dryrun.py:291-319): the max of a size axis, else the first value,
    the geometric midpoint of a log-uniform range, the midpoint of a uniform one."""
    size_axes = {"hidden_dim", "hidden", "embed_dim", "emb_dim", "num_layers", "num_heads", "K", "batch_size"}
    values: dict[str, Any] = {}
    for name, spec in space.items():
        if "values" in spec:
            values[name] = max(spec["values"]) if name in size_axes else list(spec["values"])[0]
        elif "value" in spec:
            values[name] = spec["value"]
        elif spec.get("distribution") == "log_uniform_values":
            values[name] = math.sqrt(spec["min"] * spec["max"])
        else:
            values[name] = (spec["min"] + spec["max"]) / 2
    return values


def main() -> int:
    which = sys.argv[1]
    from src.baselines.campaign import Cell, build_raw_base, fit_cell
    from src.baselines.registry import get_adapter
    from src.utils.wandb_campaign import CampaignUnit

    if which == "han_nc":
        model, task, dataset = "han", "node_classification", "freebase"
        params = dry_run_point(get_adapter(model).search_space(task, dataset))
    else:
        model, task, dataset, params = POINTS[which]
    log(f"cell {model}|{task}|{dataset} params={params} torch={torch.__version__} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None} CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    instrument()
    unit = CampaignUnit(run_index=0, num_runs=1, seed=0, label="run")
    device = os.environ.get("P6_DEVICE", "cuda")  # cpu: the device HGB's HAN README uses on Freebase
    log(f"device={device}")
    raw_base = build_raw_base(Cell(model, task, dataset), unit, device=device, epochs=1)
    start = time.perf_counter()
    try:
        fit_cell(raw_base, params, source="tuned", device=torch.device(device))
    except Exception as exc:  # the diagnosis is the output; no metric is printed either way
        import traceback

        traceback.print_exc()
        log(f"RESULT exception {type(exc).__name__} after {time.perf_counter() - start:.1f} s")
        return 1
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    log(f"RESULT completed one epoch in {time.perf_counter() - start:.1f} s; gpu peak allocated {peak:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
