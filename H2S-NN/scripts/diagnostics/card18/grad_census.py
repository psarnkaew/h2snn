"""Card 18 follow-up (design session 2026-11-22): H2G-former parameter census at the FIRST backward of a one-epoch run.

Every parameter of the model (and the LP head) is classified right after the first `loss.backward()`:
  (i) grad is None (not reached by the forward / loss), (ii) grad exists and is exactly zero, (iii) non-zero.
The graph that the forward pass before that backward received is captured with a forward pre-hook (its edge types and
edge counts), so a (i)/(ii) module keyed by an edge type can be checked against it. Defaults otherwise; no tuning.

    python scripts/diagnostics/card18/grad_census.py --task lp --dataset acm_pp --out <json>
    python scripts/diagnostics/card18/grad_census.py --task nc --dataset acm --out <json>
    python scripts/diagnostics/card18/grad_census.py --task nc --dataset ieee-cis-g --out <json>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


class FirstBackward(Exception):
    """Raised after the census: one backward is all this check needs."""


def graph_types(obj) -> dict | None:
    if not hasattr(obj, "edge_types") or not hasattr(obj, "node_types"):
        return None
    edges = {}
    for et in obj.edge_types:
        store = obj[et]
        ei = getattr(store, "edge_index", None)
        if ei is None and hasattr(store, "adj_t"):
            edges["__".join(et)] = int(store.adj_t.nnz())
        else:
            edges["__".join(et)] = int(ei.size(1)) if ei is not None else None
    nodes = {nt: int(obj[nt].num_nodes or 0) for nt in obj.node_types}
    return {"edges": edges, "nodes": nodes}


def group_of(name: str) -> str:
    """Module path without the parameter attribute, layer indices folded (`convs.0.x` -> `convs.*.x`)."""
    module = name.rsplit(".", 1)[0]
    return re.sub(r"\.\d+(?=\.|$)", ".*", module)


def edge_keys(name: str, edge_types: list[str], node_types: list[str]) -> dict:
    """Edge / node types named in a parameter path, in any of the usual key spellings."""
    found_e = [et for et in edge_types if any(sp in name for sp in
                                              (et, et.replace("__", "___"), et.replace("__", "_"),
                                               str(tuple(et.split("__")))))]
    rel = [et for et in edge_types if f".{et.split('__')[1]}." in f".{name}."]
    found_n = [nt for nt in node_types if f".{nt}." in f".{name}."]
    return {"edge_types": sorted(set(found_e + rel)), "node_types": found_n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["lp", "nc"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import src.baselines.adapters.h2gformer as h2gformer
    import src.baselines.minibatch as minibatch
    import src.baselines.training as training
    from src.baselines.adapters._h2gformer_encoders import H2GB_STACKS
    from src.baselines.bundles import build_bundle
    from src.baselines.protocol import LINK_PREDICTION, NODE_CLASSIFICATION, BaselineParams
    from src.baselines.registry import get_adapter
    from src.baselines.campaign import DEFAULT_MINIBATCH_BATCH_SIZE, batch_plan

    state: dict = {"modules": [], "graph": None, "census": None}

    def capture(original, keys):
        def wrapped(**kwargs):
            state["modules"] = [(k, kwargs[k]) for k in keys if kwargs.get(k) is not None]
            for _, module in state["modules"]:
                for sub in module.modules():   # the LP wrapper hands the graph to its inner vendor model
                    sub.register_forward_pre_hook(pre_hook, with_kwargs=True)
            return original(**kwargs)
        return wrapped

    def pre_hook(module, a, kw):
        if not module.training or state["census"] is not None:
            return
        for obj in list(a) + list(kw.values()):
            g = graph_types(obj)
            if g is not None:
                state["graph"] = g
                return

    training.train_link_prediction = capture(training.train_link_prediction, ("model", "head"))
    h2gformer.train_link_prediction = training.train_link_prediction
    h2gformer.train_full_batch_node_classification = capture(training.train_full_batch_node_classification,
                                                             ("model",))
    minibatch.train_minibatch_node_classification = capture(minibatch.train_minibatch_node_classification, ("model",))

    original_backward = torch.Tensor.backward

    def backward(self, *a, **kw):
        original_backward(self, *a, **kw)
        if state["census"] is None and state["modules"]:
            rows = []
            for prefix, module in state["modules"]:
                for name, p in module.named_parameters():
                    if not p.requires_grad:
                        cls = "frozen"
                    elif p.grad is None:
                        cls = "i_none"
                    elif not bool(p.grad.detach().abs().max() > 0):
                        cls = "ii_zero"
                    else:
                        cls = "iii_nonzero"
                    rows.append({"name": f"{prefix}.{name}", "class": cls, "numel": p.numel()})
            state["census"] = rows
            raise FirstBackward

    torch.Tensor.backward = backward

    task = LINK_PREDICTION if args.task == "lp" else NODE_CLASSIFICATION
    values = {"epochs": 1, "early_stopping": 1}
    if args.dataset not in H2GB_STACKS:
        values["metapath_pe"] = True     # H2GB datasets take H2GB's config stack; metapath_pe is not a key there
    device = torch.device(args.device)
    overrides = {"device": args.device}
    if task == NODE_CLASSIFICATION and batch_plan(task, args.dataset).mode == "minibatch":
        # as campaign._with_batch_size: a mini-batch dataset's bundle needs data.batch_size (the default B)
        overrides["data"] = {"batch_size": int(DEFAULT_MINIBATCH_BATCH_SIZE)}
    bundle = build_bundle(task, args.dataset, overrides=overrides)
    start, error = time.perf_counter(), None
    try:
        adapter = get_adapter("h2gformer")
        params = BaselineParams(values)
        adapter.fit_task(bundle, params, task=task, seed=0, device=device)
        error = "finished without a backward"
    except FirstBackward:
        pass
    except Exception as exc:
        import traceback
        error = traceback.format_exc()[-3000:]

    rows = state["census"] or []
    graph = state["graph"] or {"edges": {}, "nodes": {}}
    edge_types, node_types = list(graph["edges"]), list(graph["nodes"])
    groups: dict = defaultdict(Counter)
    keyed: dict = {}
    for row in rows:
        if row["class"] in ("i_none", "ii_zero", "frozen"):
            g = group_of(row["name"])
            groups[g][row["class"]] += 1
            keyed[g] = edge_keys(row["name"], edge_types, node_types)
    for g, k in keyed.items():
        k["edges_in_forward_graph"] = {et: graph["edges"].get(et) for et in k["edge_types"]}
    result = {"dataset": args.dataset, "task": args.task, "params": values, "error": error,
              "seconds": round(time.perf_counter() - start, 1),
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "counts": dict(Counter(r["class"] for r in rows)), "total": len(rows),
              "forward_graph": graph,
              "groups_i_ii": {g: {**dict(c), **keyed[g]} for g, c in sorted(groups.items())},
              "parameters": rows,
              "note": "card 18 follow-up: grad census at the first backward; no metric is a result"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("dataset", "error", "counts", "total", "forward_graph")}, indent=1))
    for g, c in result["groups_i_ii"].items():
        print(f"  {g}: {c}")
    sys.exit(0 if error is None and rows else 1)


if __name__ == "__main__":
    main()
