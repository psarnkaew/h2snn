"""Card 9 W6 (pre-registration v3 §4.3; built on card 7 V5): face effect F1, F2, F3, their rules and the rule-chosen
case study. Evaluation only: reads the best checkpoints of `full` and `no_path_cells` (configuration (a); `full`'s
seeds 1-4 are the main table's runs) and uses the card 5 hooks (b) `knockout` and (d) `cells_per_target`
(`src/analysis/mechanism_hooks.py`). Helpers (eta_v, eta groups, views, margins, G_low, the 4-of-5 rule):
`mechanism_common.py`. Seeds 1-5 for F1 and F2 (a run of another seed is skipped and listed); F3 on seeds 1-3.

- **F1, flips.** Per dataset and seed, every test node: `full` vs `no_path_cells` predictions; wrong->right and
  right->wrong counts per eta_v group, for the nodes without eta_v (`none`) and for all test nodes (`all`); the
  **net fix rate** of a group = (wrong->right - right->wrong) / group size. Mean and range over seeds.
- **F2, margin** (ACM, DBLP, Freebase): margin_v of `full` and of `no_path_cells` (each with its own test-x*
  centroids), raw and **scale-free** (margin_v / D, D = the mean pairwise distance between that model's class
  centroids, same seed), mean per eta_v group and over all test nodes; the scale-free difference `full -
  no_path_cells` (same test nodes). IMDB: F1 only.
- **F3, knockout** (`full`, seeds 1-3): for each test node v (at most 2,000 per dataset and seed, drawn with
  `numpy.random.default_rng(F3_SAMPLE_SEED)` without replacement when there are more), one evaluation forward with
  hook (b) on [v] (full batch: the whole evaluation graph; mini-batch: v's scope alone, base and knockout alike); node
  v only is read. Right->wrong, any prediction change, and the margin change (knockout - base, base centroids; single
  label), raw and scale-free (divided by the base model's D: the knockout keeps the base centroids), per eta_v group
  and for all sampled nodes.
- **Rules** (v3 §4.3, per dataset; booleans and counts; `None` while a seed is missing):
  F1a net fix rate of all test nodes > 0 in >= 4 of seeds 1-5; F1b net fix rate of G3 > that of G_low in >= 4 of 5;
  F2a (ACM, DBLP, Freebase) mean scale-free margin difference > 0 in >= 4 of 5; F2b that difference larger in G3 than
  in G_low in >= 4 of 5; F3 right->wrong share (all sampled nodes) > 0 in each of seeds 1-3. G_low = the lowest
  non-empty of G0, G1 (per dataset and seed). F1 answers "would a model without faces do worse"; F3 only "does the
  trained model rely on the faces through v" (v3 wording; stated separately).
- **Case study** (ACM, seed 1; no hand choice): candidates = test nodes with a different-class labelled neighbour,
  by a fact (target-target relation, ACM P-P) or by an M_2 meta-path, whose F3 knockout turns right into wrong; pick
  the largest scale-free margin drop (most negative change), ties -> the smaller node id. Shown: the node, its
  different-class neighbour (a fact neighbour if there is one, else a meta-path neighbour; the smallest id), the 3
  cells through the node with the largest w_f (hook (d); ties -> the forward's cell order) with node ids, types and
  schema; a 2-D PCA fitted on all test x* of the `full` run of h0, x* with the knockout and x* of `full`, with the
  class centroids; P(true class) with and without the knockout. JSON + one PDF; the caption says the node was
  "chosen by a fixed rule: the largest margin drop", not that it is typical.

    python scripts/analysis/face_effect.py --runs <run dir> ... --out_dir <dir> [--device cuda] \
        [--f3_seeds 1 2 3] [--f3_max_nodes 2000] [--skip_f3] [--case_dataset acm --case_seed 1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mechanism_common as mc  # noqa: E402

F1_F2_SEEDS: tuple[int, ...] = mc.RULE_SEEDS          # v3 §4.3: seeds 1-5
F3_SEEDS: tuple[int, ...] = (1, 2, 3)
F3_MAX_NODES = 2000
F3_SAMPLE_SEED = 0
CASE_DATASET, CASE_SEED = "acm", 1
CASE_CELLS = 3
ARMS: tuple[str, ...] = ("full", "no_path_cells")


# ------------------------------------------------------------------------------------------------ F1
def flip_counts(correct_full: np.ndarray, correct_ablated: np.ndarray, group: np.ndarray) -> dict[str, Any]:
    """Wrong->right (`ablated` wrong, `full` right) and right->wrong per group, `none` (nodes without eta_v) and
    `all` (every test node); the net fix rate (wrong->right - right->wrong) / nodes (`None` for an empty group)."""
    out = {}
    group = np.asarray(group)
    for name, mask in [*((n, group == q) for q, n in enumerate(mc.GROUPS)), ("none", group == -1),
                       ("all", np.ones(group.shape, dtype=bool))]:
        nodes = int(mask.sum())
        w2r = int((mask & correct_full & ~correct_ablated).sum())
        r2w = int((mask & ~correct_full & correct_ablated).sum())
        out[name] = {"nodes": nodes, "wrong_to_right": w2r, "right_to_wrong": r2w,
                     "net_fix_rate": (w2r - r2w) / nodes if nodes else None}
    return out


def group_counts(groups) -> dict[str, int]:
    """Test nodes per eta group and `none` (an empty or small G0 is reported, never merged; amendment 10)."""
    groups = np.fromiter(groups, dtype=np.int64)
    return {name: int((groups == q).sum()) for q, name in [*enumerate(mc.GROUPS), (-1, "none")]}


# ------------------------------------------------------------------------------------------------ F2
def margin_by_group(margin: np.ndarray, group: np.ndarray) -> dict[str, Any]:
    """Mean margin per eta group and over all test nodes (`all`)."""
    out = {name: (float(margin[group == q].mean()) if (group == q).any() else None)
           for q, name in enumerate(mc.GROUPS)}
    out["all"] = float(margin.mean()) if margin.size else None
    return out


def f2_entry(x: torch.Tensor, labels: torch.Tensor, centres: torch.Tensor, group: np.ndarray) -> dict[str, Any]:
    """Raw and scale-free margins of one model by group, with its centroid scale D."""
    return {"D": mc.centroid_scale(centres),
            "raw": margin_by_group(mc.margins(x, labels, centres).numpy(), group),
            "scale_free": margin_by_group(mc.scale_free_margins(x, labels, centres).numpy(), group)}


def f2_difference(f2: dict[str, Any]) -> dict[str, Any] | None:
    """Scale-free mean margin difference `full - no_path_cells` per group and over all test nodes."""
    if "full" not in f2 or "no_path_cells" not in f2:
        return None
    a, b = f2["full"]["scale_free"], f2["no_path_cells"]["scale_free"]
    return {k: (a[k] - b[k] if a[k] is not None and b[k] is not None else None) for k in a}


# ------------------------------------------------------------------------------------------------ F3
def f3_sample(test_ids: np.ndarray, max_nodes: int = F3_MAX_NODES, seed: int = F3_SAMPLE_SEED) -> np.ndarray:
    ids = np.sort(np.asarray(test_ids, dtype=np.int64))
    if ids.size <= max_nodes:
        return ids
    return np.sort(np.random.default_rng(seed).choice(ids, size=max_nodes, replace=False))


def knockout_read(run: mc.Run, node: int, base_view: mc.View | None = None) -> dict[str, Any]:
    """(base, knockout) logits and x* of node `node`: one forward each with hook (b) on [node] for the knockout. Full
    batch: the whole graph (the base forward is shared, pass `base_view`); mini-batch: the node's scope alone."""
    from src.analysis.mechanism_hooks import knockout

    view = base_view if base_view is not None else next(mc.views(run, targets=torch.tensor([node])))
    i = int((view.ids == node).nonzero()[0, 0])
    single = mc.View(view.data, view.positions[i:i + 1], view.rows[i:i + 1], view.ids[i:i + 1],
                     view.labels[i:i + 1])
    with knockout(run.model, [node]):
        ko = mc.read(run.model, single)
        betas = run.model.last_cell_betas
    return {"knockout": ko, "betas": betas}


def f3_run(run: mc.Run, base: dict[str, torch.Tensor], nodes: np.ndarray, centres: torch.Tensor | None
           ) -> list[dict[str, Any]]:
    """Per node: base / knockout correctness, prediction change, margins (single label), P(true class)."""
    rows = []
    position = {int(v): i for i, v in enumerate(base["ids"].tolist())}
    whole = None if run.minibatch else next(mc.views(run))
    for node in nodes.tolist():
        i = position[node]
        if run.minibatch:                                         # base and knockout on the same singleton scope
            b = mc.read(run.model, next(mc.views(run, targets=torch.tensor([node]))))
            base_logits, base_x = b["logits"][0], b["x_star"][0]
        else:
            base_logits, base_x = base["logits"][i], base["x_star"][i]
        ko = knockout_read(run, node, whole)["knockout"]
        label = base["labels"][i:i + 1]
        ok_base = bool(mc.correct(base_logits.view(1, -1), label, run.multilabel)[0])
        ok_ko = bool(mc.correct(ko["logits"], label, run.multilabel)[0])
        row = {"node": node, "base_correct": ok_base, "knockout_correct": ok_ko,
               "right_to_wrong": ok_base and not ok_ko,
               "prediction_changed": bool(((base_logits > 0) != (ko["logits"][0] > 0)).any()) if run.multilabel
               else int(base_logits.argmax()) != int(ko["logits"][0].argmax())}
        if centres is not None:
            m_base = float(mc.margins(base_x.view(1, -1), label, centres)[0])
            m_ko = float(mc.margins(ko["x_star"], label, centres)[0])
            sf_base = float(mc.scale_free_margins(base_x.view(1, -1), label, centres)[0])
            sf_ko = float(mc.scale_free_margins(ko["x_star"], label, centres)[0])
            p_base = float(torch.softmax(base_logits, -1)[int(label)])
            p_ko = float(torch.softmax(ko["logits"][0], -1)[int(label)])
            row.update(margin_base=m_base, margin_knockout=m_ko, margin_change=m_ko - m_base,
                       scale_free_margin_base=sf_base, scale_free_margin_knockout=sf_ko,
                       scale_free_margin_change=sf_ko - sf_base, p_true_base=p_base, p_true_knockout=p_ko)
        rows.append(row)
    return rows


def f3_summary(rows: list[dict[str, Any]], group: dict[int, int]) -> dict[str, Any]:
    """Per eta group, `none` and `all` sampled nodes: right->wrong count and share of the base-correct, prediction
    changes, the mean raw and scale-free margin change."""
    out = {}
    for name, sel in [*((n, [r for r in rows if group[r["node"]] == q]) for q, n in enumerate(mc.GROUPS)),
                      ("none", [r for r in rows if group[r["node"]] == -1]), ("all", list(rows))]:
        right = [r for r in sel if r["base_correct"]]
        changes = [r["margin_change"] for r in sel if "margin_change" in r]
        sf = [r["scale_free_margin_change"] for r in sel if "scale_free_margin_change" in r]
        out[name] = {"nodes": len(sel), "base_right": len(right),
                     "right_to_wrong": sum(r["right_to_wrong"] for r in sel),
                     "right_to_wrong_share": (sum(r["right_to_wrong"] for r in sel) / len(right)) if right else None,
                     "prediction_changed": sum(r["prediction_changed"] for r in sel),
                     "mean_margin_change": float(np.mean(changes)) if changes else None,
                     "mean_scale_free_margin_change": float(np.mean(sf)) if sf else None}
    return out


# ------------------------------------------------------------------------------------------------ rules
def _gt(a: float | None, b: float | None) -> bool | None:
    return None if a is None or b is None else bool(a > b)


def rules(dataset: str, per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """F1a, F1b, F2a, F2b (4 of seeds 1-5) and F3 (each of seeds 1-3) of one dataset (v3 §4.3)."""
    by_seed = {int(e["seed"]): e for e in per_seed}
    f1a, f1b, f2a, f2b, low = {}, {}, {}, {}, {}
    for seed in mc.RULE_SEEDS:
        e = by_seed.get(seed)
        f1 = (e or {}).get("F1")
        name = mc.g_low({g: f1[g]["nodes"] for g in mc.GROUPS}) if f1 else None
        low[seed] = name
        f1a[seed] = _gt(f1["all"]["net_fix_rate"], 0.0) if f1 else None
        f1b[seed] = _gt(f1["G3"]["net_fix_rate"], f1[name]["net_fix_rate"]) if f1 and name else None
        diff = (e or {}).get("F2_difference")
        f2a[seed] = _gt(diff["all"], 0.0) if diff else None
        f2b[seed] = _gt(diff["G3"], diff[name]) if diff and name else None
    out: dict[str, Any] = {"G_low": low, "F1a": mc.four_of_five(f1a), "F1b": mc.four_of_five(f1b)}
    if dataset in mc.SINGLE_LABEL:
        out["F2a"] = mc.four_of_five(f2a)
        out["F2b"] = mc.four_of_five(f2b)
    f3 = {}
    for seed in F3_SEEDS:
        e = by_seed.get(seed)
        share = e["F3"]["by_group"]["all"]["right_to_wrong_share"] if e and "F3" in e else None
        f3[seed] = _gt(share, 0.0)
    values = list(f3.values())
    out["F3"] = {"per_seed": f3, "count": sum(1 for v in values if v),
                 "holds": False if any(v is False for v in values) else (None if None in values else True)}
    return out


# ------------------------------------------------------------------------------------------------ case study
def fact_neighbours(data, node: int, labels: dict[int, Any]) -> list[int]:
    """Labelled target nodes joined to `node` by a fact (target-target relation; ACM P-P)."""
    ei = data.edge_index.cpu()
    out = set(ei[1, ei[0] == node].tolist()) | set(ei[0, ei[1] == node].tolist())
    out.discard(node)
    return sorted(u for u in out if u in labels)


def metapath_neighbours(ref_graph, target: str, ref, local: int) -> list[int]:
    """Target-local u != v with (u, v) in some G_P of M_2 (the eta_v convention), for one node."""
    out: set[int] = set()
    for (src, r1, mid), (_, r2, dst) in ref.h2gb_metapaths(ref_graph, target):
        second = ref_graph[(mid, r2, dst)].edge_index
        mids = second[0, second[1] == local]
        first = ref_graph[(src, r1, mid)].edge_index
        out |= set(first[0, torch.isin(first[1], mids)].tolist())
    out.discard(local)
    return sorted(out)


def different_class_neighbours(run: mc.Run, node: int, ref_graph, target: str, ref) -> dict[str, list[int]]:
    labels = mc.labelled_targets(run.data)
    own = labels[node]
    start, _ = run.data.node_slices[run.data.target_node_type]
    facts = [u for u in fact_neighbours(run.data, node, labels) if labels[u] != own]
    meta = [u + int(start) for u in metapath_neighbours(ref_graph, target, ref, node - int(start))]
    meta = [u for u in meta if u in labels and labels[u] != own]
    return {"fact": facts, "metapath": meta}


def pick_case(rows: list[dict[str, Any]], candidates: set[int]) -> dict[str, Any] | None:
    """Right->wrong under the knockout, a different-class neighbour; largest margin drop, ties -> smaller id."""
    pool = [r for r in rows if r["right_to_wrong"] and r["node"] in candidates and "scale_free_margin_change" in r]
    if not pool:
        return None
    return min(pool, key=lambda r: (r["scale_free_margin_change"], r["node"]))


def top_cells(run: mc.Run, node: int, k: int = CASE_CELLS) -> list[dict[str, Any]]:
    """The k cells through `node` with the largest w_f (hook (d), the full model's evaluation forward)."""
    from src.analysis.mechanism_hooks import cells_per_target

    view = next(mc.views(run, targets=torch.tensor([node])))
    run.model._warm_start = None
    groups = cells_per_target(run.model, view.data)
    names = run.data.node_type_names
    node_type = run.data.node_type.cpu()
    found = []
    order = 0
    for g in groups:
        ids = g["node_ids"].cpu()
        through = (ids == node).any(dim=1).nonzero().view(-1).tolist()
        for i in through:
            nodes = ids[i].tolist()
            found.append({"w": float(g["w"][i]), "beta": float(g["beta"][i]), "order": order + i,
                          "node_ids": nodes, "node_types": [names[int(node_type[v])] for v in nodes],
                          "relation_index": [int(r) for r in g["relations"][i].view(-1).tolist()],
                          "schema": str(g["schema_keys"][i]), "anchor": int(g["anchor"][i])})
        order += ids.size(0)
    found.sort(key=lambda c: (-c["w"], c["order"]))
    return found[:k]


def pca_2d(fit: np.ndarray, *points: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = fit.mean(axis=0)
    _, _, vt = np.linalg.svd(fit - mean, full_matrices=False)
    basis = vt[:2].T
    return tuple((p - mean) @ basis for p in (fit, *points))


def case_study(run: mc.Run, base: dict[str, torch.Tensor], rows: list[dict[str, Any]], out_dir: Path
               ) -> dict[str, Any]:
    ref_graph, target, ref = mc.reference_graph(run.data_root, run.dataset)
    pool = sorted(r["node"] for r in rows if r["right_to_wrong"])
    neighbours = {v: different_class_neighbours(run, v, ref_graph, target, ref) for v in pool}
    candidates = {v for v, n in neighbours.items() if n["fact"] or n["metapath"]}
    chosen = pick_case(rows, candidates)
    if chosen is None:
        return {"status": "no candidate", "right_to_wrong": len(pool)}
    v = chosen["node"]
    nb = neighbours[v]
    neighbour, via = (nb["fact"][0], "fact") if nb["fact"] else (nb["metapath"][0], "metapath")
    cells = top_cells(run, v)
    i = int((base["ids"] == v).nonzero()[0, 0])
    ko = knockout_read(run, v, None if run.minibatch else next(mc.views(run)))["knockout"]
    centres = mc.centroids(base["x_star"], base["labels"], run.data.num_classes)
    fit, h0_v, ko_v, full_v, cent = pca_2d(base["x_star"].numpy(), base["h0"][i:i + 1].numpy(),
                                           ko["x_star"].numpy(), base["x_star"][i:i + 1].numpy(), centres.numpy())
    names = run.data.node_type_names
    node_type = run.data.node_type.cpu()
    record = {"status": "chosen", "dataset": run.dataset, "seed": run.seed, "run_dir": str(run.run_dir),
              "rule": "right->wrong under the F3 knockout and a different-class neighbour; largest scale-free margin "
                      "drop; ties -> smaller id",
              "caption": "chosen by a fixed rule: the largest margin drop (not a typical node; see F1-F3)",
              "candidates": len(candidates), "node": v, "node_type": names[int(node_type[v])],
              "label": int(base["labels"][i]),
              "neighbour": {"id": neighbour, "via": via, "type": names[int(node_type[neighbour])],
                            "label": mc.labelled_targets(run.data)[neighbour]},
              "cells": cells, "margin_base": chosen["margin_base"], "margin_knockout": chosen["margin_knockout"],
              "scale_free_margin_change": chosen["scale_free_margin_change"],
              "p_true_base": chosen["p_true_base"], "p_true_knockout": chosen["p_true_knockout"],
              "pca": {"test_x_star": fit.tolist(), "test_labels": base["labels"].tolist(), "h0": h0_v[0].tolist(),
                      "x_star_knockout": ko_v[0].tolist(), "x_star_full": full_v[0].tolist(),
                      "centroids": cent.tolist()}}
    case_figure(record, out_dir / "case_study.pdf")
    return record


def case_figure(record: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4))
    v, nb = record["node"], record["neighbour"]["id"]
    nodes = [v, nb] + [u for c in record["cells"] for u in c["node_ids"] if u not in (v, nb)]
    nodes = list(dict.fromkeys(nodes))
    angle = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {u: (0.0, 0.0) if u == v else (np.cos(a), np.sin(a)) for u, a in zip(nodes, angle)}
    for k, cell in enumerate(record["cells"]):
        xs, ys = zip(*(pos[u] for u in cell["node_ids"]))
        left.plot(xs, ys, "-", color=plt.get_cmap("tab10")(k), alpha=0.8,
                  label=f"cell {k + 1}: w_f {cell['w']:.3g} ({cell['schema']})")
    left.plot(*zip(pos[v], pos[nb]), "k--", linewidth=1, label=f"different-class neighbour ({record['neighbour']['via']})")
    for u in nodes:
        left.scatter(*pos[u], s=60 if u in (v, nb) else 25, color="black" if u == v else "grey", zorder=3)
        left.annotate(str(u), pos[u], fontsize=6, xytext=(3, 3), textcoords="offset points")
    left.set_title(f"{record['dataset']} node {v} (seed {record['seed']})", fontsize=9)
    left.axis("off")
    left.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    p = record["pca"]
    fit = np.asarray(p["test_x_star"])
    right.scatter(fit[:, 0], fit[:, 1], c=p["test_labels"], s=4, alpha=0.2, cmap="tab10")
    cent = np.asarray(p["centroids"])
    right.scatter(cent[:, 0], cent[:, 1], marker="X", s=80, c=range(len(cent)), cmap="tab10", edgecolors="black")
    for key, marker, name in (("h0", "s", "h0"), ("x_star_knockout", "v", "x* knockout"), ("x_star_full", "*", "x* full")):
        right.scatter(*p[key], marker=marker, s=90, color="black", label=name)
    right.annotate("", p["x_star_full"], p["x_star_knockout"], arrowprops={"arrowstyle": "->"})
    right.set_title(f"PCA of test x*; P(true) {record['p_true_base']:.2f} -> {record['p_true_knockout']:.2f} "
                    "with the knockout", fontsize=8)
    right.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ driver
def analyse(run_dirs: list[Path], out_dir: Path, device: str = "cpu", f3_seeds: tuple[int, ...] = F3_SEEDS,
            f3_max_nodes: int = F3_MAX_NODES, skip_f3: bool = False, case: tuple[str, int] = (CASE_DATASET, CASE_SEED)
            ) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[tuple[str, int, str], Path] = {}
    skipped = []
    for path in run_dirs:
        info = mc.run_info(path)
        if info["arm"] not in ARMS or info["graph_tag"] != "original" or info["seed"] not in F1_F2_SEEDS:
            skipped.append({"run": str(path), "arm": info["arm"], "graph": info["graph_tag"], "seed": info["seed"]})
            continue
        key = (info["dataset"], info["seed"], info["arm"])
        if key in runs:
            raise ValueError(f"two runs for {key}: {runs[key]} and {path}")
        runs[key] = Path(path)
    result: dict[str, Any] = {"skipped_runs": skipped, "datasets": {}}
    for dataset in sorted({k[0] for k in runs}):
        seeds = sorted({k[1] for k in runs if k[0] == dataset})
        per_seed, eta_info = [], None
        for seed in seeds:
            pair = {arm: runs.get((dataset, seed, arm)) for arm in ARMS}
            loaded = {arm: mc.load_run(p, device) for arm, p in pair.items() if p is not None}
            full = loaded.get("full")
            if full is None:
                continue
            if eta_info is None:
                values = mc.eta(full.data_root, dataset)
                test_ids = full.data.target_indices[full.data.target_test_mask]
                test_eta = mc.eta_of_ids(values["eta"], full.data, test_ids)
                edges = mc.group_edges(test_eta)
                eta_info = {"values": values, "edges": edges,
                            "group": dict(zip(test_ids.tolist(), mc.group_of(test_eta, edges).tolist()))}
            base = {arm: mc.evaluate(r) for arm, r in loaded.items()}
            q = np.array([eta_info["group"][int(v)] for v in base["full"]["ids"].tolist()])
            entry: dict[str, Any] = {"seed": seed, "runs": {a: str(p) for a, p in pair.items() if p}}
            single = not full.multilabel
            centres = {}
            for arm, r in loaded.items():
                if single:
                    centres[arm] = mc.centroids(base[arm]["x_star"], base[arm]["labels"], r.data.num_classes)
                    entry.setdefault("F2", {})[arm] = f2_entry(base[arm]["x_star"], base[arm]["labels"],
                                                               centres[arm], q)
            if single and "no_path_cells" in loaded:
                if not torch.equal(base["full"]["ids"], base["no_path_cells"]["ids"]):
                    raise ValueError(f"{dataset} seed {seed}: the two runs read different test nodes")
                entry["F2_difference"] = f2_difference(entry["F2"])
            if "no_path_cells" in loaded:
                if not torch.equal(base["full"]["ids"], base["no_path_cells"]["ids"]):
                    raise ValueError(f"{dataset} seed {seed}: the two runs read different test nodes")
                entry["F1"] = flip_counts(mc.correct(base["full"]["logits"], base["full"]["labels"], full.multilabel),
                                          mc.correct(base["no_path_cells"]["logits"], base["no_path_cells"]["labels"],
                                                     full.multilabel), q)
            if not skip_f3 and seed in f3_seeds:
                nodes = f3_sample(base["full"]["ids"].numpy(), f3_max_nodes)
                rows = f3_run(full, base["full"], nodes, centres.get("full"))
                entry["F3"] = {"nodes": len(rows), "sampled_from": int(base["full"]["ids"].numel()),
                               "by_group": f3_summary(rows, eta_info["group"])}
                (out_dir / f"f3_{dataset}_seed{seed}.json").write_text(json.dumps(rows), encoding="utf-8")
                if (dataset, seed) == tuple(case) and single:
                    result["case_study"] = case_study(full, base["full"], rows, out_dir)
            per_seed.append(entry)
        result["datasets"][dataset] = {"eta_group_edges": eta_info["edges"] if eta_info else None,
                                       "eta_group_counts": (group_counts(eta_info["group"].values())
                                                            if eta_info else None),
                                       "test_nodes_without_eta": (sum(v == -1 for v in eta_info["group"].values())
                                                                  if eta_info else None),
                                       "seeds": per_seed, "summary": summarise(per_seed),
                                       "rules": rules(dataset, per_seed)}
    (out_dir / "face_effect.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out_dir / "face_effect.md").write_text(markdown(result), encoding="utf-8")
    return result


def summarise(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and range over seeds of every count / mean."""
    out: dict[str, Any] = {}
    names = [*mc.GROUPS, "none", "all"]
    f1 = [s["F1"] for s in per_seed if "F1" in s]
    if f1:
        out["F1"] = {n: {k: mc.mean_range([s[n][k] for s in f1]) for k in ("nodes", "wrong_to_right",
                                                                           "right_to_wrong", "net_fix_rate")}
                     for n in names}
    f2 = [s["F2"] for s in per_seed if "F2" in s]
    groups = [*mc.GROUPS, "all"]
    if f2:
        out["F2"] = {arm: {kind: {n: mc.mean_range([s[arm][kind][n] for s in f2 if arm in s]) for n in groups}
                           for kind in ("raw", "scale_free")}
                     for arm in ARMS if any(arm in s for s in f2)}
    diffs = [s["F2_difference"] for s in per_seed if s.get("F2_difference")]
    if diffs:
        out["F2_difference"] = {n: mc.mean_range([d[n] for d in diffs]) for n in groups}
    f3 = [s["F3"]["by_group"] for s in per_seed if "F3" in s]
    if f3:
        out["F3"] = {n: {k: mc.mean_range([s[n][k] for s in f3]) for k in ("nodes", "base_right", "right_to_wrong",
                                                                           "right_to_wrong_share",
                                                                           "mean_margin_change",
                                                                           "mean_scale_free_margin_change")}
                     for n in names}
    return out


def markdown(result: dict[str, Any]) -> str:
    lines = ["# Face effect F1-F3 (card 9 W6, pre-registration v3 §4.3)", "",
             "Mean [min, max] over seeds. eta groups over each dataset's test targets (amendment 10): G0 = eta_v 0, "
             "G1-G3 = tertiles of eta_v > 0 (G1 lowest); `none` = no labelled M_2 neighbour (excluded).", ""]
    for dataset, d in result["datasets"].items():
        s = d["summary"]
        lines += [f"## {dataset}", "", f"eta tertile edges t1, t2 {d['eta_group_edges']}; test nodes per group "
                  f"{d['eta_group_counts']}; test nodes without eta_v: "
                  f"{d['test_nodes_without_eta']}; seeds {[e['seed'] for e in d['seeds']]}", ""]
        if "F1" in s:
            lines += ["| F1 | nodes | wrong->right | right->wrong | net fix rate |", "|---|---|---|---|---|"]
            for n, v in s["F1"].items():
                lines.append(f"| {n} | " + " | ".join(_mr(v[k]) for k in ("nodes", "wrong_to_right",
                                                                            "right_to_wrong", "net_fix_rate")) + " |")
            lines.append("")
        groups = [*mc.GROUPS, "all"]
        if "F2" in s:
            lines += ["| F2 mean margin | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
            for arm, v in s["F2"].items():
                for kind in ("scale_free", "raw"):
                    lines.append(f"| {arm} ({kind}) | " + " | ".join(_mr(v[kind][n]) for n in groups) + " |")
            if "F2_difference" in s:
                lines.append("| full - no_path_cells (scale-free) | "
                             + " | ".join(_mr(s["F2_difference"][n]) for n in groups) + " |")
            lines.append("")
        if "F3" in s:
            lines += ["| F3 | nodes | base right | right->wrong | share | mean margin change | scale-free change |",
                      "|---" * 7 + "|"]
            for n, v in s["F3"].items():
                lines.append(f"| {n} | " + " | ".join(_mr(v[k]) for k in ("nodes", "base_right", "right_to_wrong",
                                                                            "right_to_wrong_share",
                                                                            "mean_margin_change",
                                                                            "mean_scale_free_margin_change")) + " |")
            lines.append("")
        r = d["rules"]
        lines += [f"Rules (G_low per seed {r['G_low']}): "
                  + "; ".join(f"{k} {r[k]['count']} seeds, holds {mc.fmt(r[k]['holds'])}"
                              for k in ("F1a", "F1b", "F2a", "F2b", "F3") if k in r), ""]
    case = result.get("case_study")
    if case:
        lines += ["## Case study", ""]
        if case["status"] != "chosen":
            lines.append(f"{case['status']} ({case['right_to_wrong']} right->wrong nodes)")
        else:
            lines += [f"{case['dataset']} seed {case['seed']}: node {case['node']} (label {case['label']}), "
                      f"neighbour {case['neighbour']['id']} via {case['neighbour']['via']} (label "
                      f"{case['neighbour']['label']}); margin {case['margin_base']:.4g} -> {case['margin_knockout']:.4g}; "
                      f"P(true) {case['p_true_base']:.4g} -> {case['p_true_knockout']:.4g}; {case['candidates']} "
                      "candidates.", ""]
            for c in case["cells"]:
                lines.append(f"- w_f {c['w']:.4g}: {c['node_ids']} {c['node_types']} schema `{c['schema']}`")
    return "\n".join(lines) + "\n"


def _mr(v: dict[str, Any]) -> str:
    if v["mean"] is None:
        return "—"
    return f"{mc.fmt(v['mean'])} [{mc.fmt(v['min'])}, {mc.fmt(v['max'])}]"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--f3_seeds", type=int, nargs="*", default=list(F3_SEEDS))
    parser.add_argument("--f3_max_nodes", type=int, default=F3_MAX_NODES)
    parser.add_argument("--skip_f3", action="store_true")
    parser.add_argument("--case_dataset", default=CASE_DATASET)
    parser.add_argument("--case_seed", type=int, default=CASE_SEED)
    args = parser.parse_args(argv)
    result = analyse(args.runs, args.out_dir, args.device, tuple(args.f3_seeds), args.f3_max_nodes, args.skip_f3,
                     (args.case_dataset, args.case_seed))
    print(json.dumps({d: {"seeds": [s["seed"] for s in v["seeds"]]} for d, v in result["datasets"].items()}))


if __name__ == "__main__":
    main()
