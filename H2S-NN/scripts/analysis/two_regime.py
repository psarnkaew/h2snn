"""Card 9 W7 (pre-registration v3 §4.4; built on card 7 V6): R1, R2, R3 of trained runs and their rules. Evaluation only; uses the card 5
hooks (a) `energy_at_solution`, (c) `transports_per_fact`, (d) `cells_per_target` (`src/analysis/mechanism_hooks.py`).
Helpers (eta_v, eta groups, views): `mechanism_common.py`.

- **R1, transports** (`full`): for every cell of the evaluation forward(s) whose two end nodes are labelled targets,
  the endpoint transport composed from the fact transports along the cell, as the model composes it
  (`PathTransportOperator`: T = Q_1 Q_2 ... Q_L with Q_j = Q_e on a +1 hop and Q_e^T on a -1 hop, Q_e from hook (c));
  ||T - I||_F split by same-class vs different-class end pairs (IMDB: identical genre sets): medians and the AUROC of
  ||T - I||_F separating different-class from same-class cells. ACM and Freebase (the target-target relations P-P,
  B-B): also the fact-level ||Q_e - I||_F of facts with two labelled target ends. The learned tau of the run.
- **R2, energy vs performance** (`full`): the node energy `e_tilde_v = e_v / ||x*_v||^2` (primary; e_v and x*_v
  both from hook (a), the same x* the hook returns), raw e_v beside it (`R2.raw`); for each, the AUROC separating
  wrong from correct predictions (positive = wrong: > 0.5 means wrong predictions carry more energy) and the Spearman
  correlation with eta_v.
- **R3, energy vs representation** (`full` and S2): E_tilde = E / ||x*||^2 (mini-batch: summed over the test
  batches); a linear probe (`LogisticRegression(C=1, solver="lbfgs", max_iter=1000)`, one-vs-rest per genre for IMDB)
  trained on the training targets and scored (macro-F1) on the test targets, on x* and on h0; the within/between
  class distance ratio on x* and on h0 (single label): mean distance of a test node to its class centroid / mean
  distance between two class centroids (test-node centroids).
- **Two axes:** (1) nodes: R1 by the eta_v group of the cell's start node, R2 and R3 by the test node's group
  (the probe trained once on all training targets) on the original graphs; (2) graphs: the same per perturbed graph
  (`full` and S2 on the targeted graphs, the V3 runs; h and H^2 from the graph's `graph.json`).
Runs of another arm are skipped and listed.
- **Rules** (v3 §4.4, per dataset, original graphs, seeds 1-5; booleans and counts, `None` while a seed is missing):
  R1 cell-level AUROC (different-class cells farther from I) > 0.5 in >= 4 of 5 seeds; R2 AUROC of e_tilde_v > 0.5
  in >= 4 of 5; R3 (IMDB, Freebase) (probe Macro-F1 on x* of `full` - of S2, same dataset and seed) in G3 > in G_low
  in >= 4 of 5 (G_low = the lowest non-empty of G0, G1; a missing `full` or S2 partner gives `None`). R1 is read
  together with R3 (v3 wording: never from R1 alone).

    python scripts/analysis/two_regime.py --runs <run dir> ... --out_dir <dir> [--device cuda]
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

R1_R2_ARMS: tuple[str, ...] = ("full",)
R3_ARMS: tuple[str, ...] = ("full", "S2")
FACT_LEVEL_DATASETS: tuple[str, ...] = ("acm", "freebase")
PROBE = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}
CELL_CHUNK = 2048                   # cells composed per hook (c) call
FACT_CHUNK = 8192                   # facts per hook (c) call (fact level)


# ------------------------------------------------------------------------------------------------ R1
def compose(q: torch.Tensor, facts: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """[P, d, d] endpoint transports T = Q_1 ... Q_L of P cells: `q` [k, d, d] per fact, `facts` [P, L] rows of
    `q`, `signs` [P, L] hop signs (+1: Q_e, -1: Q_e^T), multiplied left to right in path order."""
    steps = q[facts]                                                       # [P, L, d, d]
    steps = torch.where((signs < 0).view(*signs.shape, 1, 1), steps.transpose(-2, -1), steps)
    out = steps[:, 0]
    for j in range(1, steps.size(1)):
        out = out @ steps[:, j]
    return out


def distance_to_identity(t: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(t.size(-1), dtype=t.dtype, device=t.device)
    return (t - eye).flatten(1).norm(dim=1)


def split_summary(dist: np.ndarray, same: np.ndarray) -> dict[str, Any]:
    same = np.asarray(same, dtype=bool)
    return {"same": int(same.sum()), "different": int((~same).sum()),
            "median_same": float(np.median(dist[same])) if same.any() else None,
            "median_different": float(np.median(dist[~same])) if (~same).any() else None,
            "auroc_different": mc.auroc(dist, ~same)}


def r1_cells(run: mc.Run, labels: dict[int, Any], chunk: int = None) -> dict[str, np.ndarray]:
    """Per cell with two labelled target ends: start, end, length, ||T - I||_F, same class. Cells are composed
    `chunk` at a time (one hook (c) call, i.e. one forward, per chunk), so no transport of every fact is held."""
    from src.analysis.mechanism_hooks import cells_per_target, transports_per_fact
    from src.analysis.model_loading import nonempty_groups

    chunk = chunk or CELL_CHUNK
    starts, ends, lengths, dists, same = [], [], [], [], []
    for view in mc.views(run):
        run.model._warm_start = None
        rows = cells_per_target(run.model, view.data)
        groups = nonempty_groups(view.data.face_groups)
        if len(rows) != len(groups):
            raise ValueError("hook (d) and the face groups disagree")
        for row, group in zip(rows, groups):
            ids = row["node_ids"].cpu()
            keep = torch.tensor([int(a) in labels and int(b) in labels for a, b in zip(ids[:, 0], ids[:, -1])],
                                dtype=torch.bool)
            if not keep.any():
                continue
            ids, f_all, s_all = ids[keep], group.face_edge_index.cpu()[keep], group.face_edge_sign.cpu()[keep]
            for begin in range(0, ids.size(0), chunk):
                part, f, s = ids[begin:begin + chunk], f_all[begin:begin + chunk], s_all[begin:begin + chunk]
                facts = torch.unique(f.reshape(-1))
                run.model._warm_start = None
                q = transports_per_fact(run.model, view.data, facts)["Q"]
                index = torch.full((int(facts.max()) + 1,), -1, dtype=torch.long)
                index[facts] = torch.arange(facts.numel())
                t = compose(q, index[f].to(q.device), s.to(q.device))
                starts.append(part[:, 0])
                ends.append(part[:, -1])
                lengths.append(torch.full((part.size(0),), f.size(1)))
                dists.append(distance_to_identity(t.double()).cpu())
                same.append(torch.tensor([labels[int(a)] == labels[int(b)] for a, b in zip(part[:, 0], part[:, -1])],
                                         dtype=torch.bool))
    if not starts:
        return {k: np.zeros(0) for k in ("start", "end", "length", "distance", "same")}
    return {"start": torch.cat(starts).numpy(), "end": torch.cat(ends).numpy(), "length": torch.cat(lengths).numpy(),
            "distance": torch.cat(dists).numpy(), "same": torch.cat(same).numpy().astype(bool)}


def r1_facts(run: mc.Run, labels: dict[int, Any], chunk: int = None) -> dict[str, Any]:
    """Fact-level ||Q_e - I||_F of the facts with two labelled target ends (full graph), `chunk` facts per call."""
    from src.analysis.mechanism_hooks import transports_per_fact

    chunk = chunk or FACT_CHUNK
    if run.minibatch:
        return {"status": "mini-batch: not computed on the whole graph"}
    ei = run.data.edge_index.cpu()
    keep = torch.tensor([int(a) in labels and int(b) in labels for a, b in zip(ei[0], ei[1])], dtype=torch.bool)
    facts = keep.nonzero().view(-1)
    if facts.numel() == 0:
        return {"status": "no such fact"}
    dist, same = [], []
    for begin in range(0, facts.numel(), chunk):
        run.model._warm_start = None
        out = transports_per_fact(run.model, run.data, facts[begin:begin + chunk])
        dist.append(out["distance_to_identity"].double().cpu())
        same += [labels[int(a)] == labels[int(b)] for a, b in zip(out["src"].tolist(), out["tgt"].tolist())]
    return {"status": "computed", "facts": int(facts.numel()),
            **split_summary(torch.cat(dist).numpy(), np.asarray(same, dtype=bool))}


# ------------------------------------------------------------------------------------------------ R2
def normalised_node_energy(e_v: np.ndarray, x_star_v: np.ndarray) -> np.ndarray:
    """e_tilde_v = e_v / ||x*_v||^2 (v3 §4.4; `x_star_v` [n, d], the rows of hook (a)'s x* at the same nodes)."""
    x = np.asarray(x_star_v, dtype=np.float64)
    return np.asarray(e_v, dtype=np.float64) / np.square(x).sum(axis=-1)


def node_energy(run: mc.Run) -> dict[str, Any]:
    """e_v and e_tilde_v at the test targets, E and ||x*||^2 summed over the test views (hook (a))."""
    from src.analysis.mechanism_hooks import energy_at_solution

    ids, e_v, x_v, energy, norm2 = [], [], [], 0.0, 0.0
    for view in mc.views(run):
        run.model._warm_start = None
        out = energy_at_solution(run.model, view.data)
        rows = view.rows.to(out["node_energy"].device)
        e_v.append(out["node_energy"][rows].double().cpu())
        x_v.append(out["x_star"][rows].double().cpu())
        ids.append(view.ids)
        energy += float(out["energy"])
        norm2 += float(out["x_star"].double().square().sum())
    e = torch.cat(e_v).numpy()
    return {"ids": torch.cat(ids), "e_v": e, "e_tilde_v": normalised_node_energy(e, torch.cat(x_v).numpy()),
            "energy": energy, "x_norm2": norm2, "energy_tilde": energy / norm2, "tau": float(out["tau"])}


def r2(e_v: np.ndarray, wrong: np.ndarray, eta_v: np.ndarray, group: np.ndarray) -> dict[str, Any]:
    from scipy.stats import spearmanr

    defined = ~np.isnan(eta_v)
    rho = spearmanr(e_v[defined], eta_v[defined]) if defined.sum() >= 3 else None
    out = {"auroc_wrong": mc.auroc(e_v, wrong), "spearman_eta": float(rho.statistic) if rho is not None else None,
           "spearman_p": float(rho.pvalue) if rho is not None else None, "nodes": int(e_v.size),
           "wrong": int(np.asarray(wrong).sum()), "by_group": {}}
    for q, name in enumerate(mc.GROUPS):
        m = group == q
        out["by_group"][name] = {"nodes": int(m.sum()), "mean_e_v": float(e_v[m].mean()) if m.any() else None,
                                    "auroc_wrong": mc.auroc(e_v[m], np.asarray(wrong)[m]) if m.any() else None}
    return out


# ------------------------------------------------------------------------------------------------ R3
def probe(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, multilabel: bool,
          groups: np.ndarray | None = None) -> dict[str, Any]:
    """Macro-F1 (and accuracy, single label) of the linear probe on the test targets; per group with `groups`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.multiclass import OneVsRestClassifier

    clf = LogisticRegression(**PROBE)
    if multilabel:
        clf = OneVsRestClassifier(clf)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    def score(mask: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return {"nodes": 0, "macro_f1": None, "accuracy": None}
        return {"nodes": int(mask.sum()),
                "macro_f1": float(f1_score(y_test[mask], pred[mask], average="macro", zero_division=0)),
                "accuracy": None if multilabel else float(accuracy_score(y_test[mask], pred[mask]))}

    out = score(np.ones(len(y_test), dtype=bool))
    if groups is not None:
        out["by_group"] = {name: score(groups == q) for q, name in enumerate(mc.GROUPS)}
    return out


def distance_ratio(x: np.ndarray, labels: np.ndarray, num_classes: int, mask: np.ndarray | None = None
                   ) -> float | None:
    """mean ||x_v - mu_{y_v}|| over the nodes in `mask` / mean ||mu_a - mu_b|| over class pairs (all test nodes'
    centroids)."""
    centres = mc.centroids(torch.as_tensor(x), torch.as_tensor(labels), num_classes).numpy()
    present = ~np.isnan(centres).any(axis=1)
    c = centres[present]
    if len(c) < 2:
        return None
    between = np.mean([np.linalg.norm(c[a] - c[b]) for a in range(len(c)) for b in range(a + 1, len(c))])
    m = np.ones(len(labels), dtype=bool) if mask is None else mask
    if not m.any():
        return None
    within = np.linalg.norm(x[m] - centres[labels[m]], axis=1).mean()
    return float(within / between)


def r3(run: mc.Run, test: dict[str, torch.Tensor], group: np.ndarray, energy: dict[str, Any]) -> dict[str, Any]:
    train = mc.evaluate(run, "train")
    multilabel = run.multilabel
    y_tr, y_te = train["labels"].numpy(), test["labels"].numpy()
    out: dict[str, Any] = {"energy_tilde": energy["energy_tilde"]}
    for rep in ("x_star", "h0"):
        out[f"probe_{rep}"] = probe(train[rep].numpy(), y_tr, test[rep].numpy(), y_te, multilabel, group)
        if not multilabel:
            x = test[rep].numpy()
            out[f"ratio_{rep}"] = distance_ratio(x, y_te, run.data.num_classes)
            out[f"ratio_{rep}_by_group"] = {name: distance_ratio(x, y_te, run.data.num_classes, group == q)
                                              for q, name in enumerate(mc.GROUPS)}
    return out


# ------------------------------------------------------------------------------------------------ driver
def analyse_run(run: mc.Run, eta_cache: dict) -> dict[str, Any]:
    key = (str(run.data_root.resolve()), run.dataset)
    if key not in eta_cache:
        eta_cache[key] = mc.eta(run.data_root, run.dataset)
    eta_values = eta_cache[key]["eta"]
    test = mc.evaluate(run)
    test_eta = mc.eta_of_ids(eta_values, run.data, test["ids"])
    edges = mc.group_edges(test_eta)
    group = mc.group_of(test_eta, edges)
    record = run.graph_record() or {}
    out: dict[str, Any] = {"run": str(run.run_dir), "dataset": run.dataset, "seed": run.seed, "arm": run.arm,
                           "graph": run.graph_tag, "h": record.get("h"), "h2_index": record.get("h2_index"),
                           "eta_group_edges": edges, "test_nodes_without_eta": int((group == -1).sum()),
                           "eta_group_counts": {name: int((group == q).sum()) for q, name in enumerate(mc.GROUPS)}}
    energy = node_energy(run)
    if not torch.equal(energy["ids"], test["ids"]):
        raise ValueError(f"{run.run_dir}: energy and prediction views read different nodes")
    out["tau"] = float(run.model.tau()) if run.model.tau() is not None else None
    if run.arm in R1_R2_ARMS:
        labels = mc.labelled_targets(run.data)
        cells = r1_cells(run, labels)
        start_eta = mc.eta_of_ids(eta_values, run.data, cells["start"]) if cells["start"].size else np.zeros(0)
        start_q = mc.group_of(start_eta, edges)
        out["R1"] = {"cells": split_summary(cells["distance"], cells["same"]) if cells["start"].size else None,
                     "by_group": {name: split_summary(cells["distance"][start_q == q], cells["same"][start_q == q])
                                     for q, name in enumerate(mc.GROUPS) if (start_q == q).any()},
                     "by_length": {int(n): split_summary(cells["distance"][cells["length"] == n],
                                                         cells["same"][cells["length"] == n])
                                   for n in np.unique(cells["length"])}}
        if run.dataset in FACT_LEVEL_DATASETS:
            out["R1"]["facts"] = r1_facts(run, labels)
        wrong = ~mc.correct(test["logits"], test["labels"], run.multilabel)
        out["R2"] = {"energy": "e_tilde_v = e_v / ||x*_v||^2", **r2(energy["e_tilde_v"], wrong, test_eta, group),
                     "raw": r2(energy["e_v"], wrong, test_eta, group)}
    if run.arm in R3_ARMS:
        out["R3"] = r3(run, test, group, energy)
    return out


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and range over seeds per (dataset, graph, arm) of the headline numbers."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["dataset"], r["graph"], r["arm"]), []).append(r)

    def get(r: dict[str, Any], *path: str) -> Any:
        for p in path:
            if r is None:
                return None
            r = r.get(p) if isinstance(r, dict) else None
        return r

    fields = {"tau": ("tau",), "R1_median_same": ("R1", "cells", "median_same"),
              "R1_median_different": ("R1", "cells", "median_different"),
              "R1_auroc_different": ("R1", "cells", "auroc_different"),
              "R1_fact_auroc_different": ("R1", "facts", "auroc_different"),
              "R2_auroc_wrong": ("R2", "auroc_wrong"), "R2_spearman_eta": ("R2", "spearman_eta"),
              "R2_raw_auroc_wrong": ("R2", "raw", "auroc_wrong"), "R2_raw_spearman_eta": ("R2", "raw", "spearman_eta"),
              "R3_energy_tilde": ("R3", "energy_tilde"), "R3_probe_x_star": ("R3", "probe_x_star", "macro_f1"),
              "R3_probe_h0": ("R3", "probe_h0", "macro_f1"), "R3_ratio_x_star": ("R3", "ratio_x_star"),
              "R3_ratio_h0": ("R3", "ratio_h0")}
    out = []
    for (dataset, graph, arm), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        row = {"dataset": dataset, "graph": graph, "arm": arm, "seeds": sorted(r["seed"] for r in rs),
               "h": rs[0]["h"], "h2_index": rs[0]["h2_index"]}
        for name, path in fields.items():
            row[name] = mc.mean_range([get(r, *path) for r in rs])
        out.append(row)
    return out


R3_DATASETS: tuple[str, ...] = ("imdb", "freebase")


def _gt(a: float | None, b: float | None) -> bool | None:
    return None if a is None or b is None else bool(a > b)


def _probe_g(record: dict[str, Any] | None, name: str | None) -> float | None:
    if record is None or name is None or "R3" not in record:
        return None
    return (record["R3"]["probe_x_star"].get("by_group") or {}).get(name, {}).get("macro_f1")


def rules(records: list[dict[str, Any]]) -> dict[str, Any]:
    """R1, R2 (every dataset) and R3 (IMDB, Freebase) over seeds 1-5 on the original graphs (v3 §4.4)."""
    by = {(r["dataset"], r["arm"], int(r["seed"])): r for r in records if r["graph"] == "original"}
    out: dict[str, Any] = {}
    for dataset in sorted({k[0] for k in by}):
        r1, r2_, r3, low = {}, {}, {}, {}
        for seed in mc.RULE_SEEDS:
            full, s2 = by.get((dataset, "full", seed)), by.get((dataset, "S2", seed))
            cells = (full or {}).get("R1", {}).get("cells") if full else None
            r1[seed] = _gt(cells["auroc_different"], 0.5) if cells else None
            r2_[seed] = _gt(full["R2"]["auroc_wrong"], 0.5) if full and "R2" in full else None
            name = mc.g_low(full["eta_group_counts"]) if full else None
            low[seed] = name
            if dataset in R3_DATASETS:
                gain = {g: (None if _probe_g(full, g) is None or _probe_g(s2, g) is None
                            else _probe_g(full, g) - _probe_g(s2, g)) for g in ("G3", name)}
                r3[seed] = _gt(gain["G3"], gain[name]) if name else None
        out[dataset] = {"G_low": low, "R1": mc.four_of_five(r1), "R2": mc.four_of_five(r2_)}
        if dataset in R3_DATASETS:
            out[dataset]["R3"] = mc.four_of_five(r3)
    return out


def markdown(summary: list[dict[str, Any]], skipped: list[dict[str, Any]], rule_result: dict[str, Any] | None = None
             ) -> str:
    cols = ["tau", "R1_median_same", "R1_median_different", "R1_auroc_different", "R1_fact_auroc_different",
            "R2_auroc_wrong", "R2_spearman_eta", "R2_raw_auroc_wrong", "R2_raw_spearman_eta", "R3_energy_tilde",
            "R3_probe_x_star", "R3_probe_h0", "R3_ratio_x_star", "R3_ratio_h0"]
    lines = ["# Two-regime analysis R1-R3 (card 9 W7, pre-registration v3 §4.4)", "",
             "Mean [min, max] over seeds per (dataset, graph, arm). R1 AUROC: ||T - I||_F separating different-class "
             "from same-class cells (facts); R2 AUROC: e_tilde_v = e_v / ||x*_v||^2 separating wrong from correct test "
             "predictions (R2_raw: raw e_v). R1 is read with R3, never alone.", "",
             "| dataset | graph | h | arm | seeds | " + " | ".join(cols) + " |", "|---" * (5 + len(cols)) + "|"]
    for row in summary:
        h = mc.fmt(row["h"]) if row["h"] is not None else "—"
        cells = []
        for c in cols:
            v = row[c]
            cells.append("—" if v["mean"] is None else f"{mc.fmt(v['mean'])} [{mc.fmt(v['min'])}, {mc.fmt(v['max'])}]")
        lines.append(f"| {row['dataset']} | {row['graph']} | {h} | {row['arm']} | {len(row['seeds'])} | "
                     + " | ".join(cells) + " |")
    if rule_result:
        lines += ["", "| dataset | rule | seeds true | holds |", "|---|---|---|---|"]
        for dataset, r in rule_result.items():
            for name in ("R1", "R2", "R3"):
                if name in r:
                    lines.append(f"| {dataset} | {name} | {r[name]['count']}/{len(r[name]['seeds_with_value'])} | "
                                 f"{mc.fmt(r[name]['holds'])} |")
    if skipped:
        lines += ["", f"Skipped runs (arm not in R1-R3): {len(skipped)}"]
    return "\n".join(lines) + "\n"


def analyse(run_dirs: list[Path], out_dir: Path, device: str = "cpu") -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)
    records, skipped, eta_cache = [], [], {}
    for path in run_dirs:
        info = mc.run_info(path)
        if info["arm"] not in set(R1_R2_ARMS) | set(R3_ARMS):
            skipped.append({"run": str(path), "arm": info["arm"]})
            continue
        run = mc.load_run(path, device)
        record = analyse_run(run, eta_cache)
        name = f"{run.dataset}__{run.graph_tag}__{run.arm}__seed{run.seed}.json"
        (out_dir / "runs" / name).write_text(json.dumps(record, indent=1), encoding="utf-8")
        records.append(record)
        del run
    summary = aggregate(records)
    rule_result = rules(records)
    result = {"summary": summary, "rules": rule_result, "skipped_runs": skipped, "runs": len(records)}
    (out_dir / "two_regime.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out_dir / "two_regime.md").write_text(markdown(summary, skipped, rule_result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = analyse(args.runs, args.out_dir, args.device)
    print(f"{result['runs']} runs analysed, {len(result['skipped_runs'])} skipped -> {args.out_dir}")


if __name__ == "__main__":
    main()
