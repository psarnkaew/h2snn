"""Part M, M7: rerun of the F18 fingerprint probe (spec v1.10 F18; pre-registration
docs/lp_evidence/j5_preregistration.md J5.3) on the HGB evaluation lists of spec v1.15 (J5).

Same scorer and same control as `scripts/lp/lp_feature_leak.py` (rules of
docs/lp_evidence/j52_rules.json; containment of g_m(y) in the stored f(x); Ctrl-G = the rule on
x's training neighbours, read from link.dat minus link.dat.valid). Only the lists change: each
positive is ranked against the negatives of its own head (HGB groups by head), with the
tie-aware rank used by the J5 audit (mean of optimistic and pessimistic rank). Validation:
get_valid_neg (one random tail per positive); test: get_test_neigh (2-hop, seed 1). A positive
whose head has no negative is not measurable and is counted.

Gate (J5.3, fixed before J5): MRR_scorer - MRR_Ctrl-G <= 0.005 and the same for Hits@10 (the
stored features carry no information beyond the training graph); a larger difference is a stop
condition. Run from H2S-NN/.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2] / "H2S-NN"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "lp"))

import lp_feature_leak as fl  # noqa: E402
import lp_feature_rules as fr  # noqa: E402

from scripts.lp.write_hgb_negatives import TARGETS, task_config  # noqa: E402
from src.data.data_layout import REPO_DATA_ROOT, lp_derived_root  # noqa: E402
from src.data.link_prediction_dataset import (_load_local_hgb_link_data,  # noqa: E402
                                              build_link_prediction_dataset, hgb_raw_pairs)

RULES = json.loads((REPO.parent / "docs" / "lp_evidence" / "j52_rules.json").read_text(
    encoding="utf-8"))
TOL = 0.005


def grouped_metrics(pos: np.ndarray, neg: np.ndarray, s_pos: np.ndarray,
                    s_neg: np.ndarray) -> dict:
    by_head: dict[int, list[float]] = defaultdict(list)
    for (h, _), s in zip(neg, s_neg):
        by_head[int(h)].append(float(s))
    ranks, unmeasurable = [], 0
    for (h, _), s in zip(pos, s_pos):
        negs = np.array(by_head.get(int(h), []))
        if negs.size == 0:
            unmeasurable += 1
            continue
        ranks.append(0.5 * ((negs >= s).sum() + (negs > s).sum()) + 1.0)
    ranks = np.array(ranks)
    out = {"measurable_positives": int(ranks.size), "positives_without_negatives": unmeasurable}
    if ranks.size:
        out.update(mrr=float(np.mean(1.0 / ranks)), hits_at_10=float(np.mean(ranks <= 10)))
    return out


def audit(rule: dict) -> dict:
    task, type_id = rule["task"], int(rule["type"])
    lp_root = lp_derived_root(REPO_DATA_ROOT)
    spec = fr.task_spec(task, lp_root)
    g, mapping, other, release_types, rel_types, holds, amb = fl.chosen_rule(
        spec, type_id, rule["source"], rule["aggregator"])
    stored = fr.read_nodes(spec.folder)
    t_start, _ = rel_types[spec.target]
    x_is_start = type_id == t_start
    x_ids, y_ids = stored[type_id].ids, stored[other].ids
    c = fl.contribution(g, mapping)
    ctrl = fl.presence(fl.sp.csr_matrix(fl.rule_on_training(
        fl.training_pairs(spec.folder, spec.target), x_ids, y_ids, x_is_start, g, mapping,
        rule["aggregator"])))
    x_stored = stored[type_id].x

    config = task_config(REPO_DATA_ROOT, task)
    data = build_link_prediction_dataset(config)
    split = _load_local_hgb_link_data(config)
    local_to_raw = {(split.raw_to_type_name[raw], local): raw
                    for raw, local in split.raw_to_local.items()}
    src_type, _, dst_type = TARGETS[task]
    types, so, do = [src_type, dst_type], data.src_node_offset, data.dst_node_offset

    def score(pairs: np.ndarray, x_pres) -> np.ndarray:
        if pairs.size == 0:
            return np.zeros(0)
        xs_raw, ys_raw = (pairs[:, 0], pairs[:, 1]) if x_is_start else (pairs[:, 1], pairs[:, 0])
        xi, yi = np.searchsorted(x_ids, xs_raw), np.searchsorted(y_ids, ys_raw)
        assert np.array_equal(x_ids[xi], xs_raw) and np.array_equal(y_ids[yi], ys_raw)
        if x_pres is None:
            return np.zeros(len(pairs))  # dropped type (F17): every score ties
        return fl.containment(x_pres, c, xi, yi)

    out = {"task": task, "type": type_id, "rule": f"{rule['source']}/{rule['aggregator']}",
           "stored_features": x_stored is not None, "splits": {}}
    x_pres = None if x_stored is None else fl.presence(x_stored)
    for name, pos_e, neg_e in (("valid", data.val_pos_edge_index, data.val_neg_edge_index),
                               ("test", data.test_pos_edge_index, data.test_neg_edge_index)):
        pos = hgb_raw_pairs(pos_e, local_to_raw, so, do, types)
        neg = hgb_raw_pairs(neg_e, local_to_raw, so, do, types)
        res = {"scorer": grouped_metrics(pos, neg, score(pos, x_pres), score(neg, x_pres)),
               "ctrl_g": grouped_metrics(pos, neg, score(pos, ctrl), score(neg, ctrl))}
        if "mrr" in res["scorer"]:
            res["diff_mrr"] = res["scorer"]["mrr"] - res["ctrl_g"]["mrr"]
            res["diff_hits_at_10"] = res["scorer"]["hits_at_10"] - res["ctrl_g"]["hits_at_10"]
            if x_stored is None:
                # J5.3: "For a dropped type the scorer has no f(x) and is all-tie by construction;
                # its control is that level." The stored file has no feature of the type at all.
                res["gate"] = "pass (dropped type: no stored feature; scorer = all-tie level)"
            else:
                res["gate"] = ("pass" if res["diff_mrr"] <= TOL and res["diff_hits_at_10"] <= TOL
                               else "FAIL")
        else:
            res["gate"] = "not measurable (no negatives)"
        out["splits"][name] = res
    return out


results = [audit(rule) for rule in RULES]
for res in results:
    for name, r in res["splits"].items():
        s, c = r["scorer"], r["ctrl_g"]
        line = (f"{res['task']} type {res['type']} {res['rule']} {name}: measurable "
                f"{s['measurable_positives']}, without negatives {s['positives_without_negatives']}")
        if "mrr" in s:
            line += (f"; scorer MRR {s['mrr']:.4f} Hits@10 {s['hits_at_10']:.4f}; Ctrl-G MRR "
                     f"{c['mrr']:.4f} Hits@10 {c['hits_at_10']:.4f}; diff MRR {r['diff_mrr']:+.4f}"
                     f" Hits@10 {r['diff_hits_at_10']:+.4f}")
        print(line + f"; gate: {r['gate']}")
(Path(__file__).parent / "m7_leak_audit_hgb.json").write_text(json.dumps(results, indent=1),
                                                              encoding="utf-8")
sys.exit(1 if any(r["gate"] == "FAIL" for res in results for r in res["splits"].values()) else 0)
