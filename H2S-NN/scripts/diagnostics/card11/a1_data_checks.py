"""Card 11 A1: LP data checks (no training), acm_pa and dblp_ap side by side.

Builds the prepared LP data exactly as the trainer sees it (`build_link_prediction_dataset` on the
reference trial's config) and writes, per task, to `--out`:

- `a1_counts.tsv`: positives / negatives per split, head and tail types, node counts;
- `a1_leakage.tsv`: per split, positives present in the message graph `data.edge_index` in the
  stored orientation (any relation, the target relation), reversed (any relation, the reverse
  relation), duplicates inside the split, overlap with the other splits' positives; the same for
  every split's negatives that are known positives;
- `a1_degree.tsv`: message-graph degree (all relations; target relation only) of head and tail for
  every split's positives and negatives: quantiles, mean, share with degree 0;
- `a1_features.tsv`: input width per node type and whether it gets a learned per-node embedding;
- `a1_degree_auroc.tsv`: AUROC of the degree-only score `log(1+deg_u) + log(1+deg_v)` (all
  relations and target relation only; also the head-only and tail-only terms) on train (positives vs
  the epoch-0 training negatives), validation, test 2-hop and test random.

Degree = number of message-graph facts at the node, either end (`edge_index` is the tied fact
layer: each fact once, in its kept relation's orientation). "target pair" = the kept relation(s)
between the head and tail types (acm_pa: author-rel_3-paper, the target's rho partner).
AUROC is `src.utils.misc.compute_link_prediction_metrics` (the trainer's own metric).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import TrainConfig  # noqa: E402
from src.data.hgb_lp_negatives import hgb_epoch_train_negatives  # noqa: E402
from src.data.link_prediction_dataset import build_link_prediction_dataset  # noqa: E402
from src.utils.misc import compute_link_prediction_metrics, lp_test_sets  # noqa: E402

QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def keys(edge_index: torch.Tensor, n: int) -> np.ndarray:
    return (edge_index[0].long() * n + edge_index[1].long()).cpu().numpy()


def splits(data) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """split -> (positives, negatives)."""
    train_neg, _ = hgb_epoch_train_negatives(data, 0)
    sets = lp_test_sets(data)
    return {
        "train": (data.train_pos_edge_index, train_neg),
        "val": (data.val_pos_edge_index, data.val_neg_edge_index),
        "test_2hop": (data.test_pos_edge_index, sets["2hop"][0]),
        "test_random": (data.test_pos_edge_index, sets["random"][0]),
    }


def run_task(task: str, cfg_path: Path, out: dict[str, list]) -> None:
    raw = yaml.safe_load(cfg_path.read_text())
    raw["use_wandb"] = False
    cfg = TrainConfig.from_dict(raw)
    data = build_link_prediction_dataset(cfg)
    n = data.num_nodes
    head_t, rel, tail_t = data.target_relation
    rel_tuples = list(data.relation_tuples)
    # The message graph is the tied fact layer: every fact once, in the orientation of its kept
    # relation (acm_pa keeps author-rel_3-paper for the target pair, dblp_ap author-rel_0-paper).
    pair_ids = [i for i, t in enumerate(rel_tuples)
                if (t[0], t[2]) in ((head_t, tail_t), (tail_t, head_t))]
    ei, et = data.edge_index, data.edge_type
    deg_all = (torch.bincount(ei[0], minlength=n) + torch.bincount(ei[1], minlength=n)).numpy()
    tmask = torch.isin(et, torch.tensor(pair_ids, dtype=et.dtype))
    deg_tgt = (torch.bincount(ei[0][tmask], minlength=n) + torch.bincount(ei[1][tmask], minlength=n)).numpy()
    msg_all = set(keys(ei, n).tolist())
    msg_tgt = set(keys(ei[:, tmask], n).tolist())
    msg_rev = msg_tgt  # the target pair's facts, stored orientation (reverse rows checked as (v, u))
    sp = splits(data)
    pos_sets = {s: set(keys(p, n).tolist()) for s, (p, _) in sp.items()}
    known = set(keys(data.known_pos_edge_index, n).tolist())

    out["counts"].append({
        "task": task, "head_type": head_t, "relation": rel, "tail_type": tail_t,
        "num_heads": data.num_src_nodes, "num_tails": data.num_dst_nodes,
        "message_edges": int(ei.size(1)), "message_edges_target_rel": int(tmask.sum()),
        "relations": " ".join(f"{i}:{'/'.join(t)}" for i, t in enumerate(rel_tuples)),
        "target_pair_kept_relations": " ".join("/".join(rel_tuples[i]) for i in pair_ids),
        **{f"{s}_pos": int(p.size(1)) for s, (p, _) in sp.items() if s != "test_random"},
        **{f"{s}_neg": int(q.size(1)) for s, (_, q) in sp.items()},
        "test_heads_without_2hop_neg": int(data.test_heads_without_negatives.numel()
                                           if data.test_heads_without_negatives is not None else 0),
    })

    for s, (p, q) in sp.items():
        if s == "test_random":
            pk = None
        else:
            pk = keys(p, n)
            rk = keys(p.flip(0), n)
            uniq = len(set(pk.tolist()))
            if s == "train":
                others = pos_sets["val"] | pos_sets["test_2hop"]
            elif s == "val":
                others = pos_sets["train"] | pos_sets["test_2hop"]
            else:
                others = pos_sets["train"] | pos_sets["val"]
            out["leak"].append({
                "task": task, "split": s, "kind": "positives", "n": len(pk),
                "in_message_graph_any_rel": int(sum(k in msg_all for k in pk)),
                "in_message_graph_target_pair": int(sum(k in msg_tgt for k in pk)),
                "reversed_in_message_graph_any_rel": int(sum(k in msg_all for k in rk)),
                "reversed_in_target_pair": int(sum(k in msg_rev for k in rk)),
                "duplicates_within_split": int(len(pk) - uniq),
                "in_other_splits_positives": int(sum(k in others for k in pk)),
                "known_positive": int(sum(k in known for k in pk)),
            })
        qk = keys(q, n)
        out["leak"].append({
            "task": task, "split": s, "kind": "negatives", "n": len(qk),
            "in_message_graph_any_rel": int(sum(k in msg_all for k in qk)),
            "in_message_graph_target_pair": int(sum(k in msg_tgt for k in qk)),
            "reversed_in_message_graph_any_rel": int(sum(k in msg_all for k in keys(q.flip(0), n))),
            "reversed_in_target_pair": int(sum(k in msg_rev for k in keys(q.flip(0), n))),
            "duplicates_within_split": int(len(qk) - len(set(qk.tolist()))),
            "in_other_splits_positives": "",
            "known_positive": int(sum(k in known for k in qk)),
        })

        for kind, e in (("pos", p), ("neg", q)):
            if kind == "pos" and s == "test_random":
                continue
            label = s if kind == "neg" else ("test" if s == "test_2hop" else s)
            for end, idx in (("head", e[0]), ("tail", e[1])):
                for dname, dv in (("all", deg_all), ("target_rel", deg_tgt)):
                    v = dv[idx.cpu().numpy()]
                    qs = np.quantile(v, QUANTILES) if v.size else [np.nan] * len(QUANTILES)
                    out["degree"].append({
                        "task": task, "split": label, "kind": kind, "end": end, "degree": dname,
                        "n": int(v.size), "mean": f"{v.mean():.3f}" if v.size else "",
                        **{f"q{int(qq * 100)}": f"{x:g}" for qq, x in zip(QUANTILES, qs)},
                        "share_deg0": f"{(v == 0).mean():.4f}" if v.size else "",
                    })

        for dname, dv in (("all", deg_all), ("target_rel", deg_tgt)):
            ld = torch.from_numpy(np.log1p(dv.astype(np.float64)))
            for term in ("head+tail", "head", "tail"):
                def score(e):
                    h, t = ld[e[0].cpu()], ld[e[1].cpu()]
                    return h + t if term == "head+tail" else (h if term == "head" else t)
                m = compute_link_prediction_metrics(score(p), score(q))
                out["auroc"].append({"task": task, "split": s, "degree": dname, "score": term,
                                     "n_pos": int(p.size(1)), "n_neg": int(q.size(1)),
                                     "auroc": f"{float(m['auroc']):.4f}"})

    from src.models.h2snn_model import build_link_prediction_model

    model = build_link_prediction_model(cfg, data)
    enc = model.node_encoder
    for i, name in enumerate(data.node_type_names):
        width = int(data.feature_dims[name])
        lo, hi = data.node_slices[name]
        emb = enc.identity_embeddings[i] if enc is not None else None
        out["features"].append({
            "task": task, "node_type": name, "nodes": hi - lo, "input_width": width,
            "encoder": "learned per-node embedding" if isinstance(emb, torch.nn.Embedding)
            else (type(emb).__name__ + (" (features)" if width > 0 else "")),
            "deg0_share_all": f"{(deg_all[lo:hi] == 0).mean():.4f}",
            "mean_degree_all": f"{deg_all[lo:hi].mean():.3f}",
        })


def write(path: Path, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8") as h:
        h.write("\t".join(cols) + "\n")
        for r in rows:
            h.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acm", type=Path, required=True)
    ap.add_argument("--dblp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out = {"counts": [], "leak": [], "degree": [], "features": [], "auroc": []}
    for task, path in (("acm_pa", args.acm), ("dblp_ap", args.dblp)):
        run_task(task, path, out)
    for name, rows in (("a1_counts", out["counts"]), ("a1_leakage", out["leak"]),
                       ("a1_degree", out["degree"]), ("a1_features", out["features"]),
                       ("a1_degree_auroc", out["auroc"])):
        write(args.out / f"{name}.tsv", rows)
        print(f"== {name}")
        print((args.out / f"{name}.tsv").read_text())


if __name__ == "__main__":
    main()
