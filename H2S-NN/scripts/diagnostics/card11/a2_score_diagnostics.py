"""Card 11 A2: score diagnostics on the A0 models (and A3 models) at given saved states.

For one trained run directory (`run_from_config.py ... --save_epochs init,0,39`): rebuilds the data
and model from the run's own `config.yaml` (`build_experiment_components`, seed `cfg.seed`), loads
each `card11_state_<tag>.pt`, and in evaluation mode (one `encode(..., epoch=-1)` on the evaluation
draw, the trainer's `_eval_epoch` path; dropout off) computes, with the model's own `encode` and
`link_readout`:

- `a2_auroc.tsv`: AUROC per split: train (training positives vs the training negatives of that
  epoch, `hgb_epoch_train_negatives(data, e)`; `init` uses epoch 0's), validation, test 2-hop,
  test random; also the mean score of positives and negatives;
- `a2_score_degree.tsv`: Spearman of the score with the head degree and with the tail degree
  (message graph, all relations; also target relation only), separately for positives and
  negatives, per split;
- `a2_norm_degree.tsv`: per node type of the target relation, Spearman of `||x*||`,
  `||endpoint_projection(x*)||` and `||h0||` with degree, and their means per degree bin.

No `src` change; the trainer's training-mode train AUROC (dropout, the epoch's training draw) is
in the run's metrics.jsonl and is not recomputed here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import TrainConfig  # noqa: E402
from src.data.hgb_lp_negatives import hgb_epoch_train_negatives  # noqa: E402
from src.utils.misc import compute_link_prediction_metrics, lp_test_sets, seed_everything  # noqa: E402

BINS = ((0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 10**9))


def rho(a, b) -> str:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return ""
    return f"{spearmanr(a, b).statistic:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, action="append", required=True)
    ap.add_argument("--tags", default="init,epoch0,epoch39")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None, help="override cfg.device (evaluation only)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows_auc, rows_sd, rows_nd = [], [], []
    from src.builders import build_experiment_components

    for run_dir in args.run_dir:
        raw = yaml.safe_load((run_dir / "config.yaml").read_text())
        raw["use_wandb"] = False
        raw["output_dir"] = str(args.out / "_scratch")
        raw["exp_name"] = run_dir.name
        if args.device:
            raw["device"] = args.device
        cfg = TrainConfig.from_dict(raw)
        cfg.exp_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(cfg.seed, cfg.device)
        data, model, _, _, logger, _, _ = build_experiment_components(cfg)
        logger.close()
        dev = torch.device(cfg.device)
        model.to(dev)
        d = data.to(dev)
        n = data.num_nodes
        ht, tt = data.target_relation[0], data.target_relation[2]
        pair_ids = [i for i, t in enumerate(data.relation_tuples) if (t[0], t[2]) in ((ht, tt), (tt, ht))]
        ei, et = data.edge_index, data.edge_type
        deg_all = (torch.bincount(ei[0], minlength=n) + torch.bincount(ei[1], minlength=n)).numpy()
        tm = torch.isin(et, torch.tensor(pair_ids, dtype=et.dtype))
        deg_tgt = (torch.bincount(ei[0][tm], minlength=n) + torch.bincount(ei[1][tm], minlength=n)).numpy()
        sets = lp_test_sets(data)
        for tag in args.tags.split(","):
            path = run_dir / f"card11_state_{tag}.pt"
            if not path.exists():
                print(f"missing {path}", flush=True)
                continue
            model.load_state_dict(torch.load(path, map_location=dev))
            model.eval()
            epoch = 0 if tag == "init" else int(tag.removeprefix("epoch"))
            with torch.no_grad():
                enc = model.encode(x=d.x, node_type=d.node_type, edge_index=d.edge_index,
                                   edge_type=d.edge_type, face_groups=d.face_groups,
                                   num_nodes=d.num_nodes, epoch=-1)
                proj = model.link_readout.endpoint_projection(enc.representation)

                def score(e, r=None):
                    e = e.to(dev)
                    r = torch.zeros(e.size(1), dtype=torch.long, device=dev) if r is None else r.to(dev)
                    return model.link_readout(node_representation=enc.representation,
                                              edge_label_index=e, edge_relation=r).float().cpu()

                tr_neg, tr_neg_rel = hgb_epoch_train_negatives(data, epoch)
                splits = {
                    "train": (data.train_pos_edge_index, tr_neg),
                    "val": (data.val_pos_edge_index, data.val_neg_edge_index),
                    "test_2hop": (data.test_pos_edge_index, sets["2hop"][0]),
                    "test_random": (data.test_pos_edge_index, sets["random"][0]),
                }
                for split, (p, q) in splits.items():
                    sp, sq = score(p), score(q)
                    m = compute_link_prediction_metrics(sp, sq)
                    rows_auc.append({"run": run_dir.name, "state": tag, "split": split,
                                     "auroc": f"{float(m['auroc']):.4f}",
                                     "pos_score_mean": f"{float(sp.mean()):.4f}",
                                     "neg_score_mean": f"{float(sq.mean()):.4f}",
                                     "pos_score_std": f"{float(sp.std()):.4f}",
                                     "neg_score_std": f"{float(sq.std()):.4f}"})
                    for kind, e, s in (("pos", p, sp), ("neg", q, sq)):
                        h, t = e[0].cpu().numpy(), e[1].cpu().numpy()
                        rows_sd.append({"run": run_dir.name, "state": tag, "split": split, "kind": kind,
                                        "n": int(s.numel()),
                                        "rho_head_deg": rho(s, deg_all[h]),
                                        "rho_tail_deg": rho(s, deg_all[t]),
                                        "rho_head_deg_target_rel": rho(s, deg_tgt[h]),
                                        "rho_tail_deg_target_rel": rho(s, deg_tgt[t]),
                                        "rho_sum_log_deg": rho(s, np.log1p(deg_all[h]) + np.log1p(deg_all[t]))})
                norms = {"x_star": enc.x_star.norm(dim=-1).float().cpu().numpy(),
                         "endpoint_proj": proj.norm(dim=-1).float().cpu().numpy(),
                         "h0": enc.h0.norm(dim=-1).float().cpu().numpy()}
            for end, ntype in (("head", data.target_relation[0]), ("tail", data.target_relation[2])):
                lo, hi = data.node_slices[ntype]
                dg = deg_all[lo:hi]
                for name, v in norms.items():
                    vv = v[lo:hi]
                    row = {"run": run_dir.name, "state": tag, "end": end, "node_type": ntype,
                           "quantity": name, "rho_deg": rho(vv, dg), "mean": f"{vv.mean():.4f}"}
                    for a, b in BINS:
                        sel = (dg >= a) & (dg <= b)
                        label = f"deg{a}" if a == b else (f"deg{a}-{b}" if b < 10**9 else f"deg{a}+")
                        row[label] = f"{vv[sel].mean():.4f} (n={int(sel.sum())})" if sel.any() else ""
                    rows_nd.append(row)
            print(f"done {run_dir.name} {tag}", flush=True)

    for name, rows in (("a2_auroc", rows_auc), ("a2_score_degree", rows_sd), ("a2_norm_degree", rows_nd)):
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with (args.out / f"{name}.tsv").open("w", encoding="utf-8") as h:
            h.write("\t".join(cols) + "\n")
            for r in rows:
                h.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
        print(f"== {name}\n" + (args.out / f"{name}.tsv").read_text(), flush=True)


if __name__ == "__main__":
    main()
