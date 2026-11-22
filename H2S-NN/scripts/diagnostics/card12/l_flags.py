"""Card 12 unit L: per-task summary and the card's flag rule, from the A1 TSVs (`a1_data_checks.py`).

Usage: `l_flags.py --in DIR [DIR ...] --out DIR` (each input DIR holds a1_counts / a1_degree / a1_features /
a1_degree_auroc / a1_leakage TSVs). Writes `l_summary.tsv` and `l_summary.md` and prints the markdown.

Flag rule (card 12, fixed before the data): a task is flagged when both hold:
 (i)  for the tail or the head: share of degree-0 nodes (message graph, all relations) among validation
      positives >= 2 x that among validation negatives, and >= 0.2;
 (ii) the degree-only AUROC (all relations, score log(1+deg_u)+log(1+deg_v)) is > 0.5 on train and < 0.5 on
      validation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(dirs: list[Path], name: str) -> list[dict]:
    rows: list[dict] = []
    for d in dirs:
        path = d / f"{name}.tsv"
        if path.exists():
            with path.open(encoding="utf-8") as h:
                rows += list(csv.DictReader(h, delimiter="\t"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    counts = {r["task"]: r for r in read(args.inputs, "a1_counts")}
    degree = read(args.inputs, "a1_degree")
    features = read(args.inputs, "a1_features")
    auroc = read(args.inputs, "a1_degree_auroc")
    leak = read(args.inputs, "a1_leakage")

    def deg(task, split, kind, end, d="all"):
        for r in degree:
            if (r["task"], r["split"], r["kind"], r["end"], r["degree"]) == (task, split, kind, end, d):
                return r
        return None

    def au(task, split, d="all", score="head+tail"):
        for r in auroc:
            if (r["task"], r["split"], r["degree"], r["score"]) == (task, split, d, score):
                return float(r["auroc"])
        return float("nan")

    def cell(r):
        if r is None or not r["share_deg0"]:
            return "- (empty)"
        return f"{float(r['share_deg0']):.3f} / {r['q50']} / {float(r['mean']):.2f}"

    rows = []
    for task, c in counts.items():
        head, tail = c["head_type"], c["tail_type"]
        rels = [x.split(":", 1)[1] for x in c["relations"].split()]
        target = set(c["target_pair_kept_relations"].split())
        feat = {r["node_type"]: r for r in features if r["task"] == task}

        def others(t):
            return sum(1 for r in rels if r not in target and t in (r.split("/")[0], r.split("/")[-1]))

        def has_feat(t):
            r = feat.get(t)
            if r is None:
                return "?"
            return "yes" if int(r["input_width"]) > 0 else f"no ({r['encoder']})"

        row = {"task": task, "head_type": head, "tail_type": tail,
               "head_features": has_feat(head), "tail_features": has_feat(tail),
               "head_other_relations": others(head), "tail_other_relations": others(tail)}
        cond_i = []
        for end in ("tail", "head"):
            for split in ("train", "val", "test"):
                p = deg(task, split, "pos", end)
                row[f"{end}_{split}_pos_deg0/median/mean"] = (
                    cell(p))
            for split in ("train", "val", "test_2hop", "test_random"):
                q = deg(task, split, "neg", end)
                row[f"{end}_{split}_neg_deg0/median/mean"] = (
                    cell(q))
            vp, vn = deg(task, "val", "pos", end), deg(task, "val", "neg", end)
            sp, sn = float(vp["share_deg0"]), float(vn["share_deg0"])
            cond_i.append((end, sp >= 2 * sn and sp >= 0.2, sp, sn))
        for d in ("all", "target_rel"):
            for split in ("train", "val", "test_2hop", "test_random"):
                row[f"auroc_{d}_{split}"] = f"{au(task, split, d):.4f}"
        leaks = [r for r in leak if r["task"] == task and r["split"] in ("val", "test_2hop")
                 and r["kind"] == "positives"]
        row["heldout_pos_in_message_graph(any/reversed)"] = " ".join(
            f"{r['split']}:{r['in_message_graph_any_rel']}/{r['reversed_in_message_graph_any_rel']}" for r in leaks)
        row["heldout_pos_in_other_splits"] = " ".join(f"{r['split']}:{r['in_other_splits_positives']}" for r in leaks)
        i_holds = any(h for _, h, _, _ in cond_i)
        ii_holds = au(task, "train") > 0.5 and au(task, "val") < 0.5
        row["flag_i"] = "; ".join(f"{e} {sp:.3f} vs {sn:.3f} {'yes' if h else 'no'}" for e, h, sp, sn in cond_i)
        row["flag_ii"] = f"train {au(task, 'train'):.4f} val {au(task, 'val'):.4f} {'yes' if ii_holds else 'no'}"
        row["flagged"] = "FLAGGED" if (i_holds and ii_holds) else "no"
        rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with (args.out / "l_summary.tsv").open("w", encoding="utf-8") as h:
        h.write("\t".join(cols) + "\n")
        for r in rows:
            h.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    md = ["| task | head / tail | features head / tail | other rel. head / tail | val pos deg0 tail / head | "
          "val neg deg0 tail / head | AUROC all: train / val / test-2hop / test-rand | AUROC target pair: train / val | "
          "(i) | (ii) | flag |", "|" + "---|" * 11]
    for r in rows:
        md.append(
            f"| {r['task']} | {r['head_type']} / {r['tail_type']} | {r['head_features']} / {r['tail_features']} | "
            f"{r['head_other_relations']} / {r['tail_other_relations']} | "
            f"{r['tail_val_pos_deg0/median/mean'].split(' / ')[0]} / {r['head_val_pos_deg0/median/mean'].split(' / ')[0]} | "
            f"{r['tail_val_neg_deg0/median/mean'].split(' / ')[0]} / {r['head_val_neg_deg0/median/mean'].split(' / ')[0]} | "
            f"{r['auroc_all_train']} / {r['auroc_all_val']} / {r['auroc_all_test_2hop']} / {r['auroc_all_test_random']} | "
            f"{r['auroc_target_rel_train']} / {r['auroc_target_rel_val']} | "
            f"{'yes' if 'yes' in r['flag_i'] else 'no'} | {r['flag_ii'].split()[-1]} | **{r['flagged']}** |")
    (args.out / "l_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
