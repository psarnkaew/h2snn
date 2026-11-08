"""G3c: known-answer task for path-specific binding (pre-registered in g3c_preregistration.md).
Run from H2S-NN/:
    python ../docs/D12_evidence/g3c_binding.py --design v1|v2 --check_graph
    python ../docs/D12_evidence/g3c_binding.py --design v1|v2 [--arms ...] [--seeds ...] [--stop_on_leak]
The production pipeline end to end (fact layer, walk sampler, model, trainer); only the dataset
loader is replaced by the synthetic HeteroData of the design (see the pre-registration for both).
Writes docs/D12_evidence/g3c_results.json (one entry per run) and g3c_graph_check_<design>.json."""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import src.data.dataset as dataset_module  # noqa: E402
from src.builders import build_experiment_components  # noqa: E402
from src.configs.config import MetaPathConfig, TrainConfig  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

EVIDENCE = ROOT.parent / "docs" / "D12_evidence"
OUT = EVIDENCE / "g3c_results.json"
NT, KEY_BITS, NOISE, CO_PAPERS = 1000, 6, 8, 16
CHANCE, LEAK_MARGIN = 0.5, 0.1
CONTROLS = ("no_path_cells", "endpoint_edges", "h0_only")
ARMS = {"no_path_cells": {"no_path_cells": True}, "endpoint_edges": {"endpoint_edges": True},
        "h0_only": {"h0_only_readout": True}, "full": {},
        "uniform_path_gate": {"uniform_path_gate": True}}


def _codes() -> torch.Tensor:
    """All 2^KEY_BITS keys as +-1 vectors."""
    c = torch.arange(2 ** KEY_BITS)
    return ((c.unsqueeze(1) >> torch.arange(KEY_BITS)) & 1).float() * 2 - 1


def _balanced_labels(g: torch.Generator) -> torch.Tensor:
    y = torch.cat([torch.zeros(NT // 2), torch.ones(NT - NT // 2)]).long()
    return y[torch.randperm(NT, generator=g)]


def _permutation(g: torch.Generator, ok) -> list[int]:
    """A permutation p of range(NT) with ok(i, p[i]) for all i (random swaps repair violations)."""
    p = torch.randperm(NT, generator=g).tolist()
    for _ in range(1000):
        bad = [i for i in range(NT) if not ok(i, p[i])]
        if not bad:
            return p
        for i in bad:
            j = int(torch.randint(NT, (1,), generator=g))
            if ok(i, p[j]) and ok(j, p[i]):
                p[i], p[j] = p[j], p[i]
    raise RuntimeError("no admissible permutation")


def _hetero(paper_x, author_x, writes, y_targets):
    """HeteroData with targets = papers 0..NT-1, split 50/25/25 (split seed 1)."""
    from torch_geometric.data import HeteroData

    data = HeteroData()
    data["paper"].x = paper_x
    data["author"].x = author_x
    data["author", "writes", "paper"].edge_index = writes
    data["paper", "written_by", "author"].edge_index = writes.flip(0)
    y = torch.full((paper_x.size(0),), -1, dtype=torch.long)
    y[:NT] = y_targets
    order = torch.randperm(NT, generator=torch.Generator().manual_seed(1))
    cuts = {"train": order[: NT // 2], "val": order[NT // 2: 3 * NT // 4], "test": order[3 * NT // 4:]}
    data["paper"].y = y
    for name, index in cuts.items():
        mask = torch.zeros(paper_x.size(0), dtype=torch.bool)
        mask[index] = True
        data["paper"][f"{name}_mask"] = mask
    return data


def graph_v1():
    """Author's design, literal: L_i (lead 1), N_i (lead 0); one co-paper each, complementary classes."""
    g = torch.Generator().manual_seed(0)
    y = _balanced_labels(g)
    papers = 3 * NT  # targets, Q*_i = NT + i, Q'_i = 2NT + i
    paper_x = torch.zeros(papers, 2 + NOISE)
    paper_x[:, 2:] = torch.randn(papers, NOISE, generator=g)
    ar = torch.arange(NT)
    paper_x[NT + ar, y] = 1.0
    paper_x[2 * NT + ar, 1 - y] = 1.0
    author_x = torch.cat([torch.ones(NT, 1), torch.zeros(NT, 1)])  # L_i = i, N_i = NT + i
    writes = torch.stack([torch.cat([ar, NT + ar, ar, NT + ar]),
                          torch.cat([ar, ar, NT + ar, 2 * NT + ar])])
    truth = {"y": y, "L": ar, "N": NT + ar, "qstar": NT + ar, "qprime": 2 * NT + ar}
    return _hetero(paper_x, author_x, writes, y), truth, {"papers": papers, "authors": 2 * NT,
                                                          "facts": int(writes.size(1))}


def graph_v2():
    """Contingent redesign: lead bit x key; every author has 16 co-papers, 8 per class (see prereg)."""
    g = torch.Generator().manual_seed(0)
    y = _balanced_labels(g)
    codes = _codes()
    key = torch.randint(2 ** KEY_BITS, (NT,), generator=g)

    def ok(i, j):
        return j != i and int(key[j]) != int(key[i])

    sigma, tau = _permutation(g, ok), _permutation(g, ok)
    cls, ckey, c_l, c_n = [], [], [], []  # co-papers: class, key, lead author, non-lead author
    for i in range(NT):  # Q*_i
        cls.append(int(y[i])), ckey.append(int(key[i])), c_l.append(i), c_n.append(sigma[i])
    for i in range(NT):  # Q'_i
        cls.append(1 - int(y[i])), ckey.append(int(key[i])), c_l.append(tau[i]), c_n.append(i)
    half = CO_PAPERS // 2
    have = {"L": torch.zeros(NT, 2, dtype=torch.long), "N": torch.zeros(NT, 2, dtype=torch.long)}
    for c, a, b in zip(cls, c_l, c_n):
        have["L"][a, c] += 1
        have["N"][b, c] += 1
    for c in (0, 1):
        l_slots = [a for a in range(NT) for _ in range(half - int(have["L"][a, c]))]
        n_slots = [b for b in range(NT) for _ in range(half - int(have["N"][b, c]))]
        assert len(l_slots) == len(n_slots), (c, len(l_slots), len(n_slots))
        l_slots = [l_slots[k] for k in torch.randperm(len(l_slots), generator=g).tolist()]
        n_slots = [n_slots[k] for k in torch.randperm(len(n_slots), generator=g).tolist()]
        for _ in range(1000):  # never L_i with N_i of the same target
            bad = [k for k in range(len(l_slots)) if l_slots[k] == n_slots[k]]
            if not bad:
                break
            for k in bad:
                t = int(torch.randint(len(n_slots), (1,), generator=g))
                if n_slots[t] != l_slots[k] and n_slots[k] != l_slots[t]:
                    n_slots[k], n_slots[t] = n_slots[t], n_slots[k]
        else:
            raise RuntimeError("filler pairing failed")
        for a, b in zip(l_slots, n_slots):
            while True:
                k = int(torch.randint(2 ** KEY_BITS, (1,), generator=g))
                if k != int(key[a]) and k != int(key[b]):
                    break
            cls.append(c), ckey.append(k), c_l.append(a), c_n.append(b)
    num_co = len(cls)
    papers = NT + num_co
    paper_x = torch.zeros(papers, 2 + KEY_BITS + NOISE)
    paper_x[:NT, 2:2 + KEY_BITS] = codes[key]
    co = NT + torch.arange(num_co)
    cls_t, ckey_t = torch.tensor(cls), torch.tensor(ckey)
    paper_x[co, cls_t] = 1.0
    paper_x[co, 2:2 + KEY_BITS] = codes[ckey_t]
    paper_x[:, 2 + KEY_BITS:] = torch.randn(papers, NOISE, generator=g)
    author_x = torch.cat([torch.ones(NT, 1), torch.zeros(NT, 1)])  # L_a = a, N_b = NT + b
    ar = torch.arange(NT)
    writes = torch.stack([torch.cat([ar, NT + ar, torch.tensor(c_l), NT + torch.tensor(c_n)]),
                          torch.cat([ar, ar, co, co])])
    # ---- invariants (pre-registered)
    for side, authors in (("L", torch.tensor(c_l)), ("N", torch.tensor(c_n))):
        for c in (0, 1):
            counts = torch.bincount(authors[cls_t == c], minlength=NT)
            assert bool((counts == half).all()), f"invariant 1: {side} class {c}"
    l_t, n_t = torch.tensor(c_l), torch.tensor(c_n)
    for i in range(NT):
        on_l = ((l_t == i) & (ckey_t == key[i])).nonzero().flatten().tolist()
        on_n = ((n_t == i) & (ckey_t == key[i])).nonzero().flatten().tolist()
        assert on_l == [i] and on_n == [NT + i], f"invariant 2: target {i}"
    assert bool((l_t != n_t).all()), "invariant 4"  # 3 holds by construction (one L, one N column)
    assert int(y.sum()) == NT // 2, "invariant 5"
    truth = {"y": y, "L": ar, "N": NT + ar, "qstar": NT + ar, "qprime": 2 * NT + ar}
    stats = {"papers": papers, "authors": 2 * NT, "co_papers": num_co, "fillers": num_co - 2 * NT,
             "facts": int(writes.size(1)), "distinct_keys_on_targets": int(key.unique().numel()),
             "invariants": "1-5 asserted"}
    return _hetero(paper_x, author_x, writes, y), truth, stats


GRAPHS = {"v1": graph_v1, "v2": graph_v2}


def config(design: str, arm: str, seed: int) -> TrainConfig:
    raw = yaml.safe_load((ROOT / "configs" / "experiments" / "acm_h2snn.yaml").read_text(encoding="utf-8"))
    raw["data"] = {**raw["data"], "resample_paths": False, "face_sample_k": 400}
    raw.update(seed=seed, epochs=200, device="cuda", ablation=ARMS[arm], checkpoint_metric="acc",
               checkpoint_mode="max", exp_name=f"g3c_{design}_{arm}_s{seed}",
               output_dir="experiments/d12_g3c")
    cfg = TrainConfig.from_dict(raw)
    templates = (MetaPathConfig(name="PAP", node_types=("paper", "author", "paper")),)
    cfg.data = dataclasses.replace(cfg.data, metapath_templates=templates, target_node_type="paper")
    return cfg


def cell_index(data, truth, targets):
    """Cell ids of the relevant (P, L_P, Q*_P) and distractor (P, N_P, Q'_P) cells of `targets`
    (local target ids); -1 where the cell is not in the draw."""
    paper0, author0 = data.node_slices["paper"][0], data.node_slices["author"][0]
    group = next(grp for grp in data.face_groups if grp.template_name == "PAP")
    lookup = {(min(a, c), b, max(a, c)): f
              for f, (a, b, c) in enumerate(group.face_node_index.cpu().tolist())}

    def ids(middle, other):
        out = []
        for t in targets:
            p, q = paper0 + t, paper0 + int(truth[other][t])
            out.append(lookup.get((min(p, q), author0 + int(truth[middle][t]), max(p, q)), -1))
        return torch.tensor(out)

    return group, ids("L", "qstar"), ids("N", "qprime")


@torch.no_grad()
def path_weights(model, data, group):
    """w_f of every cell of `group` (eval mode), or None for arms without a path gate."""
    if model.h0_only or model.path_gate is None or model.path_encoder is None:
        return None
    model.eval()
    h0 = model.encode_nodes(data.x, data.node_type)
    geometry = model.geometry_inputs(h0, data.edge_index, data.edge_type)
    return model._path_weights(model._path_latent(geometry, group), group, geometry).cpu()


def check_graph(design: str) -> None:
    graph, truth, stats = GRAPHS[design]()
    print(design, json.dumps(stats), flush=True)
    (EVIDENCE / f"g3c_graph_check_{design}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", choices=list(GRAPHS), required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--stop_on_leak", action="store_true")
    parser.add_argument("--check_graph", action="store_true")
    args = parser.parse_args()
    if args.check_graph:
        check_graph(args.design)
        return
    graph, truth, stats = GRAPHS[args.design]()
    print(args.design, json.dumps(stats), flush=True)
    dataset_module._load_hetero_data = lambda _config: graph.clone()  # the only replaced piece
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for arm in args.arms:
        for seed in args.seeds:
            cfg = config(args.design, arm, seed)
            if (cfg.exp_dir / "metrics.jsonl").exists():
                raise SystemExit(f"{cfg.exp_dir} already has metrics.jsonl (the logger appends).")
            cfg.exp_dir.mkdir(parents=True, exist_ok=True)
            seed_everything(seed, cfg.device)
            start = time.time()
            data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
            try:
                trainer.fit(data)
            finally:
                logger.close()
            seconds = time.time() - start
            best = json.loads((cfg.exp_dir / "best_test_metrics.json").read_text(encoding="utf-8"))
            rows = [json.loads(l) for l in (cfg.exp_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            test_targets = data.target_test_mask.nonzero().flatten().tolist()  # local target ids
            group, relevant, distractor = cell_index(data, truth, test_targets)
            entry = {
                "test_acc": best["test_acc"], "val_acc": best["val_acc"], "best_epoch": best["best_epoch"],
                "epochs_run": len(rows), "train_acc_at_best_trainmode": rows[best["best_epoch"]]["train"]["acc"],
                "cells_PAP": group.num_faces, "seconds": round(seconds, 1),
                "coverage_relevant_and_distractor": float(((relevant >= 0) & (distractor >= 0)).float().mean()),
            }
            device = next(model.parameters()).device
            payload = torch.load(cfg.exp_dir / "checkpoints" / "best.pt", map_location=device)
            model.load_state_dict(payload["state_dict"])
            weights = path_weights(model, data.to(device), group)
            if weights is not None:
                entry["w_f_mean"] = {"relevant": float(weights[relevant[relevant >= 0]].mean()),
                                     "distractor": float(weights[distractor[distractor >= 0]].mean()),
                                     "all_cells": float(weights.mean())}
            entry["leak"] = arm in CONTROLS and entry["test_acc"] > CHANCE + LEAK_MARGIN
            results[f"{args.design}/{arm}/s{seed}"] = entry
            print(args.design, arm, seed, json.dumps(entry), flush=True)
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
            if args.stop_on_leak and entry["leak"]:
                print(f"LEAK: {args.design} {arm} seed {seed} test_acc {entry['test_acc']:.4f} > "
                      f"{CHANCE + LEAK_MARGIN}; design stopped (pre-registered leak rule).", flush=True)
                return


if __name__ == "__main__":
    main()
