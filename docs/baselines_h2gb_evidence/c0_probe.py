"""C0 probe (card baselines-h2gb, C0): facts about H2GB's `hgt` sampler, read through H2GB's own code.

No real data is read; only H2GB's YAML files and a synthetic graph.
1. For every (class-A / class-C reference method, dataset) H2GB config: merge the YAML into H2GB's cfg exactly as
   H2GB's main does (`set_cfg`, `merge_from_file`) and compute `sample_sizes` exactly as
   `H2GB/sampler/custom_sampler.py:323-326` does; print the per-type list, its length (= HGTLoader iterations,
   PyG `hgt_sampler.py:48` num_hops = max list length) and the model depth key of the same YAML.
2. HGTLoader on a synthetic heterogeneous graph with a per-type dict of unequal list lengths (Pokec's shape).
3. HGTLoader with fan-out >= every node: is the sampled node set the d-hop in-neighbourhood ball and are the
   edges the induced subgraph?
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2] / "H2S-NN"
sys.path.insert(0, str(REPO))

from src.baselines.adapters._pyg_compat import ensure_pyg_submodules  # noqa: E402
from src.baselines.runtime.vendor_paths import pushd_sys_path  # noqa: E402

CONFIG_DIRS = {"ogbn-mag": "ogbn-mag", "mag-year": "mag-year", "oag-cs": "oag-cs", "oag-eng": "oag-eng",
               "oag-chem": "oag-chem", "ieee-cis-g": "IEEE-CIS", "h-pokec": "Pokec", "pdns": "PDNS"}
METHODS = ("GCN", "GAT", "HAN", "H2Gformer", "HINormer")


def part1() -> None:
    ensure_pyg_submodules()
    with pushd_sys_path("h2gb"):
        import H2GB  # noqa: F401
        from H2GB.graphgym.config import cfg, set_cfg
        for key, folder in CONFIG_DIRS.items():
            for method in METHODS:
                path = REPO / "vendor" / "H2GB" / "configs" / folder / f"{folder}-{method}.yaml"
                if not path.is_file():
                    print(f"P1 {key:11s} {method:9s} no config")
                    continue
                set_cfg(cfg)
                try:
                    cfg.merge_from_file(str(path))
                except Exception as exc:  # report, do not hide
                    print(f"P1 {key:11s} {method:9s} merge error {type(exc).__name__}: {exc}")
                    continue
                try:
                    if len(cfg.train.neighbor_sizes_dict) > 0:
                        sizes = eval(cfg.train.neighbor_sizes_dict)  # noqa: S307 - H2GB's own line
                        src = "neighbor_sizes_dict"
                    else:
                        sizes = {"<every type>": list(cfg.train.neighbor_sizes)}
                        src = "neighbor_sizes"
                except Exception as exc:
                    print(f"P1 {key:11s} {method:9s} eval error {type(exc).__name__}: {exc!r} "
                          f"raw={cfg.train.neighbor_sizes_dict!r}")
                    continue
                lens = sorted({len(v) for v in sizes.values()})
                depth_keys = {"gnn.layers_mp": cfg.gnn.layers_mp,
                              "gt.layers": getattr(getattr(cfg, "gt", None), "layers", None)}
                compact = {k: v for k, v in list(sizes.items())[:3]}
                print(f"P1 {key:11s} {method:9s} src={src} sampler={cfg.train.sampler} lens={lens} "
                      f"bs={cfg.train.batch_size} eval_period={cfg.train.eval_period} "
                      f"iter_train={cfg.train.iter_per_epoch} iter_val={cfg.val.iter_per_epoch} "
                      f"model={cfg.model.type} {depth_keys} sizes(first 3)={compact}")


def toy_graph(seed: int = 0):
    from torch_geometric.data import HeteroData
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["a"].x = torch.randn(30, 4, generator=g)
    data["b"].x = torch.randn(20, 3, generator=g)
    data["a"].train_mask = torch.zeros(30, dtype=torch.bool)
    data["a"].train_mask[:10] = True
    ab = torch.stack([torch.randint(0, 30, (60,), generator=g), torch.randint(0, 20, (60,), generator=g)])
    aa = torch.stack([torch.randint(0, 30, (40,), generator=g), torch.randint(0, 30, (40,), generator=g)])
    aa = aa[:, aa[0] != aa[1]]
    data["a", "to", "b"].edge_index = torch.unique(ab, dim=1)
    data["b", "rev_to", "a"].edge_index = torch.unique(ab, dim=1).flip(0)
    data["a", "link", "a"].edge_index = torch.unique(torch.cat([aa, aa.flip(0)], 1), dim=1)
    return data


def part2() -> None:
    from torch_geometric.loader import HGTLoader
    data = toy_graph()
    for sizes in ({"a": [4, 4, 2, 2], "b": [3, 3, 3]}, {"a": [4, 4, 2], "b": [3, 3, 3, 3]}):
        try:
            loader = HGTLoader(data, num_samples=sizes, input_nodes=("a", data["a"].train_mask), batch_size=2,
                               shuffle=False)
            batch = next(iter(loader))
            print(f"P2 sizes={sizes} ok: nodes a={batch['a'].num_nodes} b={batch['b'].num_nodes}")
        except Exception as exc:
            print(f"P2 sizes={sizes} error {type(exc).__name__}: {exc}")


def part3() -> None:
    from torch_geometric.loader import HGTLoader
    data = toy_graph()
    homo = data.to_homogeneous()
    n_a = data["a"].num_nodes
    for depth in (1, 2, 3):
        loader = HGTLoader(data, num_samples={"a": [10_000] * depth, "b": [10_000] * depth},
                           input_nodes=("a", torch.tensor([0, 1])), batch_size=2, shuffle=False)
        batch = next(iter(loader))
        sampled = set(batch["a"].n_id.tolist()) | {n_a + i for i in batch["b"].n_id.tolist()}
        ball = {0, 1}
        frontier = set(ball)
        src, dst = homo.edge_index
        for _ in range(depth):
            new = set(src[torch.isin(dst, torch.tensor(sorted(frontier)))].tolist()) - ball
            ball |= new
            frontier = new
        induced = 0
        for et in data.edge_types:
            s, _, d = et
            ids_s = batch[s].n_id
            ids_d = batch[d].n_id
            ei = data[et].edge_index
            mask = torch.isin(ei[0], ids_s) & torch.isin(ei[1], ids_d)
            induced += int(mask.sum())
        got = sum(int(batch[et].edge_index.size(1)) for et in batch.edge_types)
        print(f"P3 depth={depth} sampled==ball: {sampled == ball} (|sampled|={len(sampled)}, |ball|={len(ball)}); "
              f"edges in batch={got} induced edges={induced}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
