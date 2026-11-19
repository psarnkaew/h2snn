"""Diagnostic (not repository code): eta_v on the real ACM/DBLP test targets under two readings.
union: distinct labelled neighbours over the union of the M_2 meta-path graphs (as implemented);
per-path mean: per meta-path G_P the share of different-label neighbours, then the mean over the paths where v has one."""
import sys
from pathlib import Path
import numpy as np, torch
root = Path("/mnt/data/peeradon/h2snn-model/H2S-NN")
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "scripts/analysis"))
import mechanism_common as mc
from src.configs.config import TrainConfig
from src.data.dataset import build_dataset
for ds in ("acm", "dblp"):
    data, target, ref = mc.reference_graph(root / "data", ds)
    y = data[target].y; n = int(data[target].num_nodes)
    cfg = TrainConfig.from_dict({"device": "cpu", "data": {"dataset_name": ds, "root": str(root / "data")}})
    prep = build_dataset(cfg)
    start, _ = prep.node_slices[prep.target_node_type]
    test = (prep.target_indices[prep.target_test_mask] - start).numpy()
    union = mc.eta(root / "data", ds)["eta"][test]
    per = []
    print(f"== {ds}: M_2 = {[ref.metapath_name(p) for p in ref.h2gb_metapaths(data, target)]}")
    for p in ref.h2gb_metapaths(data, target):
        dis = np.zeros(n); cnt = np.zeros(n)
        for u, v in (c for c in ref.iter_metapath_chunks(data, p, 200)):
            k = u != v; u, v = u[k], v[k]
            np.add.at(dis, v.numpy(), (y[u] != y[v]).double().numpy()); np.add.at(cnt, v.numpy(), 1.0)
        e = np.where(cnt > 0, dis / np.maximum(cnt, 1), np.nan)[test]
        per.append(e)
        print(f"  {ref.metapath_name(p)}: median neighbours {np.median(cnt[test]):.0f}, eta mean {np.nanmean(e):.4f}, "
              f"quartiles {np.round(np.nanquantile(e, [.25, .5, .75]), 4).tolist()}")
    per = np.nanmean(np.stack(per), axis=0)
    for name, e in (("union", union), ("per-path mean", per)):
        edges = mc.quartile_edges(e); q = mc.quartile_of(e, edges)
        vals, counts = np.unique(np.round(e[~np.isnan(e)], 12), return_counts=True)
        print(f"  {name}: edges {np.round(edges, 4).tolist()}, bins {[int((q == i).sum()) for i in range(4)]}, "
              f"none {int((q == -1).sum())}, distinct values {len(vals)}, largest tie {counts.max()}")
