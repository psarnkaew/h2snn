"""Card 6 H2c-1 (diagnostic): ieee-cis-g MDE closed form vs the cancellation-free form Σ‖x−μ‖²/2 per clique
(d = n), plus raw feature stats; job 17769. Run from the worktree's H2S-NN/."""
import importlib.util, json, sys
from pathlib import Path
import torch
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("stats", ROOT / "scripts" / "heterophily_stats.py")
stats = importlib.util.module_from_spec(spec); spec.loader.exec_module(stats); het = stats.het
raw, target, _ = stats.load_nc("ieee-cis-g", Path(stats.REPO_DATA_ROOT))
sym, _ = het.symmetrise_facts(raw)
paths, _ = het.hetero2net_m2("ieee-cis-g", sym.edge_types, target)
x = sym[target].x
xd = x.to(torch.float64)
print(json.dumps({"x_shape": list(x.shape), "dtype": str(x.dtype), "abs_max": float(xd.abs().max()),
      "abs_mean": float(xd.abs().mean()), "nonzero_frac": float((xd != 0).double().mean()),
      "row_norm_mean": float(xd.norm(dim=1).mean()), "row_norm_max": float(xd.norm(dim=1).max()),
      "distinct_rows_in_first_20000": int(torch.unique(xd[:20000], dim=0).size(0))}), flush=True)
for path in paths:
    mid = het.clique_gp_mid(sym, path)
    cf, edges = het.clique_mde(mid, x)
    keep = mid >= 0; m = mid[keep]; xk = xd[keep]
    n_mid = int(m.max()) + 1
    size = torch.bincount(m, minlength=n_mid).to(torch.float64)
    s1 = torch.zeros(n_mid, xk.size(1), dtype=torch.float64).index_add_(0, m, xk)
    mu = s1 / size.clamp(min=1).unsqueeze(1)
    dev = (xk - mu[m]).square().sum(dim=1)
    ss = torch.zeros(n_mid, dtype=torch.float64).index_add_(0, m, dev)
    stable = float((ss / 2).sum() / (size * size).sum())
    print(json.dumps({"metapath": het.metapath_name(path), "closed_form": cf, "stable": stable, "edges": edges,
          "max_clique": int(size.max()), "sum_sq_dev": float(ss.sum())}), flush=True)
