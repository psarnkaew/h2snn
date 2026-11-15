"""Part M3b B5 data check: PDNS as H2GB's own code loads it, and its time windows.

    python scripts/data/h2gb_pdns_check.py --root data --out <file.json> [--seed 0]

1. H2GB's `preformat_PDNS` (`H2GB/loader/master_loader.py:604-617`: `PDNSDataset(start=0, end=60,
   domain_file='domains2.csv')`, `ToUndirected`) after `seed_everything(seed)`, as `H2GB/main.py:87-100` does
   before `create_loader`: the graph H2GB trains and evaluates on (`dataset[0]`), its stored form, labels and
   the random 70/20/10 split of `_load_data` (`pdns_dataset.py:272-310`, unseeded `torch.randperm`, so a
   function of the run seed); and the test graphs `dataset.test_data` that H2GB builds (`:175-187`) but
   never reads.
2. Per time window (`raw/timestamp_<t>_edges.csv`, t = 0..60): edges per type, domains and IPs seen, and
   the labels (`domains2.csv` `type`: malicious = 1, benign = 0, other = -1) of the domains seen.
Reads `raw/` only (all files present: no download path is reached); writes only `--out`. No processed file
exists or is written (H2GB's class has no `process`). Run in a SLURM job.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.data.data_layout import nc_root  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "data"))
from h2gb_nc_counts import _graph_counts, _label_counts  # noqa: E402


def _windows(raw: Path) -> list[dict]:
    import pandas as pd

    domains = pd.read_csv(raw / "domains2.csv").sort_values("domain_node")
    label = domains["type"].map(lambda t: 1 if t == "malicious" else (0 if t == "benign" else -1)).to_numpy()
    rows = []
    files = sorted((p for p in raw.iterdir() if p.name.startswith("timestamp_")),
                   key=lambda p: int(p.name.split("_")[1]))
    seen_before: set[int] = set()
    for path in files:
        edges = pd.read_csv(path)
        doms: set[int] = set()
        ips: set[int] = set()
        per_type = {}
        for kind, part in edges.groupby("type"):
            per_type[str(kind)] = int(len(part))
            doms.update(part["source"].astype(int).tolist())
            if kind == "resolves":
                ips.update(part["target"].astype(int).tolist())
            else:
                doms.update(part["target"].astype(int).tolist())
        labels = [int(label[d]) for d in doms]
        new = doms - seen_before
        rows.append({"window": int(path.name.split("_")[1]), "edges": per_type, "domains": len(doms),
                     "ips": len(ips), "malicious": labels.count(1), "benign": labels.count(0),
                     "unlabelled": labels.count(-1), "new_domains": len(new),
                     "new_labelled": sum(1 for d in new if label[d] != -1)})
        seen_before |= doms
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO / "data"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pdns_dir = nc_root(Path(args.root)) / "PDNS"
    raw = pdns_dir / "raw"
    out: dict = {"pdns_dir": str(pdns_dir), "seed": args.seed}

    from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
    from src.baselines.runtime.vendor_paths import pushd_sys_path

    ensure_pyg_submodules()
    start = time.perf_counter()
    with pushd_sys_path("h2gb"):
        import H2GB  # noqa: F401
        import H2GB.loader.master_loader as master_loader  # type: ignore[import-not-found]
        from H2GB.graphgym.config import cfg, set_cfg  # type: ignore[import-not-found]
        from torch_geometric import seed_everything

        set_cfg(cfg)
        # every raw file H2GB lists must exist, so PyG's `_download` returns without calling `download`
        expected = master_loader.PDNSDataset.raw_file_names.fget(object())
        missing = [name for name in expected if not (raw / name).is_file()]
        if missing:
            raise FileNotFoundError(f"PDNS raw files missing (no download in this project): {missing}")
        out["raw_files_expected_by_h2gb"] = len(expected)
        seed_everything(args.seed)
        dataset = master_loader.preformat_PDNS(str(pdns_dir))
    out["load_seconds"] = round(time.perf_counter() - start, 1)
    out["transform"] = repr(dataset.transform)
    stored, loaded = dataset.get(0), dataset[0]
    out["stored"] = _graph_counts(stored)
    out["loaded"] = _graph_counts(loaded)
    out["labels"] = _label_counts(loaded["domain_node"])
    out["test_graphs"] = [{"stored": _graph_counts(g), "labels": _label_counts(g["domain_node"])}
                          for g in dataset.test_data]
    out["windows"] = _windows(raw)
    out["raw_files"] = {p.name: p.stat().st_size for p in sorted(raw.iterdir()) if p.is_file()}
    out["pdns_dir_after"] = sorted(p.name for p in pdns_dir.iterdir())
    out["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("load_seconds", "peak_rss_gib", "pdns_dir_after")}))


if __name__ == "__main__":
    main()
