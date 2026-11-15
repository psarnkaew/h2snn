"""Card baselines-h2gb C2 (slim mode): the PDNS loader equals H2GB's own code on a fixture with H2GB's file layout.

The fixture writes every raw file H2GB's `PDNSDataset` lists (`pdns_dataset.py:64-88`; 61 edge windows, features,
`domains2.csv`). Our loader (`load_h2gb_dataset("pdns", root, seed)`) must return exactly what H2GB's code returns
after `seed_everything(seed)` (`H2GB/main.py:87-101`): graph, labels and the random 70/20/10 split; the split is a
function of the seed; nothing is written or downloaded. The real data root is unreachable.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def write_pdns_fixture(root: Path, num_domains: int = 60, num_ips: int = 20, seed: int = 0) -> Path:
    """`<root>/NC/PDNS/raw/` with H2GB's file names and columns."""
    rng = np.random.default_rng(seed)
    raw = root / "NC" / "PDNS" / "raw"
    raw.mkdir(parents=True)
    domain_cols = {f"f{i}": rng.random(num_domains) for i in range(10)}
    _csv(raw / "domain_features.csv", {"domain_node": np.arange(num_domains), **domain_cols})
    _csv(raw / "ip_features.csv", {"ip_node": np.arange(num_ips), "g0": rng.random(num_ips),
                                   "g1": rng.random(num_ips)})
    types = rng.choice(["malicious", "benign", "unknown"], size=num_domains, p=[0.5, 0.3, 0.2])
    _csv(raw / "domains2.csv", {"domain_node": np.arange(num_domains), "type": types})
    for name in ("domains.csv", "domain_features_map.txt", "ip_features_map.txt", "ips.csv"):
        (raw / name).write_text("unused by the loader\n")
    for window in range(61):
        rows = []
        for kind, count in (("resolves", 12), ("similar", 4), ("apex", 3)):
            src = rng.integers(0, num_domains, count)
            dst = rng.integers(0, num_ips if kind == "resolves" else num_domains, count)
            rows += [(int(s), int(d), kind) for s, d in zip(src, dst)]
        _csv(raw / f"timestamp_{window}_edges.csv",
             {"source": [r[0] for r in rows], "target": [r[1] for r in rows], "type": [r[2] for r in rows]})
    return raw.parent


def _csv(path: Path, columns: dict) -> None:
    import pandas as pd

    pd.DataFrame(columns).to_csv(path, index=False)


def _h2gb_own(pdns_dir: Path, seed: int):
    """H2GB's code as `H2GB/main.py:87-101` runs it: seed_everything, then preformat_PDNS (its own class)."""
    from torch_geometric import seed_everything

    from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
    from src.baselines.runtime.vendor_paths import pushd_sys_path

    ensure_pyg_submodules()
    with pushd_sys_path("h2gb"):
        import H2GB  # noqa: F401
        import H2GB.loader.master_loader as master_loader  # type: ignore[import-not-found]

        seed_everything(seed)
        return master_loader.preformat_PDNS(str(pdns_dir))[0]


class PDNSLoaderTests(unittest.TestCase):
    def test_split_equals_h2gb_code_and_is_a_function_of_the_seed(self) -> None:
        from src.data.h2gb_nc import load_h2gb_dataset, split_record

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdns_dir = write_pdns_fixture(root)
            before = sorted(p.relative_to(root) for p in root.rglob("*"))
            ours = {seed: load_h2gb_dataset("pdns", root, seed=seed)[0] for seed in (0, 1)}
            theirs = _h2gb_own(pdns_dir, 0)
            self.assertEqual(sorted(p.relative_to(root) for p in root.rglob("*")), before, "wrote under the root")
        ours0 = ours[0]
        self.assertEqual(set(ours0.edge_types), set(theirs.edge_types))
        for edge_type in theirs.edge_types:
            self.assertTrue(torch.equal(ours0[edge_type].edge_index, theirs[edge_type].edge_index))
        for node_type in theirs.node_types:
            self.assertTrue(torch.equal(ours0[node_type].x, theirs[node_type].x))
        target = "domain_node"
        self.assertTrue(torch.equal(ours0[target].y, theirs[target].y))
        for split in ("train", "val", "test"):
            self.assertTrue(torch.equal(ours0[target][f"{split}_mask"], theirs[target][f"{split}_mask"]), split)
        # 70/20/10 as H2GB's int() arithmetic (pdns_dataset.py:308-310); the split depends on the seed only
        labelled = int((theirs[target].y != -1).sum())
        sizes = {s: v["size"] for s, v in split_record(ours0, target).items()}
        self.assertEqual(sizes["test"], int(labelled * 0.1))
        self.assertEqual(sizes["val"], int(labelled * 0.3) - int(labelled * 0.1))
        self.assertEqual(sum(sizes.values()), labelled)
        self.assertNotEqual(split_record(ours[0], target), split_record(ours[1], target))


if __name__ == "__main__":
    unittest.main()
