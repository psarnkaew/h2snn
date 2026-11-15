"""Dataset classes that never download (card baselines-h2gb C7 item 2; Part M3b out-of-scope finding).

PyG's `Dataset._download` calls the class's `download()` whenever one of its raw files is missing; for `HGBDataset`
that fetches the HGB release from the network into the data root. This project never downloads data: the releases
are placed on the cluster by the author. `offline_hgb_dataset` is PyG's `HGBDataset` whose `download()` raises
`DataNotOnDiskError` naming the missing files instead, so a missing release fails closed; `offline_tu_dataset` does the
same for the TU graph-classification sets (`TUDataset` fetches them from chrsmrrs.com). When every raw file is
present nothing changes (PyG returns before calling `download()`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DataNotOnDiskError(FileNotFoundError):
    """A dataset's files are not under the data root, and this project does not download them."""


def offline_hgb_dataset(root: Path | str, name: str) -> Any:
    """PyG's `HGBDataset(root, name)` that raises `DataNotOnDiskError` instead of downloading. A missing release folder
    (`<root>/<name lower>/raw`, PyG's `HGBDataset.raw_dir`) is refused before PyG runs, because PyG creates that folder
    before it calls `download()` and an empty folder would look like a present release to other readers."""
    import os

    from torch_geometric.datasets import HGBDataset

    raw = Path(root) / name.lower() / "raw"
    if not raw.is_dir() or not any(raw.iterdir()):
        raise DataNotOnDiskError(f"HGB {name}: no release under {raw}. This project never downloads data "
                                 "(card baselines-h2gb C7); place the release under the data root.")

    class OfflineHGBDataset(HGBDataset):
        def download(self) -> None:
            missing = [path for path in self.raw_paths if not os.path.exists(path)]
            raise DataNotOnDiskError(
                f"HGB {self.name}: raw file(s) missing under {self.raw_dir}: {missing}. This project never "
                "downloads data (card baselines-h2gb C7); place the release under the data root.")

    OfflineHGBDataset.__name__ = HGBDataset.__name__
    OfflineHGBDataset.__qualname__ = HGBDataset.__qualname__
    return OfflineHGBDataset(root=str(root), name=name)


def offline_tu_dataset(root: Path | str, name: str) -> Any:
    """PyG's `TUDataset(root, name)` that raises `DataNotOnDiskError` instead of downloading (a missing
    `<root>/<name>/raw`, PyG's `TUDataset.raw_dir`, is refused before PyG creates it)."""
    from torch_geometric.datasets import TUDataset

    raw = Path(root) / name / "raw"
    if not raw.is_dir() or not any(raw.iterdir()):
        raise DataNotOnDiskError(f"TU {name}: no release under {raw}. This project never downloads data "
                                 "(card baselines-h2gb C7); place the dataset under the data root.")

    class OfflineTUDataset(TUDataset):
        def download(self) -> None:
            raise DataNotOnDiskError(f"TU {self.name}: raw file(s) missing under {self.raw_dir}. This project never "
                                     "downloads data (card baselines-h2gb C7).")

    OfflineTUDataset.__name__ = TUDataset.__name__
    OfflineTUDataset.__qualname__ = TUDataset.__qualname__
    return OfflineTUDataset(root=str(root), name=name)


__all__ = ["DataNotOnDiskError", "offline_hgb_dataset", "offline_tu_dataset"]
