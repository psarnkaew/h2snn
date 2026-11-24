"""Hetero²Net on the H2GB node-classification datasets (card 1c F4; author decision G-5 as amended; inputs rule B).

Their mini-batch scheme (VERIFIED-CODE, Hetero2Net `4913394`): `NeighborLoader` with `num_neighbors=[10, 10]`
(`main.py:39, 147-175`); training batches from `train_data` (validation and test labels hidden in `y_emb`), validation
batches from `train_data`, test batches from `test_data` (test labels hidden; validation labels visible)
(`main.py:125-146, 165-175, 202-206`); one training step = their `train_mini_batch` body (`hetero2net/utils.py:14-60`:
metapath-drop when a list exists, the batch's own `y_emb` hidden with ratio `p`, cross-entropy on the first
`batch_size` targets plus the disentanglement terms). Imposed by the protocol (class B): the batch size from the shared
range and 8,192 training targets per epoch.

Inputs: datasets without a metapath list run with `metapaths = None` and the featureless types as `torch.eye`, as their
FreeBase branch (`hetero2net/dataset.py:48-53`); its lists follow no single rule (card 1c F0). ogbn-mag / mag-year read
PyG's MetaPath2Vec embeddings for author / institution / field (`dataset.py:54-63`; PyG `ogb_mag.py:150-153`) from
`<data root>/NC/mag/hetero2net/mag_metapath2vec_emb.zip` (author download); the zip's sha256 is written beside it at
first use and a different file is refused; without the file the cell waits (`AuthorFileMissing`).
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any, Iterator

import torch

#: Their MAG metapaths (`hetero2net/dataset.py:57-62`), on H2GB's MAG type names (the same four types).
MAG_METAPATHS = [
    [("paper", "author"), ("author", "institution"), ("institution", "author"), ("author", "paper")],
    [("author", "paper"), ("paper", "field_of_study"), ("field_of_study", "paper"), ("paper", "author")],
]
MAG_DATASETS = frozenset({"ogbn-mag", "mag-year"})
EMBEDDING_ZIP = ("NC", "mag", "hetero2net", "mag_metapath2vec_emb.zip")
NUM_NEIGHBORS = [10, 10]


class AuthorFileMissing(FileNotFoundError):
    """A file the author downloads is not in place yet; the cell waits."""


def read_mag_embeddings(data_root: Path) -> dict[str, torch.Tensor]:
    """PyG's MetaPath2Vec embeddings from the author's zip (sha256 recorded at first use, a different file refused)."""
    path = Path(data_root).joinpath(*EMBEDDING_ZIP)
    if not path.is_file():
        raise AuthorFileMissing(f"waiting for {path} (author download: https://data.pyg.org/datasets/"
                                "mag_metapath2vec_emb.zip)")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = path.with_name(path.name + ".sha256")
    if record.exists():
        if record.read_text().strip() != digest:
            raise RuntimeError(f"{path}: sha256 {digest} differs from the one recorded at first use ({record})")
    else:
        record.write_text(digest + "\n")
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("mag_metapath2vec_emb.pt"))
        return torch.load(io.BytesIO(archive.read(name)), map_location="cpu")


def prepare_inputs(data: Any, dataset: str, data_root: Path | None) -> list | None:
    """Their inputs on H2GB's graph, in place; returns the metapath list (None: FreeBase branch)."""
    if dataset in MAG_DATASETS:
        for key, value in read_mag_embeddings(Path(data_root)).items():
            if key != "paper":
                data[key].x = value
        return MAG_METAPATHS
    for node_type in data.node_types:
        if data[node_type].get("x") is None:
            data[node_type].x = torch.eye(int(data[node_type].num_nodes))
    return None


def label_channel(data: Any, node_type: str, num_classes: int, val_mask: torch.Tensor,
                  test_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(`train_data` y_emb, `test_data` y_emb) as `main.py:129-144`: unknown and test labels hidden in both; validation
    labels hidden in the training copy (also used for validation batches)."""
    y = data[node_type].y.clone().long()
    y[y == -1] = num_classes
    y[test_mask] = num_classes
    test_y = y.clone()
    y[val_mask] = num_classes
    return y, test_y


def epoch_batches(loader: Any, state: dict, steps: int) -> Iterator[Any]:
    """`steps` batches of a shuffled loader, its pass continued across epochs (a new pass when one ends)."""
    for _ in range(steps):
        try:
            yield next(state["it"])
        except (KeyError, StopIteration):
            state["it"] = iter(loader)
            yield next(state["it"])


def loaders(train_data: Any, test_data: Any, node_type: str, batch_size: int) -> tuple[Any, Any, Any]:
    """Card 21 A2: `LOADER_WORKERS` workers each; the training loader's persist (its pass continues across epochs,
    `epoch_batches`), an evaluation loader's live for one pass."""
    from torch_geometric.loader import NeighborLoader

    from src.baselines.minibatch import LOADER_WORKERS

    def make(data: Any, mask: torch.Tensor, shuffle: bool) -> Any:
        return NeighborLoader(data, num_neighbors=NUM_NEIGHBORS, batch_size=batch_size,
                              input_nodes=(node_type, mask), shuffle=shuffle, num_workers=LOADER_WORKERS,
                              persistent_workers=shuffle and LOADER_WORKERS > 0)

    return (make(train_data, train_data[node_type].train_mask, True),
            make(train_data, train_data[node_type].val_mask, False),   # main.py:165-174: validation on train_data
            make(test_data, test_data[node_type].test_mask, False))    # main.py:155-163: test on test_data


def index_loader(data: Any, node_type: str, ids: torch.Tensor, batch_size: int) -> Any:
    """Their evaluation NeighborLoader over explicit target ids (card 1f B1a validation subset), no shuffling."""
    from torch_geometric.loader import NeighborLoader

    from src.baselines.minibatch import LOADER_WORKERS

    return NeighborLoader(data, num_neighbors=NUM_NEIGHBORS, batch_size=batch_size, input_nodes=(node_type, ids),
                          shuffle=False, num_workers=LOADER_WORKERS)  # card 21 A2: workers for one pass


def seed_logits(model: Any, batch: Any, node_type: str) -> torch.Tensor:
    """Their `evaluate_mini_batch` forward (`hetero2net/utils.py:108-118`): logits of the first `batch_size` targets."""
    out = model(batch.x_dict, batch.edge_index_dict, batch[node_type].get("y_emb"))
    out = out[0] if isinstance(out, tuple) else out
    out = out[node_type] if isinstance(out, dict) else out
    return out[: int(batch[node_type].batch_size)]


def build_model(data: Any, node_type: str, num_classes: int, params: Any) -> Any:
    from src.baselines.runtime import pushd_sys_path

    with pushd_sys_path("hetero2net"):
        from hetero2net.models import HeteroGNN  # type: ignore[import-not-found]

        model = HeteroGNN(data.metadata(), in_channels=int(data[node_type].x.size(1)),
                          hidden_channels=int(params.get("hidden", 128)), out_channels=num_classes,
                          dropout=float(params.get("dropout", 0.0)), project=True)
    model.node_type = node_type
    return model


__all__ = ["AuthorFileMissing", "MAG_METAPATHS", "NUM_NEIGHBORS", "build_model", "epoch_batches", "label_channel",
           "loaders", "prepare_inputs", "read_mag_embeddings", "seed_logits"]
