"""Part M3b B2: the H2GB node-classification loaders (`src/data/h2gb_nc.py`) on a synthetic fixture root.

The fixture writes, for each dataset, a processed file with H2GB's layout and storage format (MAG and OAG:
`torch.save(InMemoryDataset.collate([data]))`; IEEE-CIS and Pokec: `torch.save(data)`), with H2GB's empty
`OAG/raw`. The loaders must return H2GB's types, counts, labels and splits, with H2GB's transform
applied, without downloading or writing anything. Tests never reach the real data root.

Card mutation: re-enabling a download path (the offline subclass's `_download` hook removed) ->
`test_oag_with_empty_raw_never_downloads` fails (H2GB's `OAGDataset.download` is called).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO.parent / "docs" / "partM3b_evidence"


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _paper_graph(num_papers: int, author_type: str, field_type: str, rels: dict[str, tuple[str, str]],
                 feature_dim: int, num_classes: int, seed: int):
    from torch_geometric.data import HeteroData

    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    sizes = {"paper": num_papers, author_type: 5, field_type: 3}
    for node_type, size in sizes.items():
        data[node_type].x = torch.rand(size, feature_dim, generator=generator)
    y = torch.randint(0, num_classes, (num_papers,), generator=generator)
    y[-1] = -1  # one unlabelled paper
    data["paper"].y = y
    for name, (src, dst) in rels.items():
        count = 7
        data[src, name, dst].edge_index = torch.stack([
            torch.randint(0, sizes[src], (count,), generator=generator),
            torch.randint(0, sizes[dst], (count,), generator=generator)])
    return data, sizes


class _Fixture:
    """A data root with the processed files of the seven H2GB NC datasets (synthetic)."""

    def __init__(self) -> None:
        from torch_geometric.data import HeteroData, InMemoryDataset

        self.tmp = tempfile.TemporaryDirectory(prefix="h2gb_nc_fixture_")
        self.root = Path(self.tmp.name) / "data"
        nc = self.root / "NC"
        self.expected: dict[str, dict] = {}
        # MAG (ogbn-mag, mag-year): H2GB stores both directions itself (rev_ relations)
        for key, classes in (("ogbn-mag", 4), ("mag-year", 5)):
            data, sizes = _paper_graph(10, "author", "field_of_study",
                                       {"writes": ("author", "paper"), "cites": ("paper", "paper")},
                                       feature_dim=6, num_classes=classes, seed=len(key))
            data["paper", "rev_writes", "author"].edge_index = data["author", "writes", "paper"].edge_index.flip(0)
            masks = torch.zeros(3, 10, dtype=torch.bool)
            masks[0, :6], masks[1, 6:8], masks[2, 8:] = True, True, True
            data["paper"].train_mask, data["paper"].val_mask, data["paper"].test_mask = masks
            folder = nc / "mag" / key / "processed"
            folder.mkdir(parents=True)
            torch.save(InMemoryDataset.collate([data]), folder / "data.pt")
            self.expected[key] = {"data": data, "sizes": sizes, "masks": masks, "transform_adds": set()}
        # OAG: stored one direction only; H2GB's loader applies ToUndirected(merge=True)
        (nc / "OAG" / "raw").mkdir(parents=True)  # H2GB's raw folder exists and is empty
        from src.data.data_layout import H2GB_NC_DIRS

        for key in ("oag-cs", "oag-eng", "oag-chem"):
            folder_name = H2GB_NC_DIRS[key][1][0]  # the verified identity of B3 decides the folder
            data, sizes = _paper_graph(12, "author", "field",
                                       {"AP_write_first": ("author", "paper"),
                                        "PF_in_L1": ("paper", "field"), "PP_cite": ("paper", "paper")},
                                       feature_dim=8, num_classes=3, seed=len(key) + 7)
            data["paper"].year = torch.tensor([2014] * 8 + [2017] * 2 + [2018] * 2)
            labelled = data["paper"].y != -1
            data["paper"].train_mask = labelled & (data["paper"].year <= 2016)
            data["paper"].val_mask = labelled & (data["paper"].year == 2017)
            data["paper"].test_mask = labelled & (data["paper"].year >= 2018)
            masks = torch.stack([data["paper"].train_mask, data["paper"].val_mask, data["paper"].test_mask])
            folder = nc / "OAG" / folder_name / "processed"
            folder.mkdir(parents=True)
            torch.save(InMemoryDataset.collate([data]), folder / "data.pt")
            self.expected[key] = {"data": data, "sizes": sizes, "masks": masks,
                                  "transform_adds": {("paper", "rev_AP_write_first", "author"),
                                                     ("field", "rev_PF_in_L1", "paper")}}
        # IEEE-CIS: transaction -> card nodes; H2GB's loader sets the 80/10/10 split by node order
        data = HeteroData()
        data["transaction"].num_nodes = 20
        data["transaction"].x = torch.rand(20, 5, generator=torch.Generator().manual_seed(3))
        data["transaction"].y = torch.tensor([0, 1] * 10)
        data["transaction", "to", "card1"].edge_index = torch.stack([torch.arange(20), torch.arange(20) % 4])
        data["card1"].num_nodes = 4
        folder = nc / "IEEE-CIS" / "processed"
        folder.mkdir(parents=True)
        torch.save(data, folder / "data.pt")
        masks = torch.zeros(3, 20, dtype=torch.bool)
        masks[0, :16], masks[1, 16:18], masks[2, 18:] = True, True, True
        self.expected["ieee-cis-g"] = {"data": data, "sizes": {"transaction": 20, "card1": 4}, "masks": masks,
                                       "transform_adds": {("card1", "rev_to", "transaction")}}
        # Pokec: user friendship + user -> club; ToUndirected
        data = HeteroData()
        data["user"].num_nodes = 8
        data["user"].x = torch.rand(8, 3, generator=torch.Generator().manual_seed(4))
        data["user"].y = torch.tensor([0, 1, 0, 1, -1, 0, 1, 1])
        data["user", "has_friend", "user"].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        data["music"].num_nodes = 2
        data["user", "lists", "music"].edge_index = torch.tensor([[0, 5], [1, 0]])
        masks = torch.zeros(3, 8, dtype=torch.bool)
        masks[0, [0, 1, 2, 3]], masks[1, [5]], masks[2, [6, 7]] = True, True, True
        data["user"].train_mask, data["user"].val_mask, data["user"].test_mask = masks
        folder = nc / "Pokec" / "processed"
        folder.mkdir(parents=True)
        torch.save(data, folder / "data.pt")
        self.expected["h-pokec"] = {"data": data, "sizes": {"user": 8, "music": 2}, "masks": masks,
                                    "transform_adds": {("music", "rev_lists", "user")}}

    def close(self) -> None:
        self.tmp.cleanup()


class H2GBLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _Fixture()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.close()

    def test_every_dataset_loads_with_h2gb_types_counts_labels_and_split(self) -> None:
        from src.data.h2gb_nc import H2GB_NC_SPECS, load_h2gb_nc

        self.assertEqual(sorted(H2GB_NC_SPECS), sorted(self.fixture.expected))
        before = _tree(self.fixture.root)
        for key, expected in self.fixture.expected.items():
            with self.subTest(key=key):
                spec = H2GB_NC_SPECS[key]
                data = load_h2gb_nc(key, self.fixture.root)
                stored = expected["data"]
                self.assertEqual({t: int(data[t].num_nodes) for t in data.node_types}, expected["sizes"])
                for edge_type in stored.edge_types:  # every stored relation is kept, same edges
                    kept, original = data[edge_type].edge_index, stored[edge_type].edge_index
                    if edge_type[0] == edge_type[-1] and expected["transform_adds"]:
                        # ToUndirected makes a same-type relation symmetric (both orientations, coalesced)
                        pairs = set(map(tuple, original.t().tolist()))
                        self.assertEqual(set(map(tuple, kept.t().tolist())),
                                         pairs | {(v, u) for u, v in pairs})
                    else:
                        self.assertTrue(torch.equal(kept, original))
                self.assertEqual(set(data.edge_types) - set(stored.edge_types), expected["transform_adds"])
                target = data[spec.target_node_type]
                self.assertTrue(torch.equal(target.y, stored[spec.target_node_type].y))
                got = torch.stack([target.train_mask, target.val_mask, target.test_mask])
                self.assertTrue(torch.equal(got, expected["masks"]))
        self.assertEqual(_tree(self.fixture.root), before, "the loader wrote under the data root")

    def test_ieee_cis_features_are_normalised_as_h2gb_does(self) -> None:
        from src.data.h2gb_nc import load_h2gb_nc

        from torch_geometric.transforms import NormalizeFeatures

        data = load_h2gb_nc("ieee-cis-g", self.fixture.root)
        expected = NormalizeFeatures()(self.fixture.expected["ieee-cis-g"]["data"].clone())
        self.assertTrue(torch.equal(data["transaction"].x, expected["transaction"].x))  # master_loader.py:764
        self.assertFalse(torch.equal(data["transaction"].x,
                                     self.fixture.expected["ieee-cis-g"]["data"]["transaction"].x))

    def test_missing_processed_file_is_refused_without_download(self) -> None:
        from src.data.h2gb_nc import load_h2gb_nc

        empty = Path(tempfile.mkdtemp(prefix="h2gb_empty_")) / "data"
        (empty / "NC" / "OAG" / "raw").mkdir(parents=True)
        before = _tree(empty)
        for key in ("ogbn-mag", "oag-cs", "ieee-cis-g", "h-pokec"):
            with self.subTest(key=key), self.assertRaisesRegex(FileNotFoundError, "never downloads"):
                load_h2gb_nc(key, empty)
        self.assertEqual(_tree(empty), before)

    def test_oag_with_empty_raw_never_downloads(self) -> None:
        """H2GB's OAGDataset downloads when `raw/` is empty (oag_dataset.py:87-88; PyG _download)."""
        from src.data.h2gb_nc import load_h2gb_nc
        from src.baselines.runtime.vendor_paths import pushd_sys_path

        with pushd_sys_path("h2gb"):
            import H2GB.datasets.oag_dataset as oag  # type: ignore[import-not-found]
        with mock.patch.object(oag.OAGDataset, "download",
                               side_effect=AssertionError("download called")) as download:
            data = load_h2gb_nc("oag-cs", self.fixture.root)
        download.assert_not_called()
        self.assertEqual(int(data["paper"].num_nodes), 12)


    def test_build_dataset_meets_the_h2snn_contract(self) -> None:
        """`build_dataset` (the H2S-NN and baseline-bundle path) reads the loader: target type, labels and
        masks of the fixture, mini-batch mode (no whole-graph draw)."""
        from src.configs.config import TrainConfig
        from src.data.dataset import build_dataset

        config = TrainConfig.from_dict({"device": "cpu", "data": {
            "dataset_name": "h-pokec", "root": str(self.fixture.root), "batch_size": 32,
            "wisp_walks_per_anchor": 4}})
        self.assertEqual((config.data.target_node_type, config.data.training_mode), ("user", "minibatch"))
        data = build_dataset(config)
        expected = self.fixture.expected["h-pokec"]
        self.assertEqual(data.target_node_type, "user")
        self.assertEqual(data.target_indices.numel(), 8)
        self.assertTrue(torch.equal(data.target_y, expected["data"]["user"].y))
        self.assertTrue(torch.equal(torch.stack([data.target_train_mask, data.target_val_mask,
                                                 data.target_test_mask]), expected["masks"]))
        self.assertEqual(data.face_groups, ())
        self.assertEqual(data.num_classes, 2)


class RegistryTests(unittest.TestCase):
    def test_loaded_datasets_say_adapter_pending_until_wired(self) -> None:
        from src.baselines.registry import (H2GB_ADAPTER_PENDING_REASON, H2GB_NC_ADAPTER_PENDING,
                                            H2GB_NC_CELL_REASONS, H2GB_NC_LOADED, H2GB_NC_WIRED_ADAPTERS,
                                            H2GB_NC_WIRED_CELLS, MODELS, unsupported_reason)

        for spec in MODELS:
            if "node_classification" not in spec.tasks:
                continue
            explicit = {d: r for t, d, r in spec.unsupported if t == "node_classification"}
            for dataset in H2GB_NC_LOADED:
                reason = unsupported_reason(spec.name, "node_classification", dataset)
                if dataset in explicit:  # the author's N/A cells (card baselines-h2gb: SeHGNN on H-Pokec)
                    self.assertEqual(reason, explicit[dataset])
                elif spec.name in H2GB_NC_WIRED_ADAPTERS or (spec.name, dataset) in H2GB_NC_WIRED_CELLS:
                    self.assertIsNone(reason)
                else:
                    # Part M3b B8 / card baselines-h2gb: a per-cell reason, else the adapter's own reason
                    self.assertEqual(reason, H2GB_NC_CELL_REASONS.get(
                        (spec.name, dataset), H2GB_NC_ADAPTER_PENDING.get(spec.name, H2GB_ADAPTER_PENDING_REASON)))
                    self.assertTrue(reason.startswith(("adapter pending: ", "adapter in place, pending author",
                                                       "pending author decision")))

    def test_nc_cell_is_runnable_only_with_a_loader_and_a_count_record(self) -> None:
        from src.baselines.registry import DATASET_PENDING, DATA_PENDING_REASON, H2GB_NC
        from src.data.h2gb_nc import H2GB_NC_SPECS, RAW_ONLY

        counted = json.loads((EVIDENCE / "h2gb_nc_counts.json").read_text())
        for dataset in H2GB_NC:
            with self.subTest(dataset=dataset):
                pending = ("node_classification", dataset) in DATASET_PENDING
                # card baselines-h2gb C2: PDNS is loaded by H2GB's code (raw only; its counts equal Table 1 in
                # docs/partM3b_evidence/b5_pdns_check.json and the C2 check)
                ready = (dataset in H2GB_NC_SPECS and dataset in counted.get("verified", [])) or dataset in RAW_ONLY
                self.assertEqual(pending, not ready)
                if pending:
                    self.assertTrue(DATASET_PENDING[("node_classification", dataset)])
        self.assertNotIn(("node_classification", "pdns"), DATASET_PENDING)  # card baselines-h2gb C2
        self.assertTrue(DATA_PENDING_REASON)


if __name__ == "__main__":
    unittest.main()
