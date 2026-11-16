"""Card 3 Part A (HetSheaf), Slim tests: one NC and one LP step finite and moving; preprocessing equal to their
`datasets/hgb.py:51-82` chain with `SubsetTrainSplit` removed; the shared split masks unchanged by the chain
(label-route protection; mutation: `SubsetTrainSplit` in our chain -> `test_shared_masks_survive_the_chain` fails);
the import route leaves no `lightning` / `omegaconf` / `hetsheaf` module behind registry or other adapter imports
(design-session condition 1 of S-A0-1). Synthetic graphs only; no test reads the data root."""

from __future__ import annotations

import contextlib
import importlib
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def synthetic_hgb() -> Any:
    """A small HGB-like release: target `paper` (features, labels, split masks), `author` (other width), `term`
    (no features); a duplicated `cite` edge and a self-loop so every step of their chain acts."""
    from torch_geometric.data import HeteroData

    g = torch.Generator().manual_seed(0)
    data = HeteroData()
    data["paper"].x = torch.rand(8, 6, generator=g) * 3.0
    data["paper"].y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    data["paper"].train_mask = torch.tensor([1, 1, 0, 1, 0, 0, 1, 0], dtype=torch.bool)
    data["paper"].val_mask = torch.tensor([0, 0, 1, 0, 0, 1, 0, 0], dtype=torch.bool)
    data["paper"].test_mask = torch.tensor([0, 0, 0, 0, 1, 0, 0, 1], dtype=torch.bool)
    data["author"].x = torch.rand(5, 4, generator=g)
    data["term"].num_nodes = 4
    pa = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0], [0, 1, 2, 3, 4, 0, 1, 2, 3]])
    data["paper", "to", "author"].edge_index = pa
    data["author", "to", "paper"].edge_index = pa.flip(0)
    data["paper", "cite", "paper"].edge_index = torch.tensor([[0, 1, 2, 2, 3, 5], [1, 2, 3, 3, 3, 6]])
    pt = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 0, 1, 2, 3]])
    data["paper", "to", "term"].edge_index = pt
    data["term", "to", "paper"].edge_index = pt.flip(0)
    return data


@contextlib.contextmanager
def their_hgb_module() -> Iterator[Any]:
    """Their `hetsheaf.datasets.hgb` imported through the adapter's route; its `import lightning as L`
    (`hgb.py:9`, base class `L.LightningDataModule`) is given `pytorch_lightning`'s class for this import only."""
    import pytorch_lightning

    from src.baselines.adapters import hetsheaf

    hetsheaf._vendor()
    with hetsheaf._lightning_alias():
        sys.modules["lightning"].LightningDataModule = pytorch_lightning.LightningDataModule
        module = importlib.import_module("hetsheaf.datasets.hgb")
    yield module


def run_their_prepare_data(release: Any, target: str) -> Any:
    """Their `HGBBaseDataModule.prepare_data` (`datasets/hgb.py:50-91`) on `release`, with `SubsetTrainSplit`
    replaced by the identity (H-2), `HGBDataset` returning `release` through their transform, and the Lightning
    loader replaced by a capture of the homogeneous graph it receives."""
    with their_hgb_module() as hgb:
        captured: dict[str, Any] = {}

        class Identity:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __call__(self, data: Any) -> Any:
                return data

        def fake_dataset(root: str, name: str, transform: Any) -> list[Any]:
            return [transform(release.clone())]

        def capture(data: Any, **kwargs: Any) -> Any:
            captured["data"] = data
            return SimpleNamespace()

        saved = (hgb.SubsetTrainSplit, hgb.HGBDataset, hgb.LightningNodeData)
        hgb.SubsetTrainSplit, hgb.HGBDataset, hgb.LightningNodeData = Identity, fake_dataset, capture
        try:
            module = hgb.HGBBaseDataModule(target=target, num_classes=3, dataset="ACM", homogeneous=True)
            module.prepare_data()
        finally:
            hgb.SubsetTrainSplit, hgb.HGBDataset, hgb.LightningNodeData = saved
        return captured["data"]


def tuned(**values: Any) -> Any:
    from src.baselines.protocol import BaselineParams

    return BaselineParams(values=values, source="tuned")


def parameters_snapshot(*modules: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for module in modules for p in module.parameters()]


def assert_step_finite_and_moves(case: unittest.TestCase, loss: torch.Tensor, optimizer: Any,
                                 modules: tuple[torch.nn.Module, ...]) -> None:
    before = parameters_snapshot(*modules)
    case.assertTrue(torch.isfinite(loss).item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = parameters_snapshot(*modules)
    case.assertTrue(all(torch.isfinite(p).all() for p in after))
    case.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)), "no parameter moved")


class PreprocessingTests(unittest.TestCase):
    def test_chain_equals_their_hgb_chain_without_subset_train_split(self) -> None:
        from src.baselines.adapters.hetsheaf import prepare_node_graph

        release = synthetic_hgb()
        theirs = run_their_prepare_data(release, "paper")
        ours = prepare_node_graph(release, "paper", train_mask=release["paper"].train_mask,
                                  val_mask=release["paper"].val_mask, test_mask=release["paper"].test_mask).graph
        self.assertTrue(torch.equal(ours.x, theirs.x.float()))
        self.assertTrue(torch.equal(ours.edge_index, theirs.edge_index))
        self.assertTrue(torch.equal(ours.node_type, theirs.node_type))
        self.assertTrue(torch.equal(ours.edge_type, theirs.edge_type))
        self.assertEqual((ours.num_node_types, ours.num_edge_types),
                         (int(theirs.num_node_types), int(theirs.num_edge_types)))

    def test_shared_masks_survive_the_chain(self) -> None:
        """Label-route protection: the shared split comes back unchanged (a split-changing transform such as their
        `SubsetTrainSplit` would carve a new validation set out of the training nodes)."""
        from src.baselines.adapters.hetsheaf import prepare_node_graph

        release = synthetic_hgb()
        masks = {key: release["paper"][key].clone() for key in ("train_mask", "val_mask", "test_mask")}
        prepared = prepare_node_graph(release, "paper", **masks)
        for key, expected in masks.items():
            with self.subTest(mask=key):
                self.assertTrue(torch.equal(getattr(prepared, key), expected))
        self.assertEqual(prepared.target_rows.numel(), 8)


class StepTests(unittest.TestCase):
    def test_one_node_classification_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters.hetsheaf import HetSheafNodeClassifier, build_encoder, prepare_node_graph

        torch.manual_seed(0)
        release = synthetic_hgb()
        prepared = prepare_node_graph(release, "paper", train_mask=release["paper"].train_mask,
                                      val_mask=release["paper"].val_mask, test_mask=release["paper"].test_mask)
        params = tuned(sheaf="general/attention_type_concat", d=3, layers=2, hidden_channels=8, dropout=0.1,
                       input_dropout=0.1, lr=1e-2)
        device = torch.device("cpu")
        model = HetSheafNodeClassifier(build_encoder(prepared.graph, params, device, output_dim=3), 3,
                                       prepared.target_rows)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        logits = model(prepared.graph.encoder_input(device))
        self.assertEqual(tuple(logits.shape), (8, 3))
        train = prepared.train_mask.nonzero().view(-1)
        loss = torch.nn.functional.cross_entropy(logits[train], release["paper"].y[train])
        assert_step_finite_and_moves(self, loss, optimizer, (model,))

    def test_one_link_prediction_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters.hetsheaf import HetSheafLinkEncoder, build_encoder, prepare_link_graph
        from src.baselines.heads import SharedLinkHead

        torch.manual_seed(0)
        # A flattened message graph in the bundle's layout (PubMed-like: four typed relations, one type without
        # features); every LP branch of the adapter reads the same fields.
        x = torch.zeros(9, 5)
        x[:4, :5] = torch.rand(4, 5)
        x[4:7, :3] = torch.rand(3, 3)
        data = SimpleNamespace(
            x=x, node_type_names=("gene", "disease", "species"),
            node_slices={"gene": (0, 4), "disease": (4, 7), "species": (7, 9)},
            feature_dims={"gene": 5, "disease": 3, "species": 0},
            relation_tuples=(("gene", "rel_0", "gene"), ("gene", "rel_1", "disease"), ("species", "rel_2", "gene")),
            edge_index=torch.tensor([[0, 1, 2, 0, 1, 3, 7, 8], [1, 2, 3, 4, 5, 6, 0, 2]]),
            edge_type=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]))
        device = torch.device("cpu")
        params = tuned(sheaf="bundle/type_concat", d=2, layers=2, hidden_channels=8, lr=1e-2)
        for dataset in ("pubmed", "lastfm", "acm_pa"):
            with self.subTest(branch=dataset):
                graph, global_id = prepare_link_graph(dataset, data)
                self.assertTrue(torch.equal(global_id.sort().values, torch.arange(9)))
                encoder = HetSheafLinkEncoder(build_encoder(graph, params, device, output_dim=64), global_id)
                head = SharedLinkHead("distmult", num_relations=1, dim=encoder.embed_dim)
                optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=1e-2)
                embeddings = encoder(graph.encoder_input(device))
                self.assertEqual(embeddings.size(0), 9)
                pairs = torch.tensor([[0, 1, 2, 5], [1, 3, 6, 8]])
                scores = head(embeddings, pairs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, torch.tensor([1.0, 1.0, 0.0, 0.0]))
                assert_step_finite_and_moves(self, loss, optimizer, (encoder, head))


class ImportRouteTests(unittest.TestCase):
    def test_registry_and_another_adapter_leave_no_hetsheaf_or_lightning_module(self) -> None:
        """Design-session condition 1 (S-A0-1): the empty packages and the `lightning.pytorch` alias exist only
        through the HetSheaf adapter; the alias is gone again after SheafPool's module is imported."""
        script = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(PROJECT_ROOT)!r})
            import src.baselines.registry as registry
            import src.baselines.adapters.nsd  # another adapter
            registry.iter_cells()
            bad = sorted(m for m in sys.modules if m.split('.')[0] in ('lightning', 'omegaconf', 'hetsheaf'))
            print('after_registry', bad)
            from src.baselines.adapters import hetsheaf
            hetsheaf._vendor_gc()
            print('after_hetsheaf_gc', sorted(m for m in sys.modules if m.split('.')[0] in ('lightning', 'omegaconf')))
            """)
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=PROJECT_ROOT,
                                timeout=600)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        lines = dict(line.split(" ", 1) for line in result.stdout.splitlines() if line.startswith("after_"))
        self.assertEqual(lines["after_registry"], "[]")
        self.assertEqual(lines["after_hetsheaf_gc"], "[]")


if __name__ == "__main__":
    unittest.main()
