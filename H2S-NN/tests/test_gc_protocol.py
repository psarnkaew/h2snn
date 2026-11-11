"""Part M, M5: graph classification as HetSheaf (spec v1.15 §6 GC, J6).

Readout: `[x*, h0]` -> pooling (mean, max, add, HetSheaf's attention) -> one Linear; no BatchNorm,
no MLP. Protocol: TU via PyG (use_node_attr False), node types = one-hot label argmax, constant
feature 1, one randperm(seed 42) 80/10/10 split for every training seed, batch size 1, early
stopping on validation loss (patience), test with the final weights."""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from src.configs.config import TrainConfig
from src.data.data_layout import REPO_DATA_ROOT, gc_root
from src.models.graph_readout import AttentionPooling, GraphReadout

X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, -6.0]])
BATCH = torch.tensor([0, 0, 1])


class PoolingTests(unittest.TestCase):
    def test_mean_max_add_by_hand(self) -> None:
        expected = {"mean": [[2.0, 3.0], [5.0, -6.0]], "max": [[3.0, 4.0], [5.0, -6.0]],
                    "add": [[4.0, 6.0], [5.0, -6.0]]}
        for pooling, want in expected.items():
            readout = GraphReadout(2, 3, pooling)
            with torch.no_grad():
                readout.decoder.weight.copy_(torch.eye(3, 2))
                readout.decoder.bias.zero_()
            out = readout(X, BATCH, 2)
            self.assertTrue(torch.allclose(out[:, :2], torch.tensor(want)), pooling)
            self.assertTrue(torch.equal(out[:, 2], torch.zeros(2)))

    def test_attention_pooling_by_hand(self) -> None:
        """Scores s = W2 tanh(W1 x + b1): W1 = I, b1 = 0, W2 = I (2 heads); per head a softmax
        over the graph's nodes, weighted sum, mean over heads (HetSheaf pooling.py)."""
        pool = AttentionPooling(2, heads=2)
        with torch.no_grad():
            pool.attention[0].weight.copy_(torch.eye(2))
            pool.attention[0].bias.zero_()
            pool.attention[2].weight.copy_(torch.eye(2))
        out = pool(X, BATCH, 2)
        expected = []
        for nodes in ([0, 1], [2]):
            heads = []
            for h in range(2):
                scores = [math.tanh(float(X[n, h])) for n in nodes]
                z = sum(math.exp(s) for s in scores)
                heads.append(sum(math.exp(s) / z * X[n] for s, n in zip(scores, nodes)))
            expected.append((heads[0] + heads[1]) / 2)
        self.assertTrue(torch.allclose(out, torch.stack(expected), atol=1e-6))

    def test_readout_is_pooling_then_one_linear(self) -> None:
        """No BatchNorm, no MLP: the only parameterised layer after pooling is one Linear
        (mutation: re-inserting BatchNorm fails here)."""
        for pooling in ("mean", "max", "add", "attention"):
            readout = GraphReadout(4, 3, pooling)
            norms = [m for m in readout.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
            self.assertEqual(norms, [], pooling)
            linears = [name for name, m in readout.named_modules() if isinstance(m, nn.Linear)
                       and not name.startswith("attention")]
            self.assertEqual(linears, ["decoder"], pooling)
            x = torch.randn(5, 4)
            batch = torch.tensor([0, 0, 1, 1, 1])
            readout.train()  # BatchNorm would change the output between train and eval
            out_train = readout(x, batch, 2)
            readout.eval()
            out_eval = readout(x, batch, 2)
            self.assertTrue(torch.equal(out_train, out_eval), pooling)
            pooled = (readout.attention(x, batch, 2) if pooling == "attention" else
                      __import__("src.models.graph_readout", fromlist=["pool_nodes"]).pool_nodes(
                          x, batch, 2, pooling))
            self.assertTrue(torch.allclose(out_eval, readout.decoder(pooled)), pooling)

    def test_config_pooling_values(self) -> None:
        for pooling in ("mean", "max", "add", "attention"):
            config = TrainConfig.from_dict({"data": {"dataset_name": "mutag"},
                                            "model": {"graph_pooling": pooling}})
            self.assertEqual(config.model.graph_pooling, pooling)
        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"data": {"dataset_name": "mutag"},
                                   "model": {"graph_pooling": "sum"}})


@unittest.skipUnless((gc_root(REPO_DATA_ROOT) / "MUTAG" / "raw").exists(), "TU data not present")
class ProtocolOnMutagTests(unittest.TestCase):
    def _fold(self, seed: int):
        from src.data.graph_classification_dataset import build_graph_classification_dataset

        return build_graph_classification_dataset(TrainConfig.from_dict({
            "device": "cpu", "seed": seed, "data": {"dataset_name": "mutag", "root":
                                                    str(REPO_DATA_ROOT),
                                                    "wisp_walks_per_anchor": 8}}))

    @staticmethod
    def _ids(loader) -> list[int]:
        return [int(g) for batch in loader for g in batch.graph_id.view(-1).tolist()]

    def test_split_is_hetsheaf_randperm_seed_42_for_every_training_seed(self) -> None:
        n = 188
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
        want = (perm[:150], perm[150:168], perm[168:])
        for seed in (0, 3):
            fold = self._fold(seed)
            self.assertEqual(sorted(self._ids(fold.train_loader)), sorted(want[0]), seed)
            self.assertEqual(self._ids(fold.val_loader), want[1], seed)
            self.assertEqual(self._ids(fold.test_loader), want[2], seed)
            self.assertEqual(len(want[2]), 20)

    def test_constant_feature_and_batch_size_one(self) -> None:
        fold = self._fold(0)
        for loader in (fold.train_loader, fold.val_loader, fold.test_loader):
            self.assertEqual(loader.batch_size, 1)
        batch = next(iter(fold.val_loader))
        self.assertEqual(batch.num_graphs, 1)
        self.assertTrue(torch.equal(batch.x, torch.ones(batch.num_nodes, 1)))
        self.assertEqual(fold.dims.in_channels, 1)

    def test_node_and_edge_types_are_the_one_hot_argmax(self) -> None:
        from torch_geometric.datasets import TUDataset

        from src.data.graph_classification_dataset import _load_tu_type_tensors

        for name in ("MUTAG", "PROTEINS", "ENZYMES", "NCI1"):
            if not (gc_root(REPO_DATA_ROOT) / name).exists():
                continue
            with self.subTest(dataset=name):
                dataset = TUDataset(root=str(gc_root(REPO_DATA_ROOT)), name=name)
                # use_node_attr False: x holds only the one-hot label columns
                self.assertEqual(dataset.num_node_features, dataset.num_node_labels)
                data = [dataset[i] for i in range(len(dataset))]
                node_types, edge_types, _, _ = _load_tu_type_tensors(REPO_DATA_ROOT, name, data)
                self.assertTrue(torch.equal(torch.cat(node_types),
                                            torch.cat([d.x.argmax(1) for d in data])))
                if data[0].edge_attr is not None:
                    self.assertTrue(torch.equal(torch.cat(edge_types),
                                                torch.cat([d.edge_attr.argmax(1) for d in data])))


class _Stub:
    def __getattr__(self, name):
        return lambda *args, **kwargs: False


class EarlyStoppingOnValidationLossTests(unittest.TestCase):
    def test_stops_on_validation_loss_and_reports_final_weights(self) -> None:
        """Validation loss 1.0, 0.9, 0.95, 0.96, 0.97 with patience 3: stops after epoch 4; the
        reported test record is epoch 4's (final weights), whatever the validation Macro-F1."""
        import tempfile
        from pathlib import Path

        from src.trainers.graph_classification_trainer import GraphClassificationTrainer

        losses = [1.0, 0.9, 0.95, 0.96, 0.97, 0.5, 0.4]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig.from_dict({"device": "cpu", "epochs": 7,
                                         "early_stopping_patience": 3, "output_dir": tmp,
                                         "data": {"dataset_name": "mutag"}})
            trainer = GraphClassificationTrainer.__new__(GraphClassificationTrainer)
            trainer.cfg, trainer.device, trainer.scheduler = cfg, torch.device("cpu"), None
            trainer.model = nn.Linear(1, 1)
            trainer.optimizer, trainer.logger, trainer.checkpointer = None, _Stub(), _Stub()
            trainer._best_test_metrics, trainer._best_epoch = None, -1
            calls: list[int] = []

            def run_epoch(loader, epoch, train):
                calls.append(epoch)
                metrics = {"loss": losses[epoch], "auroc": 0.5, "acc": 0.5,
                           "macro_f1": 0.9 if epoch == 1 else 0.1, "micro_f1": 0.5}
                return metrics if loader != "test" else {**metrics, "macro_f1": epoch / 10}

            trainer._run_epoch = run_epoch
            fold = type("F", (), {"train_loader": "train", "val_loader": "val",
                                  "test_loader": "test", "run_index": 0, "num_runs": 1,
                                  "fold_index": 0, "num_folds": 1, "repetition_index": 0,
                                  "num_repetitions": 1, "split_seed": 42})()
            Path(tmp, cfg.exp_name).mkdir(parents=True, exist_ok=True)
            trainer.fit(fold)
            self.assertEqual(max(calls), 4)
            self.assertEqual(trainer._best_test_metrics["final_epoch"], 4)
            self.assertAlmostEqual(trainer._best_test_metrics["test_macro_f1"], 0.4)


if __name__ == "__main__":
    unittest.main()
