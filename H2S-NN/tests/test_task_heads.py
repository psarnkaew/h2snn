"""The three task heads on the H2S-NN encoder (plan §4.8, decision D6).

- node classification: golden outputs captured before the v1 removal must be bit-identical;
- link prediction: every decoder trains one step; no path cell and no message-graph edge is
  a validation or test positive;
- graph classification: logits are invariant to a node permutation within a graph; one
  training step is finite.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.configs.config import TrainConfig
from src.data.data_layout import lp_root

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SCRIPT = REPO_ROOT / "docs" / "heads_evidence" / "v2_golden.py"
GOLDEN_REFERENCE = REPO_ROOT / "docs" / "heads_evidence" / "v2_golden.pt"
SMALL_MODEL = {"d": 8, "d_r": 4, "K": 2, "classifier_hidden_dim": 16,
               "face_mlp_hidden_dim": 16, "edge_mlp_hidden_dim": 16}


class NodeClassificationGoldenTests(unittest.TestCase):
    def test_outputs_and_gradients_are_bit_identical_to_the_pre_removal_capture(self) -> None:
        """Captured from the pre-removal code (13 cases: toy ACM/DBLP/IMDB, a paper-author
        graph with path lengths 2 and 4, dense maps, no checkpointing, every ablation)."""
        self.assertTrue(GOLDEN_REFERENCE.exists(), GOLDEN_REFERENCE)
        spec = importlib.util.spec_from_file_location("v2_golden", GOLDEN_SCRIPT)
        golden = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(golden)  # type: ignore[union-attr]
        compared, mismatches, messages = golden.compare(GOLDEN_REFERENCE)
        self.assertEqual(mismatches, 0, messages[:10])
        self.assertEqual(compared, 844)


# ----------------------------------------------------------------------------- link prediction
def _write_tiny_lastfm(root: Path) -> None:
    """HGB local format: users (type 0), artists (type 1), tags (type 2); rel 0 user->artist
    (the target), rel 1 user->user, rel 2 artist->tag; link.dat.test holds rel-0 pairs that
    are not in link.dat."""
    folder = lp_root(root) / "LastFM"
    folder.mkdir(parents=True)
    generator = torch.Generator().manual_seed(0)
    users, artists, tags = range(0, 12), range(12, 24), range(24, 30)
    with (folder / "node.dat").open("w", encoding="utf-8") as handle:
        for node_type, ids in enumerate((users, artists, tags)):
            for raw in ids:
                features = ",".join(f"{float(v):.4f}" for v in torch.rand(3, generator=generator))
                handle.write(f"{raw}\tn{raw}\t{node_type}\t{features}\n")
    train_pairs: set[tuple[int, int]] = set()
    test_pairs: set[tuple[int, int]] = set()
    for user in users:
        chosen = torch.randperm(12, generator=generator)[:6].tolist()
        for rank, artist in enumerate(chosen):
            (test_pairs if rank == 5 else train_pairs).add((user, 12 + artist))
    with (folder / "link.dat").open("w", encoding="utf-8") as handle:
        for user, artist in sorted(train_pairs):
            handle.write(f"{user}\t{artist}\t0\t1.0\n")
        for user in users:
            handle.write(f"{user}\t{(user + 1) % 12}\t1\t1.0\n")
        for artist in artists:
            handle.write(f"{artist}\t{24 + artist % 6}\t2\t1.0\n")
    with (folder / "link.dat.test").open("w", encoding="utf-8") as handle:
        for user, artist in sorted(test_pairs):
            handle.write(f"{user}\t{artist}\t0\t1.0\n")


def _link_config(root: Path, decoder_type: str = "dot") -> TrainConfig:
    return TrainConfig.from_dict({
        "device": "cpu", "checkpoint_metric": "auroc", "model": SMALL_MODEL,
        "data": {"dataset_name": "lastfm", "root": str(root), "decoder_type": decoder_type},
    })


def _pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in edge_index.t().tolist()}


def _leaked_pairs(data) -> set[tuple[int, int]]:
    """Held-out (validation or test) positives that appear, in either direction, as a
    message-graph edge or as an edge of a path cell."""
    held_out = _pairs(data.val_pos_edge_index) | _pairs(data.test_pos_edge_index)
    held_out |= {(b, a) for a, b in held_out}
    used = _pairs(data.edge_index)
    for group in data.face_groups:
        nodes = group.face_node_index
        for position in range(nodes.size(1) - 1):
            used |= _pairs(torch.stack([nodes[:, position], nodes[:, position + 1]]))
    return held_out & used


class LinkPredictionHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write_tiny_lastfm(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _data(self, decoder_type: str = "dot"):
        from src.data.link_prediction_dataset import build_link_prediction_dataset

        return build_link_prediction_dataset(_link_config(self.root, decoder_type))

    def test_no_path_cell_or_message_edge_is_a_held_out_positive(self) -> None:
        data = self._data()
        self.assertGreater(data.val_pos_edge_index.size(1), 0)
        self.assertGreater(data.test_pos_edge_index.size(1), 0)
        self.assertGreater(sum(group.num_faces for group in data.face_groups), 0)
        # Path cells only use message-graph edges, stored along the path.
        for group in data.face_groups:
            # (signed hops: a -1 hop's fact is stored against the path direction, D7)
            edges = data.edge_index[:, group.face_edge_index]
            backward = group.face_edge_sign < 0
            near = torch.where(backward, edges[1], edges[0])
            far = torch.where(backward, edges[0], edges[1])
            self.assertTrue(torch.equal(near, group.face_node_index[:, :-1]))
            self.assertTrue(torch.equal(far, group.face_node_index[:, 1:]))
        self.assertEqual(_leaked_pairs(data), set())
        # Control: the same check catches a test positive placed in the message graph.
        from src.data.link_pred_structures import PreparedLinkPredData

        fields = {name: getattr(data, name) for name in PreparedLinkPredData.__slots__}
        injected = data.test_pos_edge_index[:, :1]
        fields["edge_index"] = torch.cat([data.edge_index, injected], dim=1)
        self.assertEqual(_leaked_pairs(PreparedLinkPredData(**fields)), _pairs(injected))

    def test_each_decoder_trains_one_step_and_reaches_the_encoder(self) -> None:
        from src.builders import build_optimizer
        from src.models.h2snn_model import build_link_prediction_model
        from src.trainers.link_prediction_trainer import LinkPredictionTrainer

        for decoder_type in ("dot", "distmult", "concat_mlp"):
            with self.subTest(decoder=decoder_type):
                data = self._data(decoder_type)
                cfg = _link_config(self.root, decoder_type)
                torch.manual_seed(0)
                model = build_link_prediction_model(cfg, data)
                self.assertEqual(model.link_readout.decoder_type, decoder_type)
                trainer = LinkPredictionTrainer(model, build_optimizer(model, cfg), None, cfg,
                                                logger=None, checkpointer=None)
                metrics = trainer._train_epoch(data, epoch=0)
                self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
                # the encoder's own scalar: eta (first_order) or sum_k ||W_k|| (filter_bank, D12)
                self.assertIn("eta" if model.encoder == "first_order" else "filter_norm_sum",
                              metrics)
                # gamma exists only under path_weight: sigmoid_gamma (Part H: exp's b is path_gate)
                for prefix in ("node_encoder", "node_to_edge", "path_gate",
                               "theta_eta" if model.encoder == "first_order" else "filter_bank",
                               *(("theta_gamma",) if model.path_weight == "sigmoid_gamma" else ())):
                    grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
                    self.assertTrue(grads and all(g is not None for g in grads), prefix)
                    self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads), prefix)
                    self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0, prefix)


# ----------------------------------------------------------------------------- graph classification
def _typed_graph(seed: int, num_nodes: int = 7):
    """Undirected typed graph (both directions stored); edge type from the endpoint types."""
    from torch_geometric.data import Data

    generator = torch.Generator().manual_seed(seed)
    node_type = torch.randint(2, (num_nodes,), generator=generator)
    src, dst = [], []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if bool(torch.rand(1, generator=generator) < 0.45) or v == u + 1:
                src += [u, v]
                dst += [v, u]
    edge_index = torch.tensor([src, dst])
    edge_type = 2 * node_type[edge_index[0]] + node_type[edge_index[1]]
    return Data(x=torch.randn(num_nodes, 5, generator=generator), edge_index=edge_index,
                edge_type=edge_type, node_type=node_type,
                y=torch.randint(2, (1,), generator=generator))


def _permuted(graph, seed: int):
    from torch_geometric.data import Data

    perm = torch.randperm(graph.num_nodes, generator=torch.Generator().manual_seed(seed))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel())
    return Data(x=graph.x[perm], edge_index=inverse[graph.edge_index], edge_type=graph.edge_type,
                node_type=graph.node_type[perm], y=graph.y)


class GraphClassificationHeadTests(unittest.TestCase):
    def _setup(self, data: dict | None = None):
        from src.models.h2snn_model import ModelDims, build_graph_classification_model
        from src.trainers.graph_classification_trainer import GraphClassificationTrainer

        cfg = TrainConfig.from_dict({"device": "cpu", "checkpoint_metric": "auroc",
                                     "model": SMALL_MODEL,
                                     "data": {"dataset_name": "mutag", **(data or {})}})
        dims = ModelDims(in_channels=5, num_node_types=2, num_relations=4, num_classes=2,
                         face_lengths=(2,))
        torch.manual_seed(0)
        model = build_graph_classification_model(cfg, dims)
        from src.builders import build_optimizer

        trainer = GraphClassificationTrainer(model, build_optimizer(model, cfg), None, cfg,
                                             logger=None, checkpointer=None)
        return cfg, model, trainer

    def test_logits_are_invariant_to_node_permutation_within_a_graph(self) -> None:
        from torch_geometric.data import Batch

        # Walk sampling draws cells by node id, so permutation invariance of the logits holds
        # only in distribution under the defaults (tied + walk); this exact check fixes the
        # cells with the pre-D7 DFS builder (the operator and head are what is tested).
        _, model, trainer = self._setup({"fact_layer": "untied", "path_sampler": "dfs",
                                         "resample_paths": False})
        model.double().eval()
        graphs = [_typed_graph(seed) for seed in range(4)]
        for graph in graphs:
            graph.x = graph.x.double()
        permuted = [_permuted(graph, 10 + index) for index, graph in enumerate(graphs)]
        with torch.no_grad():
            logits = trainer.forward_batch(Batch.from_data_list(graphs))
            logits_permuted = trainer.forward_batch(Batch.from_data_list(permuted))
        self.assertGreater(float(logits.abs().max()), 0.0)
        self.assertLess(float((logits - logits_permuted).abs().max()), 1e-10)
        # Control: the same check sees a different graph.
        with torch.no_grad():
            other = trainer.forward_batch(Batch.from_data_list([_typed_graph(99)] + graphs[1:]))
        self.assertGreater(float((logits - other).abs().max()), 1e-6)

    def test_one_training_step_is_finite(self) -> None:
        from torch_geometric.loader import DataLoader

        _, model, trainer = self._setup()
        loader = DataLoader([_typed_graph(seed) for seed in range(6)], batch_size=3)
        metrics = trainer._run_epoch(loader, epoch=0, train=True)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertIn("eta" if model.encoder == "first_order" else "filter_norm_sum", metrics)
        self.assertIn("gamma" if model.path_weight == "sigmoid_gamma" else "path_log_w_max",
                      metrics)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        reached = [p.grad for n, p in model.named_parameters()
                   if n.startswith(("node_to_edge", "path_gate", "graph_readout"))]
        self.assertTrue(all(g is not None for g in reached))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
