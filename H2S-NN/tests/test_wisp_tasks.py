"""Part M, M4: WISP is the path source of link prediction and graph classification (spec v1.15
§3.2 (a), J1). Anchors: LP = every node of the two endpoint types on the message graph; GC = every
node of the graph. Evaluation draw fixed (seed of the run); training draws resampled per epoch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.configs.config import TrainConfig
from src.data.wisp_sampler import build_wisp_path_groups
from src.data.wisp_tasks import (build_graph_wisp_path_groups, graph_draw_seed,
                                 link_prediction_anchors)
from tests.test_task_heads import _write_tiny_lastfm

R, L_MAX = 4, 3


def _lp_config(root: Path, seed: int = 0) -> TrainConfig:
    return TrainConfig.from_dict({"device": "cpu", "seed": seed, "data": {
        "dataset_name": "lastfm", "root": str(root), "wisp_walks_per_anchor": R,
        "wisp_max_length": L_MAX}})


def _cells(groups) -> set[tuple]:
    """Each cell as (length, node sequence, fact sequence, weight)."""
    out = set()
    for group in groups:
        for nodes, facts, weight in zip(group.face_node_index.tolist(),
                                        group.face_edge_index.tolist(),
                                        group.path_weight.tolist()):
            out.add((len(facts), tuple(nodes), tuple(facts), round(weight, 6)))
    return out


class LinkPredictionAnchorTests(unittest.TestCase):
    def test_anchor_set_is_both_endpoint_types(self) -> None:
        slices = {"user": (0, 3), "artist": (3, 7), "tag": (7, 9)}
        self.assertEqual(link_prediction_anchors(slices, ("user", "rel_0", "artist")).tolist(),
                         list(range(0, 7)))
        self.assertEqual(link_prediction_anchors(slices, ("artist", "rel_9", "user")).tolist(),
                         [3, 4, 5, 6, 0, 1, 2])
        # same-type target (Freebase book-book, PubMed disease-disease): one range, no repeat
        self.assertEqual(link_prediction_anchors(slices, ("tag", "rel_2", "tag")).tolist(), [7, 8])


class LinkPredictionWispTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write_tiny_lastfm(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _data(self, seed: int = 0):
        from src.data.link_prediction_dataset import build_link_prediction_dataset

        return build_link_prediction_dataset(_lp_config(self.root, seed))

    def test_config_resolves_wisp_for_link_prediction(self) -> None:
        config = _lp_config(self.root)
        self.assertEqual(config.data.path_source, "wisp")
        self.assertEqual(config.data.wisp_walks_per_anchor, R)

    def test_r_must_be_set_to_draw_cells(self) -> None:
        """No calibrated r outside node classification: the config loads (data-only uses), the
        draw is refused."""
        from src.data.link_prediction_dataset import build_link_prediction_dataset

        config = TrainConfig.from_dict({"device": "cpu", "data": {
            "dataset_name": "lastfm", "root": str(self.root)}})
        self.assertEqual((config.data.path_source, config.data.wisp_walks_per_anchor), ("wisp", 0))
        with self.assertRaisesRegex(ValueError, "wisp_walks_per_anchor must be set"):
            build_link_prediction_dataset(config)

    def test_evaluation_cells_are_the_wisp_draw_from_both_endpoint_types(self) -> None:
        """The loader's cells equal a WISP draw (seed of the run) whose anchors are every user and
        every artist, computed here independently of the loader's helper (mutation: anchors
        restricted to one endpoint type)."""
        data = self._data(seed=5)
        users, artists = data.node_slices["user"], data.node_slices["artist"]
        anchors = torch.cat([torch.arange(*users), torch.arange(*artists)])
        expected = build_wisp_path_groups(
            data.edge_index, data.edge_type, data.node_type, data.fact_layer.self_inverse,
            anchors, R, L_MAX, seed=5)
        self.assertEqual(_cells(data.face_groups), _cells(expected))
        self.assertGreater(len(_cells(expected)), 0)
        heads_only = build_wisp_path_groups(
            data.edge_index, data.edge_type, data.node_type, data.fact_layer.self_inverse,
            torch.arange(*users), R, L_MAX, seed=5)
        self.assertNotEqual(_cells(data.face_groups), _cells(heads_only))
        lengths = {group.length for group in data.face_groups}
        self.assertTrue(lengths <= set(range(2, L_MAX + 1)))

    def test_no_held_out_fact_in_any_cell_of_any_draw(self) -> None:
        """Evaluation draw and three training draws: every hop of every cell is a message-graph
        fact, and no held-out pair (validation or test, either orientation) is a hop."""
        from src.data.link_prediction_dataset import build_link_path_groups

        data = self._data()
        held = torch.cat([data.val_pos_edge_index, data.test_pos_edge_index], dim=1)
        held_pairs = {tuple(p) for p in held.t().tolist()}
        held_pairs |= {(b, a) for a, b in held_pairs}
        self.assertGreater(len(held_pairs), 0)
        config = _lp_config(self.root)
        for epoch in range(4):
            groups = build_link_path_groups(config, data.fact_layer, data.edge_index,
                                            data.edge_type, data.node_type, data.node_slices,
                                            seed=config.seed + epoch)
            self.assertGreater(len(_cells(groups)), 0)
            for group in groups:
                self.assertLess(int(group.face_edge_index.max()), data.edge_index.size(1))
                nodes = group.face_node_index
                for hop in range(nodes.size(1) - 1):
                    hops = {tuple(p) for p in nodes[:, hop:hop + 2].tolist()}
                    self.assertFalse(hops & held_pairs)

    def test_evaluation_draw_fixed_training_draws_resampled(self) -> None:
        from src.trainers.link_prediction_trainer import LinkPredictionTrainer

        data = self._data()
        again = self._data()
        self.assertEqual(_cells(data.face_groups), _cells(again.face_groups))
        config = TrainConfig.from_dict({"device": "cpu", "data": {
            "dataset_name": "lastfm", "root": str(self.root), "wisp_walks_per_anchor": R,
            "wisp_max_length": L_MAX, "resample_paths": True}})
        trainer = LinkPredictionTrainer.__new__(LinkPredictionTrainer)
        trainer.cfg, trainer.device = config, torch.device("cpu")
        draws = [_cells(trainer.training_draw(data, data, epoch).face_groups) for epoch in (1, 2)]
        self.assertNotEqual(draws[0], draws[1])
        self.assertEqual(draws[0], _cells(trainer.training_draw(data, data, 1).face_groups))
        self.assertEqual(_cells(data.face_groups), _cells(again.face_groups))


def _graph(num_nodes: int, seed: int):
    from torch_geometric.data import Data

    generator = torch.Generator().manual_seed(seed)
    ring = torch.arange(num_nodes)
    src = torch.cat([ring, (ring + 2) % num_nodes])
    dst = torch.cat([(ring + 1) % num_nodes, ring])
    # one stored orientation per fact of a self-inverse relation, as `_tie_tu_facts` leaves them
    edge_index = torch.stack([src, dst])
    edge_type = torch.randint(0, 2, (edge_index.size(1),), generator=generator)
    return Data(x=torch.ones(num_nodes, 1), edge_index=edge_index, edge_type=edge_type,
                node_type=torch.randint(0, 3, (num_nodes,), generator=generator),
                y=torch.tensor([0]), graph_id=torch.tensor([seed]))


def _graph_cells(batch, seed: int) -> set[tuple]:
    groups = build_graph_wisp_path_groups(batch.edge_index, batch.edge_type, batch.node_type,
                                          batch.batch, batch.graph_id, (0, 1), R, L_MAX, seed)
    return _cells(groups)


class GraphClassificationWispTests(unittest.TestCase):
    def test_every_node_is_an_anchor(self) -> None:
        from torch_geometric.data import Batch

        from src.data.wisp_sampler import build_action_graph, sample_wisp_walks

        graph = _graph(7, 3)
        batch = Batch.from_data_list([graph])
        action = build_action_graph(graph.edge_index, graph.edge_type, 7, (0, 1))
        generator = torch.Generator().manual_seed(graph_draw_seed(11, 3))
        draw = sample_wisp_walks(action, graph.node_type, torch.arange(7), R, L_MAX, generator)
        self.assertEqual(sorted(set(draw.anchor.tolist()) | set(draw.discarded_anchor.tolist())),
                         list(range(7)))
        from src.data.wisp_sampler import merge_wisp_cells

        self.assertEqual(_graph_cells(batch, 11),
                         _cells(merge_wisp_cells(draw, graph.node_type, graph.edge_index, R)))

    def test_cells_of_a_graph_do_not_depend_on_other_graphs(self) -> None:
        from torch_geometric.data import Batch

        a, b = _graph(6, 1), _graph(9, 2)
        alone_a = _graph_cells(Batch.from_data_list([a]), 4)
        alone_b = _graph_cells(Batch.from_data_list([b]), 4)
        together = _graph_cells(Batch.from_data_list([a, b]), 4)
        shift_nodes, shift_facts = 6, a.edge_index.size(1)
        shifted_b = {(n, tuple(v + shift_nodes for v in nodes), tuple(f + shift_facts for f in facts),
                      w) for n, nodes, facts, w in alone_b}
        self.assertEqual(together, alone_a | shifted_b)
        self.assertGreater(len(alone_a), 0)
        self.assertGreater(len(alone_b), 0)

    def test_evaluation_draw_is_fixed_and_training_draws_change(self) -> None:
        from torch_geometric.data import Batch

        from src.trainers.graph_classification_trainer import GraphClassificationTrainer

        config = TrainConfig.from_dict({"device": "cpu", "seed": 7, "data": {
            "dataset_name": "mutag", "wisp_walks_per_anchor": R, "wisp_max_length": L_MAX}})
        trainer = GraphClassificationTrainer.__new__(GraphClassificationTrainer)
        trainer.cfg = config
        self.assertEqual({trainer.draw_seed(epoch, train=False) for epoch in range(5)}, {7})
        self.assertEqual([trainer.draw_seed(epoch, train=True) for epoch in range(3)], [7, 8, 9])
        batch = Batch.from_data_list([_graph(8, 5)])
        self.assertEqual(_graph_cells(batch, 7), _graph_cells(batch, 7))
        self.assertNotEqual(_graph_cells(batch, 8), _graph_cells(batch, 9))


if __name__ == "__main__":
    unittest.main()
