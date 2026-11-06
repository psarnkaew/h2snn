from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


class FakeHeteroData:
    def __init__(self) -> None:
        self.node_types = ("paper", "author", "conference")
        self.edge_types = (
            ("paper", "to", "author"),
            ("author", "to", "paper"),
        )
        self._stores = {
            "paper": SimpleNamespace(
                x=None,
                num_nodes=2,
                y=None,
                train_mask=None,
                val_mask=None,
                test_mask=None,
            ),
            "author": SimpleNamespace(
                x=None,
                num_nodes=1,
                y=None,
                train_mask=None,
                val_mask=None,
                test_mask=None,
            ),
            "conference": SimpleNamespace(
                x=None,
                num_nodes=1,
                y=None,
                train_mask=None,
                val_mask=None,
                test_mask=None,
            ),
            ("paper", "to", "author"): SimpleNamespace(edge_index=None),
            ("author", "to", "paper"): SimpleNamespace(edge_index=None),
        }

    def __getitem__(self, key: object) -> SimpleNamespace:
        return self._stores[key]


class DataPipelineTests(unittest.TestCase):
    def _require_torch(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"torch is required for this test: {exc}")
        return torch

    def test_deterministic_validation_split_is_reproducible(self) -> None:
        torch = self._require_torch()
        from src.data.preprocessing import deterministic_validation_split

        mask = torch.tensor([True, True, True, True, False])
        train_a, val_a = deterministic_validation_split(mask, val_ratio=0.25, seed=7)
        train_b, val_b = deterministic_validation_split(mask, val_ratio=0.25, seed=7)

        self.assertTrue(torch.equal(train_a, train_b))
        self.assertTrue(torch.equal(val_a, val_b))
        self.assertEqual(int(train_a.sum() + val_a.sum()), 4)

    def test_featureless_node_types_are_padded_to_common_dim(self) -> None:
        torch = self._require_torch()
        from src.data.hetero_flatten import stack_node_features

        data = FakeHeteroData()
        data["paper"].x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        data["author"].x = torch.tensor([[5.0, 6.0]])
        data["conference"].x = None

        x, node_type, node_type_names, node_slices, feature_dims = stack_node_features(data)

        self.assertEqual(tuple(node_type_names), ("paper", "author", "conference"))
        self.assertEqual(x.shape, (4, 2))
        self.assertEqual(feature_dims["conference"], 0)
        self.assertTrue(torch.equal(x[node_slices["conference"][0]], torch.zeros(2)))
        self.assertEqual(int(node_type.max().item()), 2)

    def test_encoder_sizes_each_type_to_its_own_width(self) -> None:
        torch = self._require_torch()
        from src.models.node_encoder import TypeSpecificNodeEncoder

        # Type 0 carries 2 real features, type 1 carries 5; both live in a matrix
        # padded to 5 columns. HGB gives each type a Linear at its own width.
        encoder = TypeSpecificNodeEncoder(
            input_dim=5,
            hidden_dim=4,
            num_node_types=2,
            dropout=0.0,
            type_input_dims=(2, 5),
        )

        self.assertEqual(encoder.encoders[0][0].in_features, 2)
        self.assertEqual(encoder.encoders[1][0].in_features, 5)

        # The padded columns must not reach the linear map: perturbing them cannot
        # change the encoding of a type-0 node.
        x = torch.zeros(2, 5)
        x[0, :2] = torch.tensor([1.0, 2.0])
        node_type = torch.tensor([0, 1])
        encoder.eval()
        baseline = encoder(x, node_type)
        x[0, 2:] = 9.0
        self.assertTrue(torch.allclose(baseline, encoder(x, node_type)))

    def test_encoder_gives_featureless_types_distinct_embeddings(self) -> None:
        torch = self._require_torch()
        from src.models.node_encoder import TypeSpecificNodeEncoder

        # Type 1 has no attributes. Zero-padded rows would collapse every node of that
        # type onto one vector; HGB's sp.eye rule (an embedding per node) must not.
        encoder = TypeSpecificNodeEncoder(
            input_dim=3,
            hidden_dim=4,
            num_node_types=2,
            dropout=0.0,
            type_input_dims=(3, 0),
            type_num_nodes=(1, 2),
            type_offsets=(0, 1),
        )
        encoder.eval()

        x = torch.zeros(3, 3)
        x[0] = torch.tensor([1.0, 2.0, 3.0])
        node_type = torch.tensor([0, 1, 1])
        hidden = encoder(x, node_type)

        self.assertEqual(hidden.shape, (3, 4))
        self.assertFalse(torch.allclose(hidden[1], hidden[2]))

    def test_encoder_normalises_featureless_and_featured_types_alike(self) -> None:
        torch = self._require_torch()
        from src.models.node_encoder import TypeSpecificNodeEncoder

        # Both branches end in a LayerNorm, so a drifting embedding cannot put one
        # type's stalks on a different scale from the rest before diffusion.
        encoder = TypeSpecificNodeEncoder(
            input_dim=3,
            hidden_dim=4,
            num_node_types=2,
            dropout=0.0,
            type_input_dims=(3, 0),
            type_num_nodes=(1, 2),
            type_offsets=(0, 1),
        )
        encoder.eval()
        with torch.no_grad():
            encoder.identity_embeddings[1][0].weight.mul_(1000.0)

        x = torch.zeros(3, 3)
        x[0] = torch.tensor([1.0, 2.0, 3.0])
        hidden = encoder(x, torch.tensor([0, 1, 1]))

        # A 1000x embedding blow-up must not survive into the stalk state.
        self.assertLess(float(hidden[1:].abs().max()), 10.0)

    def test_relation_diagnostics_are_a_single_global_vector(self) -> None:
        torch = self._require_torch()
        from src.data.preprocessing import compute_relation_diagnostics

        # One value per (relation, statistic) for the whole graph — never per node, which
        # is why it is a reporting artifact rather than a node feature.
        metrics = compute_relation_diagnostics(
            raw_x=torch.eye(3),
            node_type=torch.tensor([0, 0, 1]),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_type=torch.tensor([0, 1]),
            train_indices=torch.tensor([0]),
            train_labels=torch.tensor([0]),
            num_classes=2,
            num_relations=2,
            face_groups=(),
            num_label_prop_steps=1,
        )

        self.assertEqual(metrics.dim(), 1)
        self.assertEqual(metrics.numel(), 2 * 4)

    def test_flatten_and_face_enumeration_build_path_cells(self) -> None:
        torch = self._require_torch()
        from src.configs.config import MetaPathConfig
        from src.data.hetero_flatten import flatten_edges, stack_node_features
        from src.data.preprocessing import enumerate_metapath_faces

        data = FakeHeteroData()
        data["paper"].x = torch.tensor([[1.0], [2.0]])
        data["author"].x = torch.tensor([[3.0]])
        data["conference"].x = None
        data[("paper", "to", "author")].edge_index = torch.tensor([[0, 1], [0, 0]])
        data[("author", "to", "paper")].edge_index = torch.tensor([[0, 0], [1, 0]])

        _, _, _, node_slices, _ = stack_node_features(data)
        edge_index, edge_type, _, relation_tuples = flatten_edges(data, node_slices)
        face_groups = enumerate_metapath_faces(
            node_slices=node_slices,
            edge_index=edge_index,
            edge_type=edge_type,
            relation_tuples=relation_tuples,
            templates=(MetaPathConfig(name="PAP", node_types=("paper", "author", "paper")),),
            exclude_identity_metapaths=True,
            split_seed=11,
        )

        self.assertEqual(len(face_groups), 1)
        self.assertEqual(face_groups[0].template_name, "PAP")
        self.assertEqual(face_groups[0].face_edge_index.shape[1], 2)
        self.assertEqual(tuple(face_groups[0].face_node_index.shape), (face_groups[0].num_faces, 3))

    def test_graph_face_builder_uses_typed_two_hop_paths(self) -> None:
        torch = self._require_torch()
        from src.data.graph_faces import build_typed_path_face_groups

        edge_index = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 0],
            ],
            dtype=torch.long,
        )
        edge_type = torch.tensor([0, 1, 0], dtype=torch.long)
        node_type = torch.tensor([0, 1, 0], dtype=torch.long)
        batch = torch.zeros(3, dtype=torch.long)
        graph_id = torch.tensor([7], dtype=torch.long)

        face_groups = build_typed_path_face_groups(
            edge_index=edge_index,
            edge_type=edge_type,
            node_type=node_type,
            batch=batch,
            graph_id=graph_id,
            max_paths_per_anchor=10,
            exclude_identity_paths=True,
            seed=13,
        )

        self.assertEqual(sum(group.num_faces for group in face_groups), 3)
        for group in face_groups:
            self.assertEqual(group.length, 2)
            self.assertTrue(group.template_name.startswith("N"))
            self.assertIn("_R", group.template_name)
            self.assertTrue(torch.equal(group.face_edge_sign, torch.ones_like(group.face_edge_sign)))
            expected_relations = edge_type[group.face_edge_index]
            self.assertTrue(torch.equal(group.relation_index, expected_relations))

    def test_graph_face_builder_excludes_identity_backtracking(self) -> None:
        torch = self._require_torch()
        from src.data.graph_faces import build_typed_path_face_groups

        edge_index = torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.long,
        )
        edge_type = torch.zeros(2, dtype=torch.long)
        node_type = torch.zeros(2, dtype=torch.long)

        face_groups = build_typed_path_face_groups(
            edge_index=edge_index,
            edge_type=edge_type,
            node_type=node_type,
            batch=torch.zeros(2, dtype=torch.long),
            graph_id=torch.tensor([3], dtype=torch.long),
            max_paths_per_anchor=50,
            exclude_identity_paths=True,
            seed=21,
        )

        self.assertEqual(face_groups, tuple())

    def test_hgb_link_split_uses_81_9_10_ratios(self) -> None:
        torch = self._require_torch()
        from src.data.link_prediction_dataset import _split_edge_index_hgb

        edge_index = torch.stack([torch.arange(100), torch.arange(100) + 100])
        train_edges, val_edges, test_edges = _split_edge_index_hgb(
            edge_index=edge_index,
            val_ratio=0.09,
            test_ratio=0.10,
            seed=17,
        )

        self.assertEqual(train_edges.size(1), 81)
        self.assertEqual(val_edges.size(1), 9)
        self.assertEqual(test_edges.size(1), 10)
        train_set = set(map(tuple, train_edges.t().tolist()))
        val_set = set(map(tuple, val_edges.t().tolist()))
        test_set = set(map(tuple, test_edges.t().tolist()))
        self.assertFalse(train_set & val_set)
        self.assertFalse(train_set & test_set)
        self.assertFalse(val_set & test_set)

    def test_graph_holdout_split_is_stratified_and_reproducible(self) -> None:
        from src.data.graph_classification_dataset import generate_stratified_holdout_split

        labels = [0, 1] * 50
        split_a = generate_stratified_holdout_split(
            labels=labels,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=11,
        )
        split_b = generate_stratified_holdout_split(
            labels=labels,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=11,
        )
        split_c = generate_stratified_holdout_split(
            labels=labels,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=12,
        )

        self.assertEqual(split_a, split_b)
        self.assertNotEqual(split_a, split_c)
        self.assertEqual(len(split_a["train"]), 80)
        self.assertEqual(len(split_a["validation"]), 10)
        self.assertEqual(len(split_a["test"]), 10)
        train_set = set(split_a["train"])
        val_set = set(split_a["validation"])
        test_set = set(split_a["test"])
        self.assertFalse(train_set & val_set)
        self.assertFalse(train_set & test_set)
        self.assertFalse(val_set & test_set)
        for split_indices in split_a.values():
            self.assertEqual({labels[index] for index in split_indices}, {0, 1})

    def test_repeated_kfold_splits_cover_each_repetition_once(self) -> None:
        from src.data.graph_classification_dataset import generate_repeated_kfold_splits

        splits_a = generate_repeated_kfold_splits(
            num_graphs=100,
            num_folds=10,
            num_repetitions=2,
            split_seed=17,
        )
        splits_b = generate_repeated_kfold_splits(
            num_graphs=100,
            num_folds=10,
            num_repetitions=2,
            split_seed=17,
        )

        self.assertEqual(splits_a, splits_b)
        self.assertEqual(len(splits_a), 2)
        self.assertNotEqual(
            [fold["test"] for fold in splits_a[0]],
            [fold["test"] for fold in splits_a[1]],
        )
        for repetition in splits_a:
            all_test_indices: list[int] = []
            for fold in repetition:
                model_selection = fold["model_selection"][0]
                train_set = set(model_selection["train"])
                val_set = set(model_selection["validation"])
                test_set = set(fold["test"])
                self.assertEqual(len(train_set), 81)
                self.assertEqual(len(val_set), 9)
                self.assertEqual(len(test_set), 10)
                self.assertFalse(train_set & val_set)
                self.assertFalse(train_set & test_set)
                self.assertFalse(val_set & test_set)
                self.assertEqual(train_set | val_set | test_set, set(range(100)))
                all_test_indices.extend(test_set)
            self.assertEqual(sorted(all_test_indices), list(range(100)))

    def test_repeated_kfold_config_validates_indices(self) -> None:
        from src.configs.config import TrainConfig

        cfg = TrainConfig.from_dict(
            {
                "data": {
                    "dataset_name": "mutag",
                    "split_protocol": "repeated_kfold",
                    "num_folds": 10,
                    "fold_index": 9,
                    "num_repetitions": 10,
                    "repetition_index": 9,
                    "split_seed": 23,
                }
            }
        )
        self.assertEqual(cfg.data.split_seed, 23)
        with self.assertRaisesRegex(ValueError, "repetition_index"):
            TrainConfig.from_dict(
                {
                    "data": {
                        "dataset_name": "mutag",
                        "split_protocol": "repeated_kfold",
                        "num_repetitions": 10,
                        "repetition_index": 10,
                    }
                }
            )

    def test_tu_type_tensors_use_node_and_edge_labels(self) -> None:
        torch = self._require_torch()
        from torch_geometric.data import Data

        from src.data.graph_classification_dataset import _load_tu_type_tensors

        tmp_parent = Path(__file__).resolve().parent.parent
        with TemporaryDirectory(dir=tmp_parent) as tmpdir:
            raw_dir = Path(tmpdir) / "tud" / "FAKE" / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "FAKE_node_labels.txt").write_text("5\n7\n5\n", encoding="utf-8")
            (raw_dir / "FAKE_graph_indicator.txt").write_text("1\n1\n2\n", encoding="utf-8")
            (raw_dir / "FAKE_A.txt").write_text("1, 2\n2, 1\n3, 3\n", encoding="utf-8")
            (raw_dir / "FAKE_edge_labels.txt").write_text("9\n8\n9\n", encoding="utf-8")

            data_list = [
                Data(edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=2),
                Data(edge_index=torch.tensor([[0], [0]]), num_nodes=1),
            ]
            node_types, edge_types, num_node_types, num_relations = _load_tu_type_tensors(
                root=Path(tmpdir),
                tud_name="FAKE",
                data_list=data_list,
            )

        self.assertEqual(num_node_types, 2)
        self.assertEqual(num_relations, 2)
        self.assertTrue(torch.equal(node_types[0], torch.tensor([0, 1])))
        self.assertTrue(torch.equal(node_types[1], torch.tensor([0])))
        self.assertTrue(torch.equal(edge_types[0], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(edge_types[1], torch.tensor([1])))

    def test_tu_type_tensors_fallback_to_single_edge_type(self) -> None:
        torch = self._require_torch()
        from torch_geometric.data import Data

        from src.data.graph_classification_dataset import _load_tu_type_tensors

        tmp_parent = Path(__file__).resolve().parent.parent
        with TemporaryDirectory(dir=tmp_parent) as tmpdir:
            raw_dir = Path(tmpdir) / "tud" / "FAKE" / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "FAKE_node_labels.txt").write_text("1\n2\n3\n", encoding="utf-8")
            (raw_dir / "FAKE_graph_indicator.txt").write_text("1\n1\n1\n", encoding="utf-8")
            data_list = [Data(edge_index=torch.tensor([[0, 1], [1, 2]]), num_nodes=3)]
            node_types, edge_types, num_node_types, num_relations = _load_tu_type_tensors(
                root=Path(tmpdir),
                tud_name="FAKE",
                data_list=data_list,
            )

        self.assertEqual(num_node_types, 3)
        self.assertEqual(num_relations, 1)
        self.assertTrue(torch.equal(node_types[0], torch.tensor([0, 1, 2])))
        self.assertTrue(torch.equal(edge_types[0], torch.tensor([0, 0])))

    def test_graph_classification_metrics_include_binary_and_multiclass_auroc(self) -> None:
        torch = self._require_torch()
        from src.utils.misc import compute_graph_classification_metrics

        binary = compute_graph_classification_metrics(
            logits=torch.tensor([[0.0, 3.0], [3.0, 0.0], [0.0, 2.0], [2.0, 0.0]]),
            labels=torch.tensor([1, 0, 1, 0]),
        )
        multiclass = compute_graph_classification_metrics(
            logits=torch.tensor(
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 4.0],
                    [3.0, 1.0, 0.0],
                    [0.0, 3.0, 1.0],
                    [1.0, 0.0, 3.0],
                ]
            ),
            labels=torch.tensor([0, 1, 2, 0, 1, 2]),
        )

        self.assertAlmostEqual(binary["auroc"], 1.0)
        self.assertAlmostEqual(multiclass["auroc"], 1.0)

    def test_hgb_two_hop_negatives_are_grouped_by_source(self) -> None:
        torch = self._require_torch()
        from src.data.link_prediction_dataset import _sample_hgb_two_hop_negatives

        reference_pos = torch.tensor([[0, 1], [10, 11]])
        known_pos = reference_pos.clone()
        adjacency_edges = torch.tensor(
            [
                [0, 0, 1],
                [1, 10, 11],
            ]
        )

        negatives = _sample_hgb_two_hop_negatives(
            reference_pos_edge_index=reference_pos,
            adjacency_edge_index=adjacency_edges,
            known_positive_edge_index=known_pos,
            dst_range=(10, 12),
            seed=5,
            neg_sampling_ratio=1,
            allow_random_fallback=False,
        )

        self.assertTrue(torch.equal(negatives, torch.tensor([[0, 1], [11, 10]])))

    def test_link_prediction_metrics_include_hgb_mrr(self) -> None:
        torch = self._require_torch()
        from src.utils.misc import compute_link_prediction_metrics

        pos_scores = torch.tensor([0.2, 0.9])
        neg_scores = torch.tensor([0.8, 0.1, 0.1])
        pos_edge_index = torch.tensor([[0, 1], [10, 10]])
        neg_edge_index = torch.tensor([[0, 0, 1], [11, 12, 11]])

        metrics = compute_link_prediction_metrics(
            pos_scores,
            neg_scores,
            pos_edge_index=pos_edge_index,
            neg_edge_index=neg_edge_index,
        )

        self.assertIn("auroc", metrics)
        self.assertIn("aupr", metrics)
        self.assertAlmostEqual(metrics["mrr"], 0.75)
        # No relation ids supplied -> the key set is unchanged, so H2SNN's own trainer
        # and every existing record keep exactly the metrics they had.
        self.assertEqual(set(metrics), {"auroc", "aupr", "mrr"})

    def test_link_prediction_metrics_macro_average_matches_hgb(self) -> None:
        """HGB scores each test relation separately and averages the results.

        Every one of its link-prediction runners loops over
        ``dl.links_test['data'].keys()`` and divides by the relation count -- see
        HGB/LP/benchmark/methods/HGT/link_predict.py:285-288 and the ``np.mean`` over
        ``test_score`` in .../GNN/homoGNN.py:155-159. Pooling the relations into one
        AUROC is a different number whenever they differ in size or difficulty, which is
        the case on Amazon and YouTube.
        """

        torch = self._require_torch()
        from src.utils.misc import compute_link_prediction_metrics

        # Relation 0: perfectly separable, few edges. Relation 1: chance, many edges.
        pos_scores = torch.tensor([5.0, 5.0] + [0.0] * 8)
        neg_scores = torch.tensor([-5.0, -5.0] + [0.0] * 8)
        relation = torch.tensor([0, 0] + [1] * 8)

        metrics = compute_link_prediction_metrics(
            pos_scores,
            neg_scores,
            pos_relation=relation,
            neg_relation=relation,
        )

        self.assertEqual(metrics["num_eval_relations"], 2.0)
        # Relation 0 scores 1.0, relation 1 scores 0.5 (all-tied) -> macro 0.75.
        self.assertAlmostEqual(metrics["auroc_macro"], 0.75)
        # Pooled over all 20 edges the two perfect positives barely move the number,
        # so the macro average is materially higher.
        self.assertGreater(metrics["auroc_macro"], metrics["auroc"])

    def test_link_prediction_macro_equals_pooled_on_one_relation(self) -> None:
        """LastFM and PubMed target a single relation, so selection must not shift."""

        torch = self._require_torch()
        from src.utils.misc import compute_link_prediction_metrics

        pos_scores = torch.tensor([0.9, 0.4, 0.7])
        neg_scores = torch.tensor([0.1, 0.6])
        relation_pos = torch.zeros(3, dtype=torch.long)
        relation_neg = torch.zeros(2, dtype=torch.long)

        metrics = compute_link_prediction_metrics(
            pos_scores,
            neg_scores,
            pos_relation=relation_pos,
            neg_relation=relation_neg,
        )

        self.assertEqual(metrics["auroc_macro"], metrics["auroc"])
        self.assertEqual(metrics["aupr_macro"], metrics["aupr"])

    def test_local_hgb_train_valid_split_keeps_each_head_in_train(self) -> None:
        torch = self._require_torch()
        from src.data.link_prediction_dataset import _split_edge_index_hgb_by_head

        edge_index = torch.tensor(
            [
                [0, 0, 0, 1, 1, 2],
                [1, 2, 3, 2, 3, 4],
            ]
        )

        train_edges, valid_edges = _split_edge_index_hgb_by_head(
            edge_index=edge_index,
            val_ratio=1.0,
            seed=3,
        )

        train_pairs = set(map(tuple, train_edges.t().tolist()))
        valid_pairs = set(map(tuple, valid_edges.t().tolist()))
        self.assertTrue({(0, 1), (1, 2), (2, 4)} <= train_pairs)
        self.assertTrue(valid_pairs)
        self.assertFalse(train_pairs & valid_pairs)

    def test_hgb_multilabel_target_labels_are_preserved(self) -> None:
        torch = self._require_torch()
        from src.data.dataset import _prepare_target_labels

        data = FakeHeteroData()
        labels = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        data["paper"].y = labels
        data["paper"].train_mask = torch.tensor([True, False])
        data["paper"].val_mask = None
        data["paper"].test_mask = torch.tensor([False, True])

        (
            global_y,
            train_mask,
            val_mask,
            test_mask,
            target_indices,
            target_y,
            target_train_mask,
            num_classes,
            is_multilabel,
        ) = _prepare_target_labels(
            data=data,
            node_slices={"paper": (0, 2), "author": (2, 3), "conference": (3, 4)},
            target_node_type="paper",
            val_ratio=0.2,
            split_seed=3,
        )

        self.assertTrue(is_multilabel)
        self.assertEqual(num_classes, 3)
        self.assertEqual(tuple(global_y.shape), (4, 3))
        self.assertTrue(torch.equal(target_y, labels))
        self.assertTrue(torch.equal(target_indices, torch.tensor([0, 1])))
        self.assertEqual(int(train_mask.sum() + val_mask.sum()), 1)
        self.assertEqual(int(test_mask.sum()), 1)
        self.assertEqual(int(target_train_mask.sum()), 1)

    def test_multilabel_label_propagation_accepts_multi_hot_labels(self) -> None:
        torch = self._require_torch()
        from src.data.preprocessing import label_propagation

        propagated = label_propagation(
            edge_index=torch.tensor([[0], [1]]),
            train_indices=torch.tensor([0]),
            train_labels=torch.tensor([[1.0, 0.0, 1.0]]),
            num_nodes=2,
            num_classes=3,
            num_steps=1,
        )

        self.assertTrue(torch.allclose(propagated[0], torch.tensor([0.5, 0.0, 0.5])))
        self.assertTrue(torch.allclose(propagated[1], torch.tensor([0.5, 0.0, 0.5])))

    def test_multilabel_classification_metrics_follow_hgb_threshold(self) -> None:
        torch = self._require_torch()
        from src.utils.misc import compute_multilabel_classification_metrics

        logits = torch.tensor([[2.0, -1.0, 0.5], [-0.2, 0.4, -0.1]])
        labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

        metrics = compute_multilabel_classification_metrics(logits, labels)

        self.assertEqual(metrics["acc"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["micro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
