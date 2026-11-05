from __future__ import annotations

import unittest


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment specific
        raise unittest.SkipTest(f"torch is required for this test: {exc}") from exc
    return torch


def _random_orthogonal(torch, d: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(d, d, generator=generator)
    q, _ = torch.linalg.qr(matrix)
    return q


def _random_orthogonal_stack(torch, count: int, d: int, seed: int):
    """One independent O(d) matrix per node/edge — a *local* gauge, which is what a sheaf
    actually has. A single shared rotation would only test global O(d) invariance."""
    return torch.stack([_random_orthogonal(torch, d, seed + i) for i in range(count)])


def _make_synthetic_batch(torch, d: int, seed: int = 0):
    """Two tiny disjoint graphs (3 nodes/2 edges, 2 nodes/1 edge) mimicking a PyG
    Batch — the exact shape GraphPooling.forward consumes.

    `F_src`/`F_tgt` are deliberately *not* orthogonal: the endpoint-adjoint transport is
    equivariant for arbitrary restriction maps, so random maps are the stronger test.
    """
    generator = torch.Generator().manual_seed(seed)
    num_nodes = 5
    num_edges = 3
    x_star = torch.randn(num_nodes, d, generator=generator)
    h0 = torch.randn(num_nodes, d, generator=generator)
    s_star = torch.randn(num_edges, d, generator=generator)
    delta0_x_star = torch.randn(num_edges, d, generator=generator)
    delta0_h0 = torch.randn(num_edges, d, generator=generator)
    edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]])
    F_src = torch.randn(num_edges, d, d, generator=generator)
    F_tgt = torch.randn(num_edges, d, d, generator=generator)
    batch = torch.tensor([0, 0, 0, 1, 1])
    num_graphs = 2
    return (
        x_star,
        h0,
        s_star,
        delta0_x_star,
        delta0_h0,
        edge_index,
        F_src,
        F_tgt,
        batch,
        num_graphs,
    )


def _apply_local_gauge(torch, inputs, node_gauge, edge_gauge):
    """Transform every backbone quantity by an independent per-stalk orthogonal basis change.

    Node-stalk quantities move by `Q_v`, edge-stalk quantities by `R_e`, and each restriction
    map by `F -> R_e F Q_v^T` with `Q_v` taken at that map's own endpoint.
    """
    (
        x_star,
        h0,
        s_star,
        delta0_x_star,
        delta0_h0,
        edge_index,
        F_src,
        F_tgt,
        batch,
        num_graphs,
    ) = inputs

    def rotate(gauge, values):
        return torch.bmm(gauge, values.unsqueeze(-1)).squeeze(-1)

    src_gauge = node_gauge[edge_index[0]]
    tgt_gauge = node_gauge[edge_index[1]]
    return (
        rotate(node_gauge, x_star),
        rotate(node_gauge, h0),
        rotate(edge_gauge, s_star),
        rotate(edge_gauge, delta0_x_star),
        rotate(edge_gauge, delta0_h0),
        edge_index,
        torch.bmm(torch.bmm(edge_gauge, F_src), src_gauge.transpose(-2, -1)),
        torch.bmm(torch.bmm(edge_gauge, F_tgt), tgt_gauge.transpose(-2, -1)),
        batch,
        num_graphs,
    )


class EdgeStalkTransportTests(unittest.TestCase):
    """`pool_edge_stalks_to_nodes` is the correctness fix: incident edge states are pulled
    back into the node's own stalk before they are averaged."""

    def test_hand_computed_endpoint_aggregation(self) -> None:
        torch = _require_torch()
        from src.models.pooling_utils import pool_edge_stalks_to_nodes

        s_star = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        edge_index = torch.tensor([[0, 1], [1, 2]])
        F_src = torch.stack([torch.eye(2), 2.0 * torch.eye(2)])
        F_tgt = torch.stack([3.0 * torch.eye(2), 4.0 * torch.eye(2)])

        pooled = pool_edge_stalks_to_nodes(
            s_star=s_star, edge_index=edge_index, F_src=F_src, F_tgt=F_tgt, num_nodes=3
        )

        # node 0: only edge 0 as source          -> 1 * [1,2]
        # node 1: edge 0 as target, edge 1 as src -> (3*[1,2] + 2*[3,4]) / 2
        # node 2: only edge 1 as target          -> 4 * [3,4]
        expected = torch.tensor(
            [[1.0, 2.0], [(3.0 + 6.0) / 2, (6.0 + 8.0) / 2], [12.0, 16.0]]
        )
        torch.testing.assert_close(pooled, expected)

    def test_isolated_node_gets_zero_summary(self) -> None:
        torch = _require_torch()
        from src.models.pooling_utils import pool_edge_stalks_to_nodes

        pooled = pool_edge_stalks_to_nodes(
            s_star=torch.tensor([[1.0, 2.0]]),
            edge_index=torch.tensor([[0], [1]]),
            F_src=torch.eye(2).unsqueeze(0),
            F_tgt=torch.eye(2).unsqueeze(0),
            num_nodes=3,
        )
        torch.testing.assert_close(pooled[2], torch.zeros(2))

    def test_self_loop_counts_as_two_incidences(self) -> None:
        torch = _require_torch()
        from src.models.pooling_utils import pool_edge_stalks_to_nodes

        pooled = pool_edge_stalks_to_nodes(
            s_star=torch.tensor([[2.0, 4.0]]),
            edge_index=torch.tensor([[0], [0]]),
            F_src=torch.eye(2).unsqueeze(0),
            F_tgt=3.0 * torch.eye(2).unsqueeze(0),
            num_nodes=1,
        )
        # (1*[2,4] + 3*[2,4]) / 2 incidences
        torch.testing.assert_close(pooled[0], torch.tensor([4.0, 8.0]))

    def test_misaligned_inputs_raise_value_error(self) -> None:
        torch = _require_torch()
        from src.models.pooling_utils import pool_edge_stalks_to_nodes

        s_star = torch.randn(2, 3)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        F_src = torch.randn(2, 3, 3)
        F_tgt = torch.randn(2, 3, 3)
        base = dict(
            s_star=s_star, edge_index=edge_index, F_src=F_src, F_tgt=F_tgt, num_nodes=3
        )

        cases = {
            "edge_index rank": {"edge_index": torch.zeros(2, dtype=torch.long)},
            "map rank": {"F_src": torch.randn(2, 3)},
            "map shape mismatch": {"F_tgt": torch.randn(2, 4, 4)},
            "extended-domain misalignment": {"s_star": torch.randn(5, 3)},
            "stalk dimension mismatch": {
                "F_src": torch.randn(2, 4, 4),
                "F_tgt": torch.randn(2, 4, 4),
            },
        }
        for name, override in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    pool_edge_stalks_to_nodes(**{**base, **override})

    def test_transport_is_equivariant_under_independent_local_gauges(self) -> None:
        torch = _require_torch()
        from src.models.pooling_utils import pool_edge_stalks_to_nodes

        d = 4
        inputs = _make_synthetic_batch(torch, d)
        _, _, s_star, _, _, edge_index, F_src, F_tgt, _, _ = inputs
        num_nodes = inputs[0].size(0)

        node_gauge = _random_orthogonal_stack(torch, num_nodes, d, seed=11)
        edge_gauge = _random_orthogonal_stack(torch, edge_index.size(1), d, seed=71)
        gauged = _apply_local_gauge(torch, inputs, node_gauge, edge_gauge)

        original = pool_edge_stalks_to_nodes(
            s_star=s_star,
            edge_index=edge_index,
            F_src=F_src,
            F_tgt=F_tgt,
            num_nodes=num_nodes,
        )
        transformed = pool_edge_stalks_to_nodes(
            s_star=gauged[2],
            edge_index=edge_index,
            F_src=gauged[6],
            F_tgt=gauged[7],
            num_nodes=num_nodes,
        )
        expected = torch.bmm(node_gauge, original.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(transformed, expected, atol=1e-5, rtol=1e-4)

    def test_raw_coordinate_aggregation_is_not_equivariant(self) -> None:
        """Negative control: proves the equivariance assertion above can actually fail.

        The pre-fix raw aggregation is reproduced inline so the regression remains an
        explicit negative control after the legacy helper's removal.
        """
        torch = _require_torch()

        d = 4
        inputs = _make_synthetic_batch(torch, d)
        x_star, _, s_star, _, _, edge_index, _, _, _, _ = inputs
        num_nodes = x_star.size(0)

        def raw_aggregate(edge_values):
            pooled = torch.zeros((num_nodes, d))
            counts = torch.zeros((num_nodes, 1))
            ones = torch.ones((edge_values.size(0), 1))
            pooled.index_add_(0, edge_index[0], edge_values)
            pooled.index_add_(0, edge_index[1], edge_values)
            counts.index_add_(0, edge_index[0], ones)
            counts.index_add_(0, edge_index[1], ones)
            return pooled / counts.clamp_min(1.0)

        node_gauge = _random_orthogonal_stack(torch, num_nodes, d, seed=11)
        edge_gauge = _random_orthogonal_stack(torch, edge_index.size(1), d, seed=71)

        original = raw_aggregate(s_star)
        transformed = raw_aggregate(
            torch.bmm(edge_gauge, s_star.unsqueeze(-1)).squeeze(-1)
        )
        expected = torch.bmm(node_gauge, original.unsqueeze(-1)).squeeze(-1)
        self.assertFalse(torch.allclose(transformed, expected, atol=1e-4))


class DecoderTransportWiringTests(unittest.TestCase):
    def test_node_readout_uses_endpoint_specific_transport(self) -> None:
        torch = _require_torch()
        from src.models.readout import NodeReadout

        readout = NodeReadout(d=2, num_classes=2, hidden_dim=4, dropout=0.0)
        readout.mlp = torch.nn.Identity()

        output = readout(
            x_star=torch.zeros(2, 2),
            h0=torch.zeros(2, 2),
            s_star=torch.tensor([[1.0, 2.0]]),
            edge_index=torch.tensor([[0], [1]]),
            F_src=2.0 * torch.eye(2).unsqueeze(0),
            F_tgt=3.0 * torch.eye(2).unsqueeze(0),
            target_indices=torch.tensor([0, 1]),
        )

        expected = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 2.0, 4.0], [0.0, 0.0, 0.0, 0.0, 3.0, 6.0]]
        )
        torch.testing.assert_close(output, expected)

    def test_link_readout_uses_endpoint_specific_transport(self) -> None:
        torch = _require_torch()
        from src.models.link_readout import LinkPredictionReadout

        readout = LinkPredictionReadout(
            d=2, decoder_type="dot", hidden_dim=4, dropout=0.0
        )
        projection = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            projection.weight.zero_()
            projection.weight[:, 2:] = torch.eye(2)
        readout.endpoint_projection = projection

        score = readout(
            x_star=torch.zeros(2, 2),
            s_star=torch.tensor([[1.0, 2.0]]),
            edge_index=torch.tensor([[0], [1]]),
            F_src=2.0 * torch.eye(2).unsqueeze(0),
            F_tgt=3.0 * torch.eye(2).unsqueeze(0),
            edge_label_index=torch.tensor([[0], [1]]),
        )

        torch.testing.assert_close(score, torch.tensor([30.0]))

    def test_relation_bilinear_decoder_uses_src_matrix_dst_order(self) -> None:
        torch = _require_torch()
        from src.models.link_readout import LinkPredictionReadout

        readout = LinkPredictionReadout(
            d=2,
            decoder_type="distmult",
            hidden_dim=4,
            dropout=0.0,
            num_relations=1,
        )
        projection = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            projection.weight.zero_()
            projection.weight[:, :2] = torch.eye(2)
            readout.relation_matrices[0].copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        readout.endpoint_projection = projection

        score = readout(
            x_star=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            s_star=torch.zeros(1, 2),
            edge_index=torch.tensor([[0], [1]]),
            F_src=torch.eye(2).unsqueeze(0),
            F_tgt=torch.eye(2).unsqueeze(0),
            edge_label_index=torch.tensor([[0], [1]]),
            edge_relation=torch.zeros(1, dtype=torch.long),
        )

        torch.testing.assert_close(score, torch.tensor([61.0]))


class GraphPoolingShapeTests(unittest.TestCase):
    def test_sheafcov_feature_width_is_configurable_without_changing_gram_input(self) -> None:
        torch = _require_torch()
        from src.models.graph_pooling import SheafCovPool, build_graph_pooling

        for feat_dim in (4, 8, 16):
            with self.subTest(feat_dim=feat_dim):
                pooling = build_graph_pooling(
                    pooling_type="sheafcov",
                    d=6,
                    hidden_dim=8,
                    num_classes=3,
                    dropout=0.0,
                    sheafcov_feat_dim=feat_dim,
                )
                self.assertIsInstance(pooling, SheafCovPool)
                self.assertEqual(pooling.feat_dim, feat_dim)
                # The learned projection still consumes vec(Z^T Z) for exactly
                # three semantic views: a fixed raw 3x3 Gram descriptor.
                self.assertEqual(pooling.node_feat[0].in_features, 9)
                self.assertEqual(pooling.edge_feat[0].in_features, 9)
                self.assertEqual(pooling.node_feat[-1].out_features, feat_dim)
                self.assertEqual(pooling.edge_feat[-1].out_features, feat_dim)

                with torch.no_grad():
                    logits = pooling(*_make_synthetic_batch(torch, d=6))
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertTrue(torch.isfinite(logits).all())

        default_pool = build_graph_pooling(
            pooling_type="sheafcov",
            d=6,
            hidden_dim=8,
            num_classes=3,
            dropout=0.0,
        )
        explicit_default_pool = build_graph_pooling(
            pooling_type="sheafcov",
            d=6,
            hidden_dim=8,
            num_classes=3,
            dropout=0.0,
            sheafcov_feat_dim=8,
        )
        self.assertEqual(
            {name: tuple(value.shape) for name, value in default_pool.state_dict().items()},
            {
                name: tuple(value.shape)
                for name, value in explicit_default_pool.state_dict().items()
            },
        )

    def test_sheafcov_feature_width_config_defaults_and_validation(self) -> None:
        from src.configs.config import ModelConfig

        self.assertEqual(ModelConfig().sheafcov_feat_dim, 8)
        self.assertEqual(
            ModelConfig.from_dict({"sheafcov_feat_dim": 16}).sheafcov_feat_dim,
            16,
        )
        for invalid in (0, -1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "sheafcov_feat_dim must be positive"
            ):
                ModelConfig(sheafcov_feat_dim=invalid)

    def test_h2snn_passes_configured_feature_width_to_sheafcov(self) -> None:
        _require_torch()
        from src.configs.config import TrainConfig
        from src.models.graph_pooling import SheafCovPool
        from src.models.h2snn_model import H2SNN, ModelDims

        cfg = TrainConfig.from_dict(
            {
                "data": {"dataset_name": "mutag", "pooling_type": "sheafcov"},
                "model": {
                    "d": 4,
                    "d_r": 2,
                    "d_h": 4,
                    "K": 1,
                    "edge_mlp_hidden_dim": 8,
                    "edge_init_hidden_dim": 8,
                    "face_mlp_hidden_dim": 8,
                    "classifier_hidden_dim": 8,
                    "sheafcov_feat_dim": 16,
                },
            }
        )
        model = H2SNN(
            cfg,
            ModelDims(
                in_channels=3,
                num_node_types=1,
                num_relations=1,
                num_classes=2,
            ),
        )

        self.assertIsInstance(model.graph_pooling, SheafCovPool)
        self.assertEqual(model.graph_pooling.feat_dim, 16)

    def test_all_variants_produce_expected_shape_with_no_nans(self) -> None:
        torch = _require_torch()
        from src.models.graph_pooling import build_graph_pooling

        d = 6
        num_classes = 3
        inputs = _make_synthetic_batch(torch, d)
        num_graphs = inputs[-1]

        for pooling_type in ("meanpool", "grampool", "s2pool", "sheafpool", "sheafcov"):
            with self.subTest(pooling_type):
                pooling = build_graph_pooling(
                    pooling_type=pooling_type,
                    d=d,
                    hidden_dim=8,
                    num_classes=num_classes,
                    dropout=0.0,
                )
                pooling.eval()
                with torch.no_grad():
                    logits = pooling(*inputs)
                self.assertEqual(tuple(logits.shape), (num_graphs, num_classes))
                self.assertTrue(
                    torch.isfinite(logits).all(), f"{pooling_type} produced non-finite logits"
                )

    def test_graph_schema_enumeration_stops_at_guard_plus_one(self) -> None:
        torch = _require_torch()
        from src.data.graph_faces import build_typed_path_face_groups

        edge_index = torch.tensor(
            [[0, 0, 1, 2], [1, 2, 3, 3]],
            dtype=torch.long,
        )
        edge_type = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        node_type = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "guard \\+ 1"):
            build_typed_path_face_groups(
                edge_index=edge_index,
                edge_type=edge_type,
                node_type=node_type,
                min_path_length=2,
                max_path_length=2,
                max_paths_per_anchor=None,
                max_template_identities=1,
            )


class GraphPoolingInvarianceTests(unittest.TestCase):
    """The point of the Gram-descriptor readouts: logits must not depend on the arbitrary
    basis chosen in each stalk. Every node and every edge gets its *own* O(d) matrix, which
    is the gauge freedom a sheaf actually has."""

    def _assert_local_gauge_invariant(self, torch, pooling_type: str) -> None:
        from src.models.graph_pooling import build_graph_pooling

        d = 6
        inputs = _make_synthetic_batch(torch, d)
        num_nodes = inputs[0].size(0)
        num_edges = inputs[5].size(1)

        pooling = build_graph_pooling(
            pooling_type=pooling_type, d=d, hidden_dim=8, num_classes=3, dropout=0.0
        )
        pooling.eval()

        node_gauge = _random_orthogonal_stack(torch, num_nodes, d, seed=101)
        edge_gauge = _random_orthogonal_stack(torch, num_edges, d, seed=307)

        with torch.no_grad():
            original = pooling(*inputs)
            transformed = pooling(*_apply_local_gauge(torch, inputs, node_gauge, edge_gauge))

        torch.testing.assert_close(original, transformed, atol=1e-4, rtol=1e-3)

    def test_grampool_invariant_under_independent_local_gauges(self) -> None:
        torch = _require_torch()
        self._assert_local_gauge_invariant(torch, "grampool")

    def test_sheafcovpool_invariant_under_independent_local_gauges(self) -> None:
        torch = _require_torch()
        self._assert_local_gauge_invariant(torch, "sheafcov")

    def test_sheafcovpool_invariant_to_node_and_edge_reindexing(self) -> None:
        torch = _require_torch()
        from src.models.graph_pooling import build_graph_pooling

        d = 6
        (
            x_star,
            h0,
            s_star,
            delta0_x_star,
            delta0_h0,
            edge_index,
            F_src,
            F_tgt,
            batch,
            num_graphs,
        ) = _make_synthetic_batch(torch, d)

        pooling = build_graph_pooling(
            pooling_type="sheafcov", d=d, hidden_dim=8, num_classes=3, dropout=0.0
        )
        pooling.eval()

        node_perm = torch.tensor([2, 0, 1, 4, 3])  # keeps each node inside its own graph
        edge_perm = torch.tensor([2, 0, 1])
        relabel = torch.empty_like(node_perm)
        relabel[node_perm] = torch.arange(node_perm.numel())

        with torch.no_grad():
            original = pooling(
                x_star, h0, s_star, delta0_x_star, delta0_h0, edge_index,
                F_src, F_tgt, batch, num_graphs,
            )
            permuted = pooling(
                x_star[node_perm],
                h0[node_perm],
                s_star[edge_perm],
                delta0_x_star[edge_perm],
                delta0_h0[edge_perm],
                relabel[edge_index][:, edge_perm],
                F_src[edge_perm],
                F_tgt[edge_perm],
                batch[node_perm],
                num_graphs,
            )

        torch.testing.assert_close(original, permuted, atol=1e-4, rtol=1e-3)

    def test_every_extended_domain_edge_participates(self) -> None:
        """Closure 1-cells sit after the real edges in the same extended domain; the edge
        lane and the incident-edge node view must both read them."""
        torch = _require_torch()
        from src.models.graph_pooling import build_graph_pooling

        d = 6
        inputs = list(_make_synthetic_batch(torch, d))
        pooling = build_graph_pooling(
            pooling_type="sheafcov", d=d, hidden_dim=8, num_classes=3, dropout=0.0
        )
        pooling.eval()

        with torch.no_grad():
            original = pooling(*inputs)
            for edge in range(inputs[5].size(1)):
                muted = list(inputs)
                muted[2] = inputs[2].clone()
                muted[2][edge] = 0.0
                with self.subTest(edge=edge):
                    self.assertFalse(
                        torch.allclose(original, pooling(*muted), atol=1e-6),
                        f"edge {edge} had no effect on the graph logits",
                    )


if __name__ == "__main__":
    unittest.main()
