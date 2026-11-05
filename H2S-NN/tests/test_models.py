from __future__ import annotations

import unittest


def _make_prepared_data(schema: str, with_discovery: bool = False):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment specific
        raise unittest.SkipTest(f"torch is required for this test: {exc}") from exc

    from src.data.structures import FaceGroup, PreparedH2SData
    from src.data.schema_structures import DiscoveryMetadata

    if schema == "acm":
        node_type_names = ("paper", "author")
        relation_tuples = (("paper", "to", "author"), ("author", "to", "paper"))
        target_node_type = "paper"
        template_name = "PAP"
        node_slices = {"paper": (0, 2), "author": (2, 4)}
        node_type = torch.tensor([0, 0, 1, 1])
    elif schema == "dblp":
        node_type_names = ("author", "paper", "term")
        relation_tuples = (
            ("author", "to", "paper"),
            ("paper", "to", "term"),
            ("term", "to", "paper"),
            ("paper", "to", "author"),
        )
        target_node_type = "author"
        template_name = "APTPA"
        node_slices = {"author": (0, 2), "paper": (2, 3), "term": (3, 4)}
        node_type = torch.tensor([0, 0, 1, 2])
    elif schema == "imdb":
        node_type_names = ("movie", "actor")
        relation_tuples = (("movie", "to", "actor"), ("actor", "to", "movie"))
        target_node_type = "movie"
        template_name = "MAM"
        node_slices = {"movie": (0, 2), "actor": (2, 4)}
        node_type = torch.tensor([0, 0, 1, 1])
    else:  # pragma: no cover - safety
        raise ValueError(f"Unknown schema '{schema}'.")

    x = torch.randn(4, 6)
    y = torch.tensor([0, 1, -1, -1])
    edge_index = torch.tensor([[0, 2], [2, 1]])
    edge_type = torch.tensor([0, min(1, len(relation_tuples) - 1)])
    target_indices = torch.tensor([0, 1])
    face_group = FaceGroup(
        template_name=template_name,
        node_type_sequence=(target_node_type, node_type_names[1], target_node_type),
        face_node_index=torch.tensor([[0, 2, 1]]),
        face_edge_index=torch.tensor([[0, 1]]),
        relation_index=edge_type.view(1, -1),
        face_edge_sign=torch.ones(1, 2),
        closure_edge_index=torch.tensor([[1, 0]]),
    )
    discovery = None
    if with_discovery:
        discovery = DiscoveryMetadata(
            num_schemas=1,
            schema_ids=torch.tensor([0]),
            schema_incidence_mass=torch.tensor([3.0]),
            coverage_schema_index=torch.tensor([0]),
            coverage_unit_index=torch.tensor([0]),
            num_coverage_units=2,
        )

    return PreparedH2SData(
        x=x,
        y=y,
        node_type=node_type,
        node_type_names=node_type_names,
        node_slices=node_slices,
        feature_dims={name: 6 for name in node_type_names},
        edge_index=edge_index,
        edge_type=edge_type,
        relation_names=tuple("__".join(value) for value in relation_tuples[: edge_type.numel()]),
        relation_tuples=tuple(relation_tuples[: edge_type.numel()]),
        train_mask=torch.tensor([True, False, False, False]),
        val_mask=torch.tensor([False, True, False, False]),
        test_mask=torch.tensor([False, True, False, False]),
        target_node_type=target_node_type,
        target_indices=target_indices,
        target_y=torch.tensor([0, 1]),
        target_train_mask=torch.tensor([True, False]),
        target_val_mask=torch.tensor([False, True]),
        target_test_mask=torch.tensor([False, True]),
        num_classes=2,
        is_multilabel=False,
        face_groups=(face_group,),
        closure_edge_index=torch.tensor([[1], [0]]),
        metrics=torch.zeros(edge_type.numel() * 4),
        discovery=discovery,
    )


class CoboundaryOperatorTests(unittest.TestCase):
    def _require_torch(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"torch is required for this test: {exc}")
        return torch

    def test_hodge_diffusion_layer_dropout_is_train_only(self) -> None:
        torch = self._require_torch()
        from src.models.diffusion import HodgeDiffusion

        x0 = torch.ones(512, 4)
        s0 = torch.ones(512, 4)

        def zero_l0(x):
            return torch.zeros_like(x)

        def zero_l1(s):
            return torch.zeros_like(s)

        diffusion = HodgeDiffusion(
            num_steps=1,
            eta_0=0.1,
            eta_1=0.1,
            learnable_eta=False,
            layer_dropout=0.5,
        )

        torch.manual_seed(7)
        diffusion.train()
        train_x, train_s, _ = diffusion(x0, s0, zero_l0, zero_l1)
        self.assertGreater(int((train_x == 0).sum().item()), 0)
        self.assertGreater(int((train_s == 0).sum().item()), 0)

        diffusion.eval()
        eval_x, eval_s, _ = diffusion(x0, s0, zero_l0, zero_l1)
        self.assertTrue(torch.equal(eval_x, x0))
        self.assertTrue(torch.equal(eval_s, s0))

    def test_delta_operators_match_dense_reference(self) -> None:
        torch = self._require_torch()
        from src.models.coboundary_ops import CoboundaryOps
        from src.models.sheaf_maps import FaceMapCache

        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
        F_src = torch.eye(2).repeat(2, 1, 1)
        F_tgt = torch.eye(2).repeat(2, 1, 1)
        src_idx = torch.tensor([0, 2])
        tgt_idx = torch.tensor([2, 1])
        delta0 = CoboundaryOps.delta0(x, F_src, F_tgt, src_idx, tgt_idx)
        expected_delta0 = torch.stack([x[2] - x[0], x[1] - x[2]], dim=0)
        self.assertTrue(torch.allclose(delta0, expected_delta0))

        delta0_t = CoboundaryOps.delta0_T(delta0, F_src, F_tgt, src_idx, tgt_idx, num_nodes=3)
        # Edges: 0→2 (src_idx[0]=0, tgt_idx[0]=2) and 2→1 (src_idx[1]=2, tgt_idx[1]=1).
        # Flat node ordering: [node0_d0, node0_d1, node1_d0, node1_d1, node2_d0, node2_d1].
        # Row (e0,d0): tgt pos=4 gets +1, src pos=0 gets -1  →  [-1, 0, 0, 0, 1, 0]
        # Row (e0,d1): tgt pos=5 gets +1, src pos=1 gets -1  →  [0, -1, 0, 0, 0, 1]
        # Row (e1,d0): tgt pos=2 gets +1, src pos=4 gets -1  →  [0, 0, 1, 0, -1, 0]
        # Row (e1,d1): tgt pos=3 gets +1, src pos=5 gets -1  →  [0, 0, 0, 1, 0, -1]
        dense_delta0 = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
            ],
        )
        dense_delta0_t = dense_delta0.t() @ expected_delta0.reshape(-1)
        self.assertTrue(torch.allclose(delta0_t.reshape(-1), dense_delta0_t))

        face_cache = FaceMapCache(
            template_name="toy",
            face_edge_index=torch.tensor([[0, 1]]),
            relation_index=torch.tensor([[0, 1]]),
            face_edge_sign=torch.ones(1, 2),
            maps=torch.eye(2).view(1, 1, 2, 2).repeat(1, 2, 1, 1),
            group_id=0,
            instance_start=0,
            instance_end=1,
        )
        map_builder = lambda cache: torch.eye(2).repeat(cache.face_edge_index.size(0), cache.face_edge_index.size(1), 1, 1)
        delta1 = CoboundaryOps.delta1(delta0, (face_cache,), map_builder)
        self.assertTrue(torch.allclose(delta1.concatenated, delta0.sum(dim=0, keepdim=True)))
        delta1_t = CoboundaryOps.delta1_T(delta1, (face_cache,), num_edges=2, d=2)
        self.assertEqual(delta1_t.shape, delta0.shape)

    def test_delta1_respects_face_edge_sign(self) -> None:
        torch = self._require_torch()
        from src.models.coboundary_ops import CoboundaryOps
        from src.models.sheaf_maps import FaceMapCache

        s = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        face_cache = FaceMapCache(
            template_name="toy",
            face_edge_index=torch.tensor([[0, 1]]),
            relation_index=torch.tensor([[0, 1]]),
            face_edge_sign=torch.tensor([[1.0, -1.0]]),
            maps=torch.eye(2).view(1, 1, 2, 2).repeat(1, 2, 1, 1),
            group_id=0,
            instance_start=0,
            instance_end=1,
        )
        map_builder = lambda cache: torch.eye(2).repeat(
            cache.face_edge_index.size(0), cache.face_edge_index.size(1), 1, 1
        )

        delta1 = CoboundaryOps.delta1(s, (face_cache,), map_builder)
        # eta = [+1, -1] with identity maps: (delta1 s)_f = s[0] - s[1].
        expected_face_value = (s[0] - s[1]).unsqueeze(0)
        self.assertTrue(torch.allclose(delta1.concatenated, expected_face_value))

        delta1_t = CoboundaryOps.delta1_T(delta1, (face_cache,), num_edges=2, d=2)
        # Adjoint: edge 0 (eta=+1) gets +q_f, edge 1 (eta=-1) gets -q_f.
        expected_delta1_t = torch.stack([expected_face_value[0], -expected_face_value[0]], dim=0)
        self.assertTrue(torch.allclose(delta1_t, expected_delta1_t))

    def test_delta1_face_weights_align_across_chunk_boundary(self) -> None:
        # Regression test: SequenceEdgeToFaceSheaf splits one face group into
        # multiple FaceMapCache chunks when num_faces > face_chunk_size, but
        # H2SNN._build_face_weights returns one weight vector per *group* (full,
        # unchunked length). _delta1_grouped must slice each chunk's own
        # [instance_start:instance_end] range out of its group's weight vector —
        # not assume one weight tensor per chunk — or it crashes/misapplies
        # weights as soon as a schema's instance count crosses the chunk size.
        torch = self._require_torch()
        from src.models.coboundary_ops import CoboundaryOps
        from src.models.sheaf_maps import FaceMapCache

        s = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        map_builder = lambda cache: torch.eye(2).repeat(
            cache.face_edge_index.size(0), cache.face_edge_index.size(1), 1, 1
        )

        # One group of 3 faces, chunked 2 + 1 (as if face_chunk_size=2).
        chunk0 = FaceMapCache(
            template_name="toy",
            face_edge_index=torch.tensor([[0, 1], [0, 1]]),
            relation_index=torch.tensor([[0, 1], [0, 1]]),
            face_edge_sign=torch.ones(2, 2),
            maps=torch.eye(2).view(1, 1, 2, 2).repeat(2, 2, 1, 1),
            group_id=0,
            instance_start=0,
            instance_end=2,
        )
        chunk1 = FaceMapCache(
            template_name="toy",
            face_edge_index=torch.tensor([[0, 1]]),
            relation_index=torch.tensor([[0, 1]]),
            face_edge_sign=torch.ones(1, 2),
            maps=torch.eye(2).view(1, 1, 2, 2).repeat(1, 2, 1, 1),
            group_id=0,
            instance_start=2,
            instance_end=3,
        )
        # Full-group weight vector (length 3), as H2SNN._build_face_weights produces.
        face_weights = (torch.tensor([1.0, 4.0, 9.0]),)

        raw_delta1 = CoboundaryOps.delta1(s, (chunk0, chunk1), map_builder)
        delta1 = CoboundaryOps.scale_delta1_rows(
            raw_delta1,
            (chunk0, chunk1),
            face_weights,
            power=0.5,
        )
        unweighted_face_value = (s[0] + s[1]).unsqueeze(0)  # identity maps, eta=+1 both edges
        expected = unweighted_face_value * torch.sqrt(face_weights[0]).unsqueeze(-1)
        self.assertTrue(torch.allclose(delta1.concatenated, expected))

    def test_upper_operator_is_linear_in_face_weight_and_psd(self) -> None:
        torch = self._require_torch()
        from src.models.coboundary_ops import CoboundaryOps
        from src.models.sheaf_maps import FaceMapCache

        cache = FaceMapCache(
            template_name="toy",
            face_edge_index=torch.tensor([[0, 1]]),
            relation_index=torch.tensor([[0, 1]]),
            face_edge_sign=torch.tensor([[1.0, -1.0]]),
            maps=torch.ones(1, 2, 1, 1),
            group_id=0,
            instance_start=0,
            instance_end=1,
        )
        s = torch.tensor([[2.0], [-1.0]])
        raw = CoboundaryOps.delta1(s, (cache,), lambda value: value.maps)

        def upper(weight: float):
            scaled = CoboundaryOps.scale_delta1_rows(
                raw,
                (cache,),
                (torch.tensor([weight]),),
                power=1.0,
            )
            return CoboundaryOps.delta1_T(
                scaled,
                (cache,),
                num_edges=2,
                d=1,
            )

        full = upper(1.0)
        self.assertTrue(torch.equal(upper(0.0), torch.zeros_like(full)))
        self.assertTrue(torch.allclose(upper(0.25), 0.25 * full))
        self.assertGreaterEqual(float((s * full).sum()), 0.0)

    def test_face_instance_weight_uses_ordered_intermediate_state(self) -> None:
        torch = self._require_torch()
        from src.data.structures import FaceGroup
        from src.models.sheaf_maps import SequenceEdgeToFaceSheaf

        module = SequenceEdgeToFaceSheaf(
            d=1,
            d_r=1,
            d_h=2,
            hidden_dim=2,
            face_encoder="concat",
            face_lengths=(2,),
        )
        with torch.no_grad():
            projection = module.input_proj["2"]
            projection.weight.zero_()
            projection.bias.zero_()
            projection.weight[0, 1] = 1.0

        h = torch.tensor([[0.0], [1.0], [3.0], [0.0]], requires_grad=True)
        group = FaceGroup(
            template_name="toy",
            node_type_sequence=("a", "b", "a"),
            face_node_index=torch.tensor([[0, 1, 3], [0, 2, 3]]),
            face_edge_index=torch.tensor([[0, 1], [2, 3]]),
            relation_index=torch.tensor([[0, 0], [0, 0]]),
            face_edge_sign=torch.ones(2, 2),
            closure_edge_index=torch.tensor([[3, 0], [3, 0]]),
        )
        identity_maps = torch.ones(4, 1, 1)
        output = module(
            h=h,
            face_groups=(group,),
            relation_embeddings=torch.zeros(2, 1),
            F_src=identity_maps,
            F_tgt=identity_maps,
            F_src_scale=torch.ones(4),
            num_real_edges=4,
            instance_weight_scorer=lambda _group_id, latent: 2.0
            * torch.sigmoid(latent[:, 0]),
        )
        assert output.instance_weights is not None
        weights = output.instance_weights[0]
        self.assertNotEqual(float(weights[0]), float(weights[1]))
        weights.sum().backward()
        self.assertNotEqual(float(h.grad[1]), 0.0)
        self.assertNotEqual(float(h.grad[2]), 0.0)

    def test_closure_edges_enter_the_extended_edge_cochain(self) -> None:
        """Closure 1-cells extend the edge list, so L0 picks them up through delta0 itself.

        Replaces an older test of `CoboundaryOps.closure_L0`, a separate correction applied
        *outside* delta0 with frozen `scale*I` maps. That operator is gone: it made delta0's
        codomain disagree with delta1's domain, which is exactly what stopped
        `delta1 . delta0 = 0` from being achievable.
        """
        try:
            import torch
            from src.configs.config import TrainConfig
            from src.models.h2snn_model import build_model
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        data = _make_prepared_data("acm")
        cfg = TrainConfig.from_dict(
            {
                "device": "cpu",
                "model": {"d": 4, "d_r": 2, "d_h": 8, "K": 2, "classifier_hidden_dim": 8},
                "checkpoint_metric": "acc",
            }
        )
        model = build_model(cfg, data)
        encoded = model.encode(
            x=data.x,
            node_type=data.node_type,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            face_groups=data.face_groups,
            closure_edge_index=data.closure_edge_index,
            num_nodes=data.num_nodes,
            num_edges=data.num_edges,
        )

        num_faces = sum(group.num_faces for group in data.face_groups)
        self.assertEqual(encoded.num_real_edges, data.num_edges)
        self.assertEqual(encoded.edge_index.size(1), data.num_edges + num_faces)
        # The edge cochain and both restriction-map stacks span the extended edge set.
        self.assertEqual(encoded.s_star.size(0), data.num_edges + num_faces)
        self.assertEqual(encoded.F_src.size(0), data.num_edges + num_faces)
        self.assertEqual(encoded.F_tgt.size(0), data.num_edges + num_faces)
        # The appended columns really are the closure edges the data carried.
        self.assertTrue(
            torch.equal(
                encoded.edge_index[:, data.num_edges :],
                data.closure_edge_index[:, :num_faces],
            )
        )

    def test_closure_edges_add_no_learned_parameters(self) -> None:
        """Closure restriction maps are *defined* by the path transport, not learned.

        `NodeToEdgeSheaf` therefore still holds exactly one MLP pair per data relation --
        no per-relation MLP is added for the synthetic closure edge. It does get a relation
        *embedding* row, since `EdgeInitializer` needs one to build its edge cochain.
        """
        try:
            import torch
            from src.configs.config import TrainConfig
            from src.models.h2snn_model import build_model
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        data = _make_prepared_data("acm")
        cfg = TrainConfig.from_dict(
            {
                "device": "cpu",
                "model": {"d": 4, "d_r": 2, "d_h": 8, "K": 2, "classifier_hidden_dim": 8},
                "checkpoint_metric": "acc",
            }
        )
        model = build_model(cfg, data)
        self.assertEqual(len(model.node_to_edge.src_mlps), len(data.relation_names))
        self.assertEqual(len(model.node_to_edge.tgt_mlps), len(data.relation_names))
        self.assertEqual(model.closure_relation_id, len(data.relation_names))
        self.assertEqual(
            model.relation_embedding.num_embeddings, len(data.relation_names) + 1
        )

        # Gradient still reaches the node signal through the closure edges' derived maps.
        output = model(data)
        output.s_star.square().sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                for parameter in model.parameters()
            )
        )

    def test_face_cache_covers_boundary_including_closure(self) -> None:
        """Face maps span L+1 boundary edges and are precomputed once per chunk."""
        try:
            import torch
            from src.data.structures import face_lengths_of
            from src.models.sheaf_maps import SequenceEdgeToFaceSheaf
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        data = _make_prepared_data("acm")
        d = 4
        module = SequenceEdgeToFaceSheaf(
            d=d,
            d_r=2,
            d_h=8,
            hidden_dim=16,
            face_lengths=face_lengths_of(data.face_groups),
        )
        h = torch.randn(data.num_nodes, d)
        # +1 embedding row for the synthetic closure relation.
        relation_embeddings = torch.randn(len(data.relation_names) + 1, 2)
        identity = torch.eye(d).unsqueeze(0).expand(data.num_edges, -1, -1).contiguous()

        output = module(
            h=h,
            face_groups=data.face_groups,
            relation_embeddings=relation_embeddings,
            F_src=identity,
            F_tgt=identity,
            F_src_scale=torch.ones(data.num_edges),
            num_real_edges=data.num_edges,
        )

        self.assertEqual(len(output.caches), 1)
        cache = output.caches[0]
        path_length = data.face_groups[0].length
        self.assertEqual(cache.maps.shape[1], path_length + 1)
        self.assertEqual(cache.maps.shape[-2:], (d, d))
        self.assertEqual(cache.face_edge_index.shape[1], path_length + 1)
        self.assertEqual(cache.face_edge_sign.shape[1], path_length + 1)
        # Closure ids are handed out immediately after the real edges.
        self.assertEqual(int(cache.face_edge_index[0, -1]), data.num_edges)
        self.assertEqual(output.closure_F_src.shape, (1, d, d))
        # With identity node->edge maps the path holonomy is the identity.
        torch.testing.assert_close(output.closure_F_src[0], torch.eye(d))

    def _run_face_encoder(self, face_encoder: str, face_map_param: str, d: int = 4):
        import torch

        from src.data.structures import face_lengths_of
        from src.models.sheaf_maps import SequenceEdgeToFaceSheaf

        data = _make_prepared_data("acm")
        module = SequenceEdgeToFaceSheaf(
            d=d,
            d_r=2,
            d_h=8,
            hidden_dim=16,
            face_encoder=face_encoder,
            face_map_param=face_map_param,
            face_lengths=face_lengths_of(data.face_groups),
        )
        identity = torch.eye(d).unsqueeze(0).expand(data.num_edges, -1, -1).contiguous()
        output = module(
            h=torch.randn(data.num_nodes, d),
            face_groups=data.face_groups,
            relation_embeddings=torch.randn(len(data.relation_names) + 1, 2),
            F_src=identity,
            F_tgt=identity,
            F_src_scale=torch.ones(data.num_edges),
            num_real_edges=data.num_edges,
        )
        return module, data, output

    def test_both_face_encoders_produce_the_same_map_shapes(self) -> None:
        """`concat` is an encoder swap: it must be interchangeable with `lstm` downstream."""
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        shapes = {}
        for face_encoder in ("concat", "lstm"):
            with self.subTest(face_encoder=face_encoder):
                _, data, output = self._run_face_encoder(face_encoder, "full")
                cache = output.caches[0]
                path_length = data.face_groups[0].length
                self.assertEqual(cache.maps.shape[1], path_length + 1)
                self.assertEqual(cache.maps.shape[-2:], (4, 4))
                shapes[face_encoder] = tuple(cache.maps.shape)
        self.assertEqual(shapes["concat"], shapes["lstm"])

    def test_triangular_face_maps_are_upper_triangular_and_span_psd(self) -> None:
        """Gauge fixing: A_f is upper triangular, and A_f^T A_f -- the only observable -- is PSD."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        d = 4
        for face_encoder in ("concat", "lstm"):
            with self.subTest(face_encoder=face_encoder):
                module, _, output = self._run_face_encoder(face_encoder, "triangular", d=d)
                self.assertEqual(module.map_output_dim, d * (d + 1) // 2)
                # The closure edge's face map is A_f itself, so it exposes the raw matrix.
                A_f = output.caches[0].maps[:, -1]
                self.assertTrue(torch.allclose(A_f, torch.triu(A_f)))
                gram = A_f.transpose(-2, -1) @ A_f
                torch.testing.assert_close(gram, gram.transpose(-2, -1))
                eigenvalues = torch.linalg.eigvalsh(gram)
                self.assertGreaterEqual(float(eigenvalues.min()), -1e-5)

    def test_concat_encoder_rejects_an_unbuilt_path_length(self) -> None:
        """A length with no projection must fail loudly, not index into a missing module."""
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        with self.assertRaisesRegex(KeyError, "no input projection for path length"):
            self._run_face_encoder_without_lengths()

    def _run_face_encoder_without_lengths(self):
        import torch

        from src.models.sheaf_maps import SequenceEdgeToFaceSheaf

        data = _make_prepared_data("acm")
        d = 4
        module = SequenceEdgeToFaceSheaf(d=d, d_r=2, d_h=8, hidden_dim=16, face_lengths=())
        identity = torch.eye(d).unsqueeze(0).expand(data.num_edges, -1, -1).contiguous()
        return module(
            h=torch.randn(data.num_nodes, d),
            face_groups=data.face_groups,
            relation_embeddings=torch.randn(len(data.relation_names) + 1, 2),
            F_src=identity,
            F_tgt=identity,
            F_src_scale=torch.ones(data.num_edges),
            num_real_edges=data.num_edges,
        )

    def test_relation_sequence_guard_rejects_a_mixed_schema_group(self) -> None:
        """The concat encoder reads row 0 as the group's schema; a mixed group must be caught."""
        try:
            import torch

            from src.data.face_geometry import validate_relation_sequence_constant
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        constant = torch.tensor([[0, 1], [0, 1], [0, 1]])
        validate_relation_sequence_constant(constant, "ok")  # must not raise
        validate_relation_sequence_constant(torch.empty(0, 2, dtype=torch.long), "empty")

        mixed = torch.tensor([[0, 1], [1, 1]])
        with self.assertRaisesRegex(ValueError, "relation ids vary across instances"):
            validate_relation_sequence_constant(mixed, "mixed")


class H2SNNForwardTests(unittest.TestCase):
    def _assert_forward_backward(self, schema: str) -> None:
        try:
            import torch
            from src.configs.config import TrainConfig
            from src.models.h2snn_model import build_model
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        data = _make_prepared_data(schema)
        cfg = TrainConfig.from_dict(
            {
                "device": "cpu",
                "model": {"d": 4, "d_r": 2, "d_h": 8, "K": 2, "classifier_hidden_dim": 8},
                "checkpoint_metric": "acc",
            }
        )
        model = build_model(cfg, data)
        output = model(data)
        loss = output.logits.sum() + output.s_star.sum() + output.x_star.sum()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(any(gradient is not None for gradient in gradients))

    def test_forward_backward_smoke_acm(self) -> None:
        self._assert_forward_backward("acm")

    def test_forward_backward_smoke_dblp(self) -> None:
        self._assert_forward_backward("dblp")

    def test_forward_backward_smoke_imdb(self) -> None:
        self._assert_forward_backward("imdb")

    def test_chain_condition_holds_for_every_face_encoder_and_parameterisation(self) -> None:
        """delta1.delta0 = 0 for all four encoder x parameterisation combinations.

        Proposition 1's telescoping never inspects A_f, so the residual must stay at
        roundoff no matter how A_f is produced. If this fails, the transport wiring broke --
        not the encoder.
        """
        try:
            import torch
            from src.configs.config import TrainConfig
            from src.models.h2snn_model import build_model
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        data = _make_prepared_data("acm")
        for face_encoder in ("concat", "lstm"):
            for face_map_param in ("full", "triangular"):
                with self.subTest(encoder=face_encoder, param=face_map_param):
                    cfg = TrainConfig.from_dict(
                        {
                            "device": "cpu",
                            "model": {
                                "d": 4,
                                "d_r": 2,
                                "d_h": 8,
                                "K": 2,
                                "classifier_hidden_dim": 8,
                                "face_encoder": face_encoder,
                                "face_map_param": face_map_param,
                            },
                            "checkpoint_metric": "acc",
                        }
                    )
                    model = build_model(cfg, data)
                    with torch.no_grad():
                        output = model(data)
                    residual = (
                        output.delta1_delta0_h0.norm() / output.delta0_h0.norm().clamp_min(1e-12)
                    )
                    self.assertLess(float(residual), 1e-5)


class AMPSSInstanceWeightTests(unittest.TestCase):
    def test_node_task_loss_reaches_instance_weight_network(self) -> None:
        import torch
        from src.configs.config import TrainConfig
        from src.models.h2snn_model import build_model

        data = _make_prepared_data("acm", with_discovery=True)
        cfg = TrainConfig.from_dict(
            {
                "device": "cpu",
                "model": {
                    "d": 4,
                    "d_r": 2,
                    "d_h": 8,
                    "K": 1,
                    "classifier_hidden_dim": 8,
                },
                "discovery": {
                    "enabled": True,
                    "mode": "auto",
                    "max_total_instances": 10,
                    "max_total_incidences": 30,
                },
            }
        )
        model = build_model(cfg, data)
        model.train()
        output = model(data, epoch=-1)
        output.logits.square().sum().backward()
        self.assertIsNotNone(model.instance_weight_network)
        weight_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.instance_weight_network.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(weight_grad, 0.0)

    def test_operator_ramp_removes_the_post_warmup_cliff(self) -> None:
        try:
            import torch
            from src.configs.config import TrainConfig
            from src.models.h2snn_model import build_model
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        data = _make_prepared_data("acm", with_discovery=True)
        cfg = TrainConfig.from_dict(
            {
                "device": "cpu",
                "model": {"d": 4, "d_r": 2, "d_h": 8, "K": 1},
                "discovery": {
                    "enabled": True,
                    "warmup_epochs": 2,
                    "operator_anneal_epochs": 4,
                },
            }
        )
        model = build_model(cfg, data)
        model.eval()
        h0 = torch.randn(data.num_nodes, cfg.model.d)
        learned_weights = tuple(
            h0.new_full((group.num_faces,), 0.5)
            for group in data.face_groups
        )

        before = model._build_face_weights(
            data.face_groups,
            data.discovery,
            h0,
            epoch=1,
            learned_instance_weights=None,
        )
        at_boundary = model._build_face_weights(
            data.face_groups,
            data.discovery,
            h0,
            epoch=2,
            learned_instance_weights=learned_weights,
        )
        midpoint = model._build_face_weights(
            data.face_groups,
            data.discovery,
            h0,
            epoch=4,
            learned_instance_weights=learned_weights,
        )
        finished = model._build_face_weights(
            data.face_groups,
            data.discovery,
            h0,
            epoch=6,
            learned_instance_weights=learned_weights,
        )
        self.assertTrue(
            torch.allclose(before[0][0], torch.ones_like(before[0][0]))
        )
        self.assertTrue(
            torch.allclose(at_boundary[0][0], torch.ones_like(at_boundary[0][0]))
        )
        self.assertTrue(
            torch.allclose(midpoint[0][0], torch.full_like(midpoint[0][0], 0.75))
        )
        self.assertTrue(
            torch.allclose(finished[0][0], torch.full_like(finished[0][0], 0.5))
        )

    def test_task_gradient_reaches_weights_only_after_ramp_starts(self) -> None:
        import torch
        from src.configs.config import TrainConfig
        from src.models.h2snn_model import build_model

        data = _make_prepared_data("acm", with_discovery=True)
        cfg = TrainConfig.from_dict(
            {
                "device": "cpu",
                "model": {"d": 4, "d_r": 2, "d_h": 8, "K": 1},
                "discovery": {
                    "enabled": True,
                    "mode": "auto",
                    "warmup_epochs": 1,
                    "operator_anneal_epochs": 2,
                    "max_total_instances": 10,
                    "max_total_incidences": 30,
                },
            }
        )
        model = build_model(cfg, data)
        model.train()

        at_boundary = model(data, epoch=1)
        self.assertEqual(at_boundary.operator_ramp, 0.0)
        at_boundary.logits.square().sum().backward()
        assert model.instance_weight_network is not None
        boundary_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.instance_weight_network.parameters()
            if parameter.grad is not None
        )
        self.assertEqual(boundary_grad, 0.0)

        model.zero_grad(set_to_none=True)
        after_boundary = model(data, epoch=2)
        after_boundary.logits.square().sum().backward()
        self.assertGreater(after_boundary.operator_ramp, 0.0)
        weight_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.instance_weight_network.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(weight_grad, 0.0)

    def test_instance_weight_network_is_neutral_at_initialization(self) -> None:
        try:
            import torch
            from src.configs.config import TrainConfig
            from src.models.h2snn_model import build_model
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        data = _make_prepared_data("acm", with_discovery=True)
        cfg = TrainConfig.from_dict(
            {
                "device": "cpu",
                "model": {"d": 4, "d_r": 2, "d_h": 8, "K": 1},
                "discovery": {
                    "enabled": True,
                    "warmup_epochs": 0,
                },
            }
        )
        model = build_model(cfg, data)
        output = model(data, epoch=-1)
        assert output.instance_weight_means is not None
        assert output.instance_weight_stds is not None
        self.assertTrue(
            torch.allclose(
                output.instance_weight_means,
                torch.ones_like(output.instance_weight_means),
            )
        )
        self.assertTrue(
            torch.equal(
                output.instance_weight_stds,
                torch.zeros_like(output.instance_weight_stds),
            )
        )

    def test_instance_and_schema_features_can_change_the_weight(self) -> None:
        import torch
        from src.configs.config import TrainConfig
        from src.data.schema_structures import DiscoveryMetadata
        from src.models.h2snn_model import build_model

        data = _make_prepared_data("acm", with_discovery=True)
        cfg = TrainConfig.from_dict(
            {
                "device": "cpu",
                "model": {"d": 4, "d_r": 2, "d_h": 8, "K": 1},
                "discovery": {"enabled": True},
            }
        )
        model = build_model(cfg, data)
        assert model.instance_weight_network is not None
        assert model.schema_embeddings is not None
        discovery = DiscoveryMetadata(
            num_schemas=2,
            schema_ids=torch.tensor([0, 1]),
            schema_incidence_mass=torch.ones(2),
            coverage_schema_index=torch.empty(0, dtype=torch.long),
            coverage_unit_index=torch.empty(0, dtype=torch.long),
            num_coverage_units=0,
        )
        first = model.instance_weight_network.net[0]
        output_layer = model.instance_weight_network.net[-1]
        latent_dim = model.edge_to_face.face_latent_dim
        with torch.no_grad():
            first.weight.zero_()
            first.bias.zero_()
            first.weight[0, 0] = 1.0
            first.weight[0, latent_dim] = 1.0
            output_layer.weight.zero_()
            output_layer.bias.zero_()
            output_layer.weight[0, 0] = 1.0
            model.schema_embeddings = torch.nn.Embedding(2, cfg.model.d_r)
            model.schema_embeddings.weight.zero_()
            model.schema_embeddings.weight[1, 0] = 1.0

        scorer = model._build_instance_weight_scorer(discovery, epoch=-1)
        assert scorer is not None
        zero_latent = torch.zeros(1, latent_dim)
        path_changed = torch.ones(1, latent_dim)
        base = scorer(0, zero_latent)
        self.assertFalse(torch.equal(base, scorer(0, path_changed)))
        self.assertFalse(torch.equal(base, scorer(1, zero_latent)))

    def test_uniform_mode_bypasses_weight_network(self) -> None:
        import torch
        from src.configs.config import TrainConfig
        from src.models.h2snn_model import build_model

        data = _make_prepared_data("acm", with_discovery=True)
        cfg = TrainConfig.from_dict(
            {
                "device": "cpu",
                "model": {"d": 4, "d_r": 2, "d_h": 8, "K": 1},
                "discovery": {
                    "enabled": True,
                    "mode": "auto",
                    "instance_weight_mode": "uniform",
                    "max_total_instances": 10,
                    "max_total_incidences": 30,
                },
            }
        )
        model = build_model(cfg, data)
        self.assertIsNone(model.instance_weight_network)
        output = model(data, epoch=-1)
        assert output.instance_weight_means is not None
        self.assertTrue(
            torch.equal(
                output.instance_weight_means,
                torch.ones_like(output.instance_weight_means),
            )
        )


class LinkPredictionDiscoveryTests(unittest.TestCase):
    def test_link_prediction_model_builds_instance_weights_from_data(self) -> None:
        try:
            import torch
            from src.configs.config import TrainConfig
            from src.data.link_pred_structures import PreparedLinkPredData
            from src.data.schema_structures import DiscoveryMetadata
            from src.data.structures import FaceGroup
            from src.models.h2snn_model import build_link_prediction_model
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        face_group = FaceGroup(
            template_name="user__artist__user",
            node_type_sequence=("user", "artist", "user"),
            face_node_index=torch.tensor([[0, 2, 1]]),
            face_edge_index=torch.tensor([[0, 1]]),
            relation_index=torch.tensor([[0, 1]]),
            face_edge_sign=torch.ones(1, 2),
            closure_edge_index=torch.tensor([[1, 0]]),
        )
        discovery = DiscoveryMetadata(
            num_schemas=1,
            schema_ids=torch.tensor([0]),
            schema_incidence_mass=torch.tensor([3.0]),
            coverage_schema_index=torch.tensor([0]),
            coverage_unit_index=torch.tensor([0]),
            num_coverage_units=1,
        )
        data = PreparedLinkPredData(
            x=torch.randn(3, 4),
            node_type=torch.tensor([0, 0, 1]),
            node_type_names=("user", "artist"),
            node_slices={"user": (0, 2), "artist": (2, 3)},
            feature_dims={"user": 4, "artist": 4},
            edge_index=torch.tensor([[0, 2], [2, 1]]),
            edge_type=torch.tensor([0, 1]),
            relation_names=("user__rel_0__artist", "artist__rel_1__user"),
            relation_tuples=(("user", "rel_0", "artist"), ("artist", "rel_1", "user")),
            target_relation=("user", "rel_0", "artist"),
            train_pos_edge_index=torch.tensor([[0], [2]]),
            train_neg_candidates=None,
            val_pos_edge_index=torch.tensor([[1], [2]]),
            val_neg_edge_index=torch.tensor([[0], [2]]),
            test_pos_edge_index=torch.tensor([[1], [2]]),
            test_neg_edge_index=torch.tensor([[0], [2]]),
            known_pos_edge_index=torch.tensor([[0, 1], [2, 2]]),
            num_target_relations=1,
            train_pos_edge_relation=torch.zeros(1, dtype=torch.long),
            val_pos_edge_relation=torch.zeros(1, dtype=torch.long),
            val_neg_edge_relation=torch.zeros(1, dtype=torch.long),
            test_pos_edge_relation=torch.zeros(1, dtype=torch.long),
            test_neg_edge_relation=torch.zeros(1, dtype=torch.long),
            neg_sampling_mode="fixed",
            neg_sampling_ratio=1,
            num_src_nodes=2,
            num_dst_nodes=1,
            src_node_offset=0,
            dst_node_offset=2,
            face_groups=(face_group,),
            closure_edge_index=torch.tensor([[1], [0]]),
            discovery=discovery,
        )
        cfg = TrainConfig.from_dict(
            {
                "data": {"dataset_name": "lastfm"},
                "device": "cpu",
                "model": {"d": 4, "d_r": 2, "d_h": 8, "K": 1, "classifier_hidden_dim": 8},
                "discovery": {
                    "enabled": True,
                    "mode": "auto",
                    "max_total_instances": 10,
                    "max_total_incidences": 30,
                },
            }
        )
        model = build_link_prediction_model(cfg, data)
        self.assertIsNotNone(model.instance_weight_network)
        self.assertIsNotNone(model.schema_embeddings)
        encoded = model.encode(
            x=data.x,
            node_type=data.node_type,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            face_groups=data.face_groups,
            closure_edge_index=data.closure_edge_index,
            num_nodes=data.num_nodes,
            num_edges=data.num_edges,
            epoch=-1,
            discovery=data.discovery,
        )
        self.assertIsNotNone(encoded.instance_weight_means)
        self.assertIsNotNone(encoded.instance_weight_stds)
        self.assertIsNotNone(encoded.instance_weight_nonempty_mask)
        scores = model.link_readout(
            x_star=encoded.x_star,
            s_star=encoded.s_star,
            edge_index=encoded.edge_index,
            F_src=encoded.F_src,
            F_tgt=encoded.F_tgt,
            edge_label_index=data.train_pos_edge_index,
            edge_relation=data.train_pos_edge_relation,
        )
        scores.square().sum().backward()
        weight_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.instance_weight_network.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(weight_grad, 0.0)


if __name__ == "__main__":
    unittest.main()
