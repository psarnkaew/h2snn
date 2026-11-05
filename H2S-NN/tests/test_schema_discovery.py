from __future__ import annotations

import unittest


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment specific
        raise unittest.SkipTest(f"torch is required for this test: {exc}") from exc
    return torch


class GenerateSchemaCandidatesTests(unittest.TestCase):
    def test_respects_length_start_end_and_candidate_cap(self) -> None:
        from src.data.schema_discovery import generate_schema_candidates

        # author <-> paper <-> term, dense enough to produce many candidate schemas.
        relation_tuples = (
            ("author", "to", "paper"),
            ("paper", "to", "author"),
            ("paper", "to", "term"),
            ("term", "to", "paper"),
        )

        candidates = generate_schema_candidates(
            relation_tuples=relation_tuples,
            target_node_type="author",
            min_length=2,
            max_length=4,
            require_target_start=True,
            require_target_end=True,
            allow_repeated_node_types=True,
            max_identities=64,
        )

        self.assertGreater(len(candidates), 0)
        for candidate in candidates:
            self.assertGreaterEqual(candidate.length, 2)
            self.assertLessEqual(candidate.length, 4)
            self.assertEqual(candidate.node_types[0], "author")
            self.assertEqual(candidate.node_types[-1], "author")
        self.assertIn(("author", "paper", "author"), {c.node_types for c in candidates})

        with self.assertRaises(ValueError):
            generate_schema_candidates(
                relation_tuples=relation_tuples,
                target_node_type="author",
                min_length=2,
                max_length=4,
                require_target_start=True,
                require_target_end=True,
                allow_repeated_node_types=True,
                max_identities=1,
            )


class SelectSchemaPoolTests(unittest.TestCase):
    def test_picks_top_k_by_topology_aware_score(self) -> None:
        from src.data.schema_discovery import select_schema_pool
        from src.data.schema_structures import SchemaCandidate, SchemaStats

        def make(schema_id: int, disc: float, cov: float, div: float) -> tuple[SchemaCandidate, SchemaStats]:
            candidate = SchemaCandidate(
                schema_id=schema_id,
                name=f"schema_{schema_id}",
                node_types=("author", "paper", "author"),
                relation_ids=(0, 1),
                relation_names=("to", "to"),
            )
            stats = SchemaStats(
                schema_id=schema_id,
                num_instances=10,
                start_coverage=cov,
                end_coverage=cov,
                relation_diversity=1.0,
                type_diversity=div,
                avg_branching=2.0,
                estimated_cost=10.0,
                endpoint_label_disc=disc,
                start_node_sample=frozenset({schema_id}),
                end_node_sample=frozenset({schema_id + 100}),
            )
            return candidate, stats

        # schema 2 and 3 are constructed to score higher than 0 and 1 on every factor.
        pairs = [
            make(0, disc=0.001, cov=0.1, div=0.1),
            make(1, disc=0.01, cov=0.2, div=0.2),
            make(2, disc=1.0, cov=0.9, div=0.9),
            make(3, disc=0.8, cov=0.8, div=0.8),
        ]
        candidates = tuple(c for c, _ in pairs)
        stats = tuple(s for _, s in pairs)

        selected_cands, selected_stats, topology_scores = select_schema_pool(
            candidates=candidates, stats=stats, selected_schema_k=2
        )

        self.assertEqual(len(selected_cands), 2)
        self.assertEqual({c.schema_id for c in selected_cands}, {2, 3})
        self.assertEqual(topology_scores.shape[0], 2)


class EnumerateDiscoveredFacesTests(unittest.TestCase):
    def _build_dense_author_paper_graph(self):
        torch = _require_torch()
        from src.data.schema_structures import SchemaCandidate, SchemaStats

        # 2 authors (0-1), 4 papers (2-5); every author cites every paper both ways,
        # so a length-2 author->paper->author schema has a large branching factor and
        # comfortably produces more raw instances than a small max_total_instances cap.
        node_slices = {"author": (0, 2), "paper": (2, 6)}
        src, dst = [], []
        for author in range(2):
            for paper in range(2, 6):
                src.append(author)
                dst.append(paper)
        edge_index_fwd = list(zip(src, dst))
        edge_index_rev = [(d, s) for s, d in edge_index_fwd]
        all_edges = edge_index_fwd + edge_index_rev
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(
            [0] * len(edge_index_fwd) + [1] * len(edge_index_rev), dtype=torch.long
        )

        schema = SchemaCandidate(
            schema_id=0,
            name="APA",
            node_types=("author", "paper", "author"),
            relation_ids=(0, 1),
            relation_names=("to", "to"),
        )
        stats = SchemaStats(
            schema_id=0,
            num_instances=8,
            start_coverage=1.0,
            end_coverage=1.0,
            relation_diversity=1.0,
            type_diversity=1.0,
            avg_branching=4.0,
            estimated_cost=8.0,
            endpoint_label_disc=1.0,
            start_node_sample=frozenset({0, 1}),
            end_node_sample=frozenset({0, 1}),
        )
        return node_slices, edge_index, edge_type, schema, stats

    def test_global_cap_bounds_total_faces_across_pool(self) -> None:
        torch = _require_torch()
        from src.data.schema_discovery import enumerate_discovered_faces

        node_slices, edge_index, edge_type, schema, stats = self._build_dense_author_paper_graph()
        # Two copies of the same schema in the "selected pool" so faces accumulate
        # across schemas, exercising the cross-schema running total.
        schemas = (schema, schema)
        selected_stats = (stats, stats)
        topology_scores = torch.tensor([1.0, 1.0])

        uncapped_groups, _, _ = enumerate_discovered_faces(
            node_slices=node_slices,
            edge_index=edge_index,
            edge_type=edge_type,
            schemas=schemas,
            selected_stats=selected_stats,
            topology_scores=topology_scores,
            max_instances_per_anchor=10,
            max_instances_per_schema=10,
            exclude_identity_metapaths=True,
            split_seed=0,
            max_schema_length=2,
            max_total_instances=None,
        )
        uncapped_total = sum(fg.face_node_index.size(0) for fg in uncapped_groups)
        self.assertGreater(uncapped_total, 5, "test graph should produce >5 raw instances")

        capped_groups, _, _ = enumerate_discovered_faces(
            node_slices=node_slices,
            edge_index=edge_index,
            edge_type=edge_type,
            schemas=schemas,
            selected_stats=selected_stats,
            topology_scores=topology_scores,
            max_instances_per_anchor=10,
            max_instances_per_schema=10,
            exclude_identity_metapaths=True,
            split_seed=0,
            max_schema_length=2,
            max_total_instances=5,
        )
        capped_total = sum(fg.face_node_index.size(0) for fg in capped_groups)
        self.assertLessEqual(capped_total, 5)
        self.assertLess(capped_total, uncapped_total)


class LinkStage1SelectorTests(unittest.TestCase):
    def test_selector_negatives_preserve_positive_source_alignment(self) -> None:
        torch = _require_torch()
        from src.data.link_stage1 import _build_selector_edges

        train_pos = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
        # Source 0 is saturated over the full destination range; source 1 is not.
        known_train = torch.tensor([[0, 0, 1], [2, 3, 2]], dtype=torch.long)
        edge_labels, labels, kept_positives = _build_selector_edges(
            train_pos_edge_index=train_pos,
            known_positive_edge_index=known_train,
            dst_range=(2, 4),
            max_examples=4,
            seed=13,
        )
        self.assertEqual(kept_positives.size(1), 1)
        self.assertEqual(int(kept_positives[0, 0]), 1)
        negatives = edge_labels[:, labels < 0.5]
        self.assertEqual(int(negatives[0, 0]), 1)
        self.assertEqual(int(kept_positives[0, 0]), int(negatives[0, 0]))

    def test_inner_selection_target_and_reverse_edges_are_removed(self) -> None:
        torch = _require_torch()
        from src.data.link_stage1 import _remove_inner_selection_edges

        edge_index = torch.tensor(
            [[0, 4, 1, 5], [4, 0, 5, 1]],
            dtype=torch.long,
        )
        edge_type = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        filtered_edges, filtered_types = _remove_inner_selection_edges(
            edge_index=edge_index,
            edge_type=edge_type,
            relation_tuples=(
                ("user", "to_artist", "artist"),
                ("artist", "to_user", "user"),
            ),
            source_node_type="user",
            destination_node_type="artist",
            held_out_positive_edge_index=torch.tensor([[0], [4]]),
        )
        self.assertTrue(
            torch.equal(filtered_edges, torch.tensor([[1, 5], [5, 1]]))
        )
        self.assertTrue(torch.equal(filtered_types, torch.tensor([0, 1])))

    def test_link_role_evidence_uses_only_inner_fit_positives(self) -> None:
        torch = _require_torch()
        from src.data.link_stage1 import _build_fit_role_signal

        fit_positive = torch.tensor([[0], [2]], dtype=torch.long)
        roles = _build_fit_role_signal(
            edge_index=fit_positive,
            fit_positive_edge_index=fit_positive,
            num_nodes=4,
            num_steps=0,
        )

        self.assertTrue(torch.equal(roles[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(roles[2], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.equal(roles[1], torch.zeros(2)))
        self.assertTrue(torch.equal(roles[3], torch.zeros(2)))

    def test_selector_positive_subset_is_independent_of_negative_pool(self) -> None:
        torch = _require_torch()
        from src.data.link_stage1 import _build_selector_edges

        train_pos = torch.tensor([[0, 1], [4, 5]], dtype=torch.long)
        known_a = train_pos
        known_b = torch.tensor(
            [[0, 1, 0, 1], [4, 5, 6, 7]],
            dtype=torch.long,
        )
        _, _, positive_a = _build_selector_edges(
            train_pos_edge_index=train_pos,
            known_positive_edge_index=known_a,
            dst_range=(4, 8),
            max_examples=4,
            seed=37,
        )
        _, _, positive_b = _build_selector_edges(
            train_pos_edge_index=train_pos,
            known_positive_edge_index=known_b,
            dst_range=(4, 8),
            max_examples=4,
            seed=37,
        )

        self.assertTrue(torch.equal(positive_a, positive_b))

    def test_link_probe_selector_selects_task_aligned_schema(self) -> None:
        torch = _require_torch()
        from src.configs.config import TrainConfig
        from src.data.link_stage1 import build_link_amp_ss_faces

        cfg = TrainConfig.from_dict(
            {
                "data": {"dataset_name": "lastfm"},
                "discovery": {
                    "enabled": True,
                    "mode": "auto",
                    "min_schema_length": 2,
                    "max_schema_length": 2,
                    "max_schema_identities": 8,
                    "max_schema_candidates": 8,
                    "selected_schema_k": 1,
                    "max_instances_per_schema": 8,
                    "max_total_instances": 8,
                    "max_total_incidences": 24,
                    "selector_num_attempts": 4,
                    "selector_max_examples": 4,
                    "selector_holdout_ratio": 0.5,
                    "selector_probe_max_iter": 25,
                    "selector_stop_tolerance": 1e-12,
                },
            }
        )
        node_slices = {"user": (0, 2), "tag": (2, 4), "artist": (4, 6)}
        relation_tuples = (
            ("user", "to_tag", "tag"),
            ("tag", "to_artist", "artist"),
        )
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3],
                [2, 3, 4, 5],
            ],
            dtype=torch.long,
        )
        edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        train_pos = torch.tensor(
            [
                [0, 1],
                [4, 5],
            ],
            dtype=torch.long,
        )

        face_groups, metadata, closure, report = build_link_amp_ss_faces(
            config=cfg,
            node_slices=node_slices,
            edge_index=edge_index,
            edge_type=edge_type,
            relation_tuples=relation_tuples,
            source_node_type="user",
            destination_node_type="artist",
            train_pos_edge_index=train_pos,
            known_positive_edge_index=train_pos,
            dst_range=(4, 6),
            propagated_roles=torch.zeros(6, 2),
        )

        self.assertIsNotNone(report)
        self.assertEqual(report.selected_indices, (0,))  # type: ignore[union-attr]
        self.assertEqual(report.stop_reason, "max_schemas_reached")  # type: ignore[union-attr]
        self.assertEqual(  # type: ignore[union-attr]
            report.protocol_version,
            "schema_conditioned_v1",
        )
        self.assertTrue(report.selector_unit_hash)  # type: ignore[union-attr]
        self.assertTrue(report.materialized_face_hash)  # type: ignore[union-attr]
        self.assertEqual(len(report.candidate_statistics), 1)  # type: ignore[union-attr]
        self.assertEqual(
            {item.definition for item in report.coverage_diagnostics},  # type: ignore[union-attr]
            {
                "link_positive_pairs_postquota",
                "link_endpoint_nodes_postquota_diagnostic_only",
            },
        )
        self.assertIsNotNone(metadata)
        self.assertEqual(len(face_groups), 1)
        self.assertEqual(face_groups[0].node_type_sequence, ("user", "tag", "artist"))
        self.assertEqual(closure.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
