"""Chain-condition tests: delta1 . delta0 == 0.

This is the correctness gate for the Road A face construction. `delta1 . delta0 = 0`
is a *theorem* for a cellular sheaf on a 2-complex, resting on two hypotheses:

1. functoriality -- the edge->face map composes with the node->edge map, and
2. a genuine closed boundary cycle with consistent incidence numbers.

The face maps are therefore built by transport along the path rather than emitted
freely by a network:

    T_{v0} = A_f                                   (one free d x d map per instance)
    T_{v_i} = T_{v_{i-1}} (F^{e_i}_src)^-1 F^{e_i}_tgt
    F_{e_i<|f} = T_{v_{i-1}} (F^{e_i}_src)^-1

and the synthetic closure edge eC = (v_L, v_0) -- which points *along* the cycle, so
its incidence sign is +1 like every other boundary edge -- gets

    F^{eC}_tgt = I,  F^{eC}_src = T_{v0}^-1 T_{v_L},  F_{eC<|f} = T_{v0} = A_f

Every boundary term then collapses to `T_v x_v - T_u x_u` and the sum around the closed
cycle telescopes to exactly zero, for all x, with no holonomy assumption on the sheaf.
Note `F^{eC}_src` equals the path holonomy and is independent of `A_f`.
"""

from __future__ import annotations

import unittest


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment specific
        raise unittest.SkipTest(f"torch is required for this test: {exc}") from exc
    return torch


def _random_conformal_orthogonal(torch, count: int, d: int, seed: int):
    """Mimic NodeToEdgeSheaf._materialise_maps: an orthogonal matrix times a positive scale."""
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(count, d, d, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw)
    scales = torch.exp(
        torch.randn(count, generator=generator, dtype=torch.float64).clamp(-1.0, 1.0)
    )
    return q * scales.view(-1, 1, 1)


class _PathFixture:
    """A single simple path v0 -> ... -> vL plus its closure edge, as one face.

    Edge ids 0..L-1 are the path edges in traversal order; edge id L is the closure
    edge (v_L, v_0). Node ids are 0..L, with v_L reusing no earlier node -- the path
    is simple, which is what the transport construction requires.
    """

    def __init__(self, torch, length: int, d: int, seed: int) -> None:
        self.torch = torch
        self.length = length
        self.d = d
        self.num_nodes = length + 1
        self.node_path = list(range(length + 1))

        path_src = self.node_path[:-1]
        path_tgt = self.node_path[1:]
        # closure edge points along the cycle: (v_L -> v_0)
        self.edge_index = torch.tensor(
            [path_src + [self.node_path[-1]], path_tgt + [self.node_path[0]]],
            dtype=torch.long,
        )
        self.F_src = _random_conformal_orthogonal(torch, length, d, seed)
        self.F_tgt = _random_conformal_orthogonal(torch, length, d, seed + 1)
        self.A_f = _random_conformal_orthogonal(torch, 1, d, seed + 2)
        self.h0 = torch.randn(
            self.num_nodes,
            d,
            generator=torch.Generator().manual_seed(seed + 3),
            dtype=torch.float64,
        )

    def face_edge_index(self):
        """[1, L+1] -- path edges then the closure edge."""
        return self.torch.arange(self.length + 1, dtype=self.torch.long).view(1, -1)


class TransportCompositionTest(unittest.TestCase):
    """The construction itself, isolated from the model plumbing."""

    def _check_telescopes(self, length: int, d: int = 5, seed: int = 0) -> None:
        torch = _torch()
        from src.models.sheaf_maps import compose_face_transport

        fixture = _PathFixture(torch, length=length, d=d, seed=seed)

        transport = compose_face_transport(
            A_f=fixture.A_f,
            F_src_path=fixture.F_src.unsqueeze(0),
            F_tgt_path=fixture.F_tgt.unsqueeze(0),
        )

        # Full edge set: path edges then closure.
        F_src_all = torch.cat([fixture.F_src, transport.closure_F_src], dim=0)
        F_tgt_all = torch.cat([fixture.F_tgt, transport.closure_F_tgt], dim=0)

        from src.models.coboundary_ops import CoboundaryOps

        delta0_h0 = CoboundaryOps.delta0(
            x=fixture.h0,
            F_src=F_src_all,
            F_tgt=F_tgt_all,
            src_idx=fixture.edge_index[0],
            tgt_idx=fixture.edge_index[1],
        )

        # delta1 for the single face: sum over boundary edges of F_{e<|f} s_e, all signs +1.
        per_edge = torch.matmul(
            transport.face_maps[0], delta0_h0.unsqueeze(-1)
        ).squeeze(-1)
        delta1_delta0 = per_edge.sum(dim=0)

        scale = delta0_h0.norm()
        self.assertGreater(float(scale), 1e-6, "degenerate fixture: delta0 h0 vanished")
        relative = float(delta1_delta0.norm() / scale)
        self.assertLess(
            relative,
            1e-9,
            f"L={length}: delta1(delta0 h0) should vanish, relative norm={relative:.3e}",
        )

    def test_telescopes_at_length_2(self) -> None:
        self._check_telescopes(length=2)

    def test_telescopes_at_length_4(self) -> None:
        """The case the manual DBLP APTPA template and every AMPSS schema use."""
        self._check_telescopes(length=4)

    def test_telescopes_at_length_3(self) -> None:
        self._check_telescopes(length=3, seed=17)

    def test_closure_map_is_path_holonomy_independent_of_A_f(self) -> None:
        """F^{eC}_src is the path holonomy: changing A_f must not change it."""
        torch = _torch()
        from src.models.sheaf_maps import compose_face_transport

        fixture = _PathFixture(torch, length=4, d=5, seed=31)
        other_A_f = _random_conformal_orthogonal(torch, 1, 5, seed=99)

        first = compose_face_transport(
            A_f=fixture.A_f,
            F_src_path=fixture.F_src.unsqueeze(0),
            F_tgt_path=fixture.F_tgt.unsqueeze(0),
        )
        second = compose_face_transport(
            A_f=other_A_f,
            F_src_path=fixture.F_src.unsqueeze(0),
            F_tgt_path=fixture.F_tgt.unsqueeze(0),
        )

        torch.testing.assert_close(first.closure_F_src, second.closure_F_src)
        torch.testing.assert_close(
            first.closure_F_tgt, second.closure_F_tgt
        )  # both are identity

    def test_face_map_for_closure_edge_equals_A_f(self) -> None:
        torch = _torch()
        from src.models.sheaf_maps import compose_face_transport

        fixture = _PathFixture(torch, length=2, d=4, seed=5)
        transport = compose_face_transport(
            A_f=fixture.A_f,
            F_src_path=fixture.F_src.unsqueeze(0),
            F_tgt_path=fixture.F_tgt.unsqueeze(0),
        )
        torch.testing.assert_close(transport.face_maps[:, -1], fixture.A_f)


class Delta1AdjointnessTest(unittest.TestCase):
    """delta1_T must be the exact transpose of delta1 once face_edge_index gains a column."""

    def test_delta1_transpose_is_adjoint(self) -> None:
        torch = _torch()
        from src.models.coboundary_ops import CoboundaryOps
        from src.models.sheaf_maps import FaceMapCache, compose_face_transport

        length, d = 4, 5
        fixture = _PathFixture(torch, length=length, d=d, seed=11)
        transport = compose_face_transport(
            A_f=fixture.A_f,
            F_src_path=fixture.F_src.unsqueeze(0),
            F_tgt_path=fixture.F_tgt.unsqueeze(0),
        )
        num_edges = length + 1
        face_edge_index = fixture.face_edge_index()

        cache = FaceMapCache(
            template_name="synthetic",
            face_edge_index=face_edge_index,
            relation_index=torch.zeros_like(face_edge_index),
            face_edge_sign=torch.ones_like(face_edge_index, dtype=torch.float64),
            maps=transport.face_maps,
            group_id=0,
            instance_start=0,
            instance_end=1,
        )

        generator = torch.Generator().manual_seed(77)
        s = torch.randn(num_edges, d, generator=generator, dtype=torch.float64)
        f = torch.randn(1, d, generator=generator, dtype=torch.float64)

        forward = CoboundaryOps.delta1(
            s=s, face_caches=(cache,), map_builder=lambda c: c.maps
        )
        lhs = float((forward.concatenated * f).sum())

        adjoint_result = forward._replace(grouped=(f,), concatenated=f)
        back = CoboundaryOps.delta1_T(
            delta1_result=adjoint_result,
            face_caches=(cache,),
            num_edges=num_edges,
            d=d,
        )
        rhs = float((s * back).sum())

        self.assertAlmostEqual(
            lhs, rhs, places=9, msg="<delta1 s, f> != <s, delta1^T f>"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
