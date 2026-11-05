"""Mutation check: inject one defect at a time (monkeypatch, no file edits) and confirm the
targeted D1 tests FAIL. A mutation that leaves its target tests green is a vacuous test."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch  # noqa: E402

import src.models.coboundary_ops as cob  # noqa: E402
import src.models.h2snn_model as h2  # noqa: E402
import src.models.map_operators as mo  # noqa: E402
import src.models.sheaf_maps as sm  # noqa: E402
import tests.test_factorized_maps as T  # noqa: E402

Op = mo.FactorizedMapOperator


def reflect_wrong_order(self, y, transpose):
    k = self.householder_k
    order = range(k - 1, -1, -1) if transpose else range(k)  # swapped
    middle = (1,) * (y.dim() - 2)
    for j in order:
        v = self.reflectors[:, j].reshape((self.reflectors.size(0),) + middle + (self.d,))
        y = y - 2.0 * v * (v * y).sum(dim=-1, keepdim=True)
    return y


def inverse_multiplies(self, y):
    return self._reflect(self._rotate(y, True), True) * mo._broadcast_rows(self.scale, y)


orig_forward = sm.FactorizedNodeToEdgeSheaf.forward


def tied_sides(self, h, edge_index, edge_type, relation_embeddings):
    out = orig_forward(self, h, edge_index, edge_type, relation_embeddings)
    return sm.FactorizedNodeToEdgeSheafOutput(maps=mo.EdgeMapPair(out.maps.src, out.maps.src))


orig_side = sm.FactorizedNodeToEdgeSheaf._side_factors


def node_blind(self, h, rel, self_idx, other_idx, edge_type, side, rotations):
    return orig_side(self, h * 0, rel, self_idx, other_idx, edge_type, side, rotations)


def delta0_T_sign(s, maps, src_idx, tgt_idx, num_nodes):
    out = torch.zeros((num_nodes, s.size(-1)), dtype=s.dtype)
    out.index_add_(0, tgt_idx, maps.tgt.apply_T(s))
    out.index_add_(0, src_idx, maps.src.apply_T(s))  # wrong sign
    return out


orig_gather = sm._gather_factorized_path_maps


def gather_transpose_as_inverse(maps, idx):
    src, tgt, _ = orig_gather(maps, idx)
    return src, tgt, src.transpose(-2, -1)


def forward_materializes(self, h, edge_index, edge_type, relation_embeddings):
    out = orig_forward(self, h, edge_index, edge_type, relation_embeddings)
    _ = out.maps.src.materialize()  # full [M, d, d]
    return out


def unguarded_normalize(raw, eps):
    return raw / raw.norm(dim=-1, keepdim=True)


orig_from_raw = Op.from_raw.__func__


def from_raw_no_eps(cls, raw_reflectors, raw_scale, rotations, group_ids, scale_eps, reflector_eps):
    op = orig_from_raw(cls, raw_reflectors, raw_scale, rotations, group_ids, scale_eps, reflector_eps)
    op.scale = op.scale - scale_eps
    return op


def transport_wrong_order(self, x):
    values = x[self.face_node_index]
    for j in range(1, self.length + 1):  # ascending: Q_i ... Q_1 instead of Q_1 ... Q_i
        tail = self._src[j - 1].apply_inverse(self._tgt[j - 1].apply(values[:, j:]))
        values = torch.cat([values[:, :j], tail], dim=1)
    return values


orig_materialise = sm.NodeToEdgeSheaf._materialise_maps


def dense_scaled(self, raw_params):
    maps, scale = orig_materialise(self, raw_params)
    return maps * 1.001, scale


def forward_allocates(shape_fn):
    def forward(self, h, edge_index, edge_type, relation_embeddings):
        out = orig_forward(self, h, edge_index, edge_type, relation_embeddings)
        _ = h.new_zeros(shape_fn(edge_index.size(1), self.d))
        return out
    return forward


orig_init = Op.__init__


def init_with_cache(self, scale, reflectors, rotations, group_ids):
    orig_init(self, scale, reflectors, rotations, group_ids)
    self._cache = scale.new_zeros(scale.shape)


MUTATIONS = [
    ("dense v1 maps x1.001 (reviewer M11)", [(sm.NodeToEdgeSheaf, "_materialise_maps", dense_scaled)],
     ["ConfigAndScopeTests.test_T11_dense_matches_pre_change_golden_outputs"]),
    ("forward allocates [M, d*d]", [(sm.FactorizedNodeToEdgeSheaf, "forward",
                                     forward_allocates(lambda m, d: (m, d * d)))],
     ["ModelIntegrationTests.test_no_dense_map_tensor_over_observed_edges_in_forward_or_backward"]),
    ("forward allocates [2M, d, d]", [(sm.FactorizedNodeToEdgeSheaf, "forward",
                                       forward_allocates(lambda m, d: (2 * m, d, d)))],
     ["ModelIntegrationTests.test_no_dense_map_tensor_over_observed_edges_in_forward_or_backward"]),
    ("operator holds an extra float cache", [(Op, "__init__", init_with_cache)],
     ["NumericalGuardTests.test_T10_storage_matches_the_bound_exactly"]),
    ("reflection order swapped", [(Op, "_reflect", reflect_wrong_order)],
     ["OperatorAlgebraTests.test_T2_apply_and_apply_T_match_dense_reference",
      "OperatorAlgebraTests.test_T3_closed_form_inverse_is_the_true_inverse"]),
    ("inverse multiplies by c", [(Op, "apply_inverse", inverse_multiplies)],
     ["OperatorAlgebraTests.test_T3_closed_form_inverse_is_the_true_inverse",
      "ChainConditionTests.test_T8_matrix_free_path_transport_equals_dense_product"]),
    ("src and tgt maps tied", [(sm.FactorizedNodeToEdgeSheaf, "forward", tied_sides)],
     ["SideAndNodeDependenceTests.test_T4_maps_are_side_specific_and_node_dependent",
      "HolonomyTests.test_T5_holonomy_is_nontrivial_and_varies_within_a_relation_sequence"]),
    ("maps ignore node states", [(sm.FactorizedNodeToEdgeSheaf, "_side_factors", node_blind)],
     ["SideAndNodeDependenceTests.test_T4_maps_are_side_specific_and_node_dependent",
      "HolonomyTests.test_T5_holonomy_is_nontrivial_and_varies_within_a_relation_sequence"]),
    ("delta0^T sign error", [(cob.CoboundaryOps, "_delta0_T_mf", staticmethod(delta0_T_sign))],
     ["ModelIntegrationTests.test_matrix_free_coboundary_and_pooling_match_dense",
      "ModelIntegrationTests.test_factorized_model_equals_v1_pipeline_fed_the_same_maps"]),
    ("face transport uses F^T as inverse", [(sm, "_gather_factorized_path_maps", gather_transpose_as_inverse)],
     ["ChainConditionTests.test_T6_full_model_chain_condition_float64",
      "ModelIntegrationTests.test_factorized_model_equals_v1_pipeline_fed_the_same_maps"]),
    ("forward materializes [M,d,d]", [(sm.FactorizedNodeToEdgeSheaf, "forward", forward_materializes)],
     ["ModelIntegrationTests.test_no_dense_map_tensor_over_observed_edges_in_forward_or_backward"]),
    ("no reflector-norm guard", [(mo, "normalize_reflectors", unguarded_normalize)],
     ["NumericalGuardTests.test_T9_finite_gradients_with_near_zero_reflectors",
      "NumericalGuardTests.test_T9_zero_reflector_mlp_keeps_model_finite"]),
    ("no scale lower bound", [(Op, "from_raw", classmethod(from_raw_no_eps))],
     ["NumericalGuardTests.test_T9_finite_gradients_with_near_zero_reflectors"]),
    ("path transport order reversed", [(mo.PathTransportOperator, "apply", transport_wrong_order)],
     ["ChainConditionTests.test_T8_matrix_free_path_transport_equals_dense_product"]),
    ("out-of-scope guard removed", [(h2, "_DENSE_MAP_ONLY_TASKS", ())],
     ["ConfigAndScopeTests.test_out_of_scope_tasks_raise_not_implemented"]),
]


def run(names: list[str]) -> dict[str, bool]:
    results = {}
    for name in names:
        suite = unittest.defaultTestLoader.loadTestsFromName(name, T)
        outcome = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
        results[name] = outcome.wasSuccessful()
    return results


def main() -> None:
    all_names = sorted({n for _, _, names in MUTATIONS for n in names})
    clean = run(all_names)
    print("unmutated:", "all pass" if all(clean.values()) else clean)
    vacuous = 0
    for label, patches, names in MUTATIONS:
        saved = [(target, attr, target.__dict__[attr] if isinstance(target, type) else getattr(target, attr))
                 for target, attr, _ in patches]
        for target, attr, value in patches:
            setattr(target, attr, value)
        try:
            results = run(names)
        finally:
            for target, attr, value in saved:
                setattr(target, attr, value)
        for name, passed in results.items():
            status = "STILL PASSES (vacuous!)" if passed else "fails as required"
            vacuous += passed
            print(f"[{label}] {name.split('.')[-1]}: {status}")
    print(f"vacuous={vacuous}")


if __name__ == "__main__":
    main()
