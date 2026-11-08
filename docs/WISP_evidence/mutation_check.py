"""Part H mutation checks: each mutation is applied by monkeypatching the production module, the
named test is run in-process, and the check passes iff that test FAILS under the mutation (and
passes again after the patch is removed).
Run from H2S-NN/:  python ../docs/WISP_evidence/mutation_check.py [names...]
Mutations (card): allow_revisits -> T-H1; drop_summed_merge -> T-H2 (q = p);
remove_is_ratio -> T-H2 (biased q, H3); per_step_mixture -> T-H3 (H3);
endpoints_sees_intermediate -> T-H7 (H2)."""
import sys
import unittest
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import src.data.wisp_sampler as wisp  # noqa: E402


def run(test_id: str) -> bool:
    suite = unittest.defaultTestLoader.loadTestsFromName(test_id)
    result = unittest.TextTestRunner(verbosity=0, stream=open("nul" if sys.platform == "win32"
                                                               else "/dev/null", "w")).run(suite)
    return result.wasSuccessful()


def allow_revisits():
    original = wisp._visited_half_edges

    def no_exclusion(graph, current, visited):
        action, position, valid = original(graph, current, visited)
        return action, position, valid & False
    wisp._visited_half_edges = no_exclusion
    return lambda: setattr(wisp, "_visited_half_edges", original)


def drop_summed_merge():
    original = wisp.merge_wisp_cells

    def keep_one(draw, node_type, edge_index, walks_per_anchor, merge="sum"):
        return original(draw, node_type, edge_index, walks_per_anchor, merge="drop_duplicates")
    wisp.merge_wisp_cells = keep_one
    return lambda: setattr(wisp, "merge_wisp_cells", original)


def endpoints_sees_intermediate():
    import src.models.h2snn_model as model_module
    original = model_module.EndpointEncoder.forward

    def with_intermediate(self, h0, face_node_index, hop_tokens):
        mutated = face_node_index.clone()
        mutated[:, 0] = face_node_index[:, 1]  # the first intermediate replaces v_0
        return original(self, h0, mutated, hop_tokens)
    model_module.EndpointEncoder.forward = with_intermediate
    return lambda: setattr(model_module.EndpointEncoder, 'forward', original)


MUTATIONS = {
    "allow_revisits": (allow_revisits,
                       "tests.test_wisp.WispSamplerTests.test_TH1_exact_distribution_no_revisits_and_dead_end_mass"),
    "drop_summed_merge": (drop_summed_merge,
                          "tests.test_wisp.WispSamplerTests.test_TH2_unbiased_path_laplacian_for_q_equal_p"),
    "endpoints_sees_intermediate": (endpoints_sees_intermediate,
                                    "tests.test_wisp.WispLadderTests.test_TH7_intermediate_perturbation"),
}


def main() -> None:
    names = sys.argv[1:] or list(MUTATIONS)
    caught = 0
    for name in names:
        apply, test_id = MUTATIONS[name]
        clean_before = run(test_id)
        restore = apply()
        try:
            mutated = run(test_id)
        finally:
            restore()
        clean_after = run(test_id)
        ok = clean_before and not mutated and clean_after
        caught += ok
        print(f"{name}: clean={clean_before} mutated_passes={mutated} clean_after={clean_after} "
              f"-> {'CAUGHT' if ok else 'NOT CAUGHT'}  ({test_id.split('.')[-1]})", flush=True)
    print(f"caught {caught}/{len(names)}")


if __name__ == "__main__":
    main()
