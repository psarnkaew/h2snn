"""K1 mutations: each must make its targeted tests fail (assertions, not shape errors: each
mutation changes the input width and the input together).
Run from H2S-NN/: python ../docs/WISP_evidence/k1_mutation_check.py"""
import subprocess
import sys
from pathlib import Path

CTX = "tests.test_v2_model.TwoLatentContextTests."
MUTATIONS = {
    "map latent fed h0 (the D-I1a behaviour with two encoders)": (
        [("            self.map_encoder = encoder_class(geometry_dim, hop_dim,",
          "            self.map_encoder = encoder_class(path_input_dim, hop_dim,"),
         ("map_latent = latent if self.map_encoder is None else self._map_latent(h0, group)",
          "map_latent = latent if self.map_encoder is None else self._map_latent(path_input, group)")],
        [CTX + "test_a_endpoint_edge_maps_read_the_endpoints_context_not_intermediates",
         CTX + "test_c_free_maps_read_every_instance_node_context_through_the_latent"]),
    "g fed the geometry under the default": (
        [("        self.path_weight_context = ablation.path_weight_context",
          "        self.path_weight_context = True")],
        [CTX + "test_b_g_reads_h0_only_unless_the_appendix_switch",
         "tests.test_wisp.WispLadderTests.test_TH7_intermediate_and_its_neighbourhood_under_the_default_context"]),
    "pair-attention latents fed h0 (extra)": (
        [("        path_input_dim = (geometry_dim if self.path_weight_context\n"
          "                          or ablation.coupling == \"pair_attention\" else d)",
          "        path_input_dim = (geometry_dim if self.path_weight_context else d)"),
         ("latents.append(self._path_latent(h0 if geometry is None else geometry, group))",
          "latents.append(self._path_latent(h0, group))")],
        [CTX + "test_d_pair_attention_cell_scores_read_the_context"]),
}
path = Path("src/models/h2snn_model.py")
original = path.read_text(encoding="utf-8")
caught_all = 0
for name, (edits, tests) in MUTATIONS.items():
    text = original
    for old, new in edits:
        assert text.count(old) == 1, (name, old[:60])
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    try:
        results = []
        for test in tests:
            out = subprocess.run([sys.executable, "-m", "unittest", test], capture_output=True, text=True)
            tail = out.stderr.strip().splitlines()[-1]
            kind = "assertion" if "AssertionError" in out.stderr else ("error" if out.returncode else "-")
            results.append((test.split(".")[-1], out.returncode != 0, kind, tail))
    finally:
        path.write_text(original, encoding="utf-8")
    caught = all(r[1] for r in results)
    caught_all += caught
    print(f"{name}: {'CAUGHT' if caught else 'NOT CAUGHT'}")
    for test, failed, kind, tail in results:
        print(f"    {test}: {'fails' if failed else 'passes'} ({kind}; {tail})")
clean = subprocess.run([sys.executable, "-m", "unittest", CTX[:-1]], capture_output=True, text=True)
print(f"caught {caught_all}/{len(MUTATIONS)}; clean after restore: {clean.stderr.strip().splitlines()[-1]}")
