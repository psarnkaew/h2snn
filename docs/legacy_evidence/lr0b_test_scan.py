"""LR-0b aid: per test function, the legacy names in its body (and in the helpers of its module it calls
directly), and whether the function builds a config/model without `anchored` (it then runs the v2 model
through today's default). Run from H2S-NN/: python ../docs/legacy_evidence/lr0b_test_scan.py > out.tsv
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

LEGACY = re.compile(
    r"chebyshev|filter_bank|first_order|pair_attention|PairAttention|listen_only|cooperative|"
    r"sigmoid_gamma|endpoint_edges|holonomy_edges|free_path_maps|uniform_instance|coupling_gate|"
    r"eta_init|gamma_init|per_side_scale|layer_dropout|identity_maps|no_path_cells|theta_eta|"
    r"theta_gamma|filter_norm|listen_gate|broadcast_gate|VertexPathSheafLaplacian|build_operator|"
    r"\bdiffuse\b|propagate\(|EdgePairTerm|FreeIncidenceOperator|delta_loss|analyze_kernel|"
    r"operator_diagnostics|eta_lambda_max|lambda_max|energy_trace|coupling_stats|\.eta\(\)|"
    r"\.gamma\(\)|\"K\"|'K'|v2_golden")
BUILDS = re.compile(r"TrainConfig\.from_dict|TrainConfig\(|build_model\(|H2SNN\(|"
                    r"build_link_prediction_model|build_graph_classification_model|_config\(|config\(")


def main() -> None:
    files = sorted(Path("tests").glob("test_*.py"))
    print("file\tclass\ttest\tlegacy_names\tbuilds\tanchored_in_body\tanchored_in_module")
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        module_anchored = "anchored" in source
        helpers = {n.name: ast.get_source_segment(source, n) or "" for n in tree.body
                   if isinstance(n, ast.FunctionDef)}
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)
                       and n.name.startswith("test")]:
                body = ast.get_source_segment(source, fn) or ""
                called = {m.group(1) for m in re.finditer(r"\b(_?\w+)\(", body)} & set(helpers)
                text = body + "\n".join(helpers[h] for h in called)
                names = sorted(set(m.group(0) for m in LEGACY.finditer(text)))
                builds = bool(BUILDS.search(text))
                print(f"{path.name}\t{cls.name}\t{fn.name}\t{','.join(names)}\t{int(builds)}\t"
                      f"{int('anchored' in text)}\t{int(module_anchored)}")


if __name__ == "__main__":
    sys.exit(main())
