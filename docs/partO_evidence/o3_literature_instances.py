"""Part O O3 evidence (data read only; no model, no metric): for each dataset's literature list
(`LITERATURE_METAPATHS`), the resolved relation-level hops on the real HGB graph (tied fact layer)
and the number of target nodes from which 8 uniform walks along the path give at least one simple
instance, plus the total accepted walks. Run from H2S-NN/:

    python ../docs/partO_evidence/o3_literature_instances.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "H2S-NN"
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.configs.config import LITERATURE_METAPATHS  # noqa: E402
from src.data.path_sampler import build_half_edges, hop_csr, sample_walks  # noqa: E402
from src.data.schemas import resolve_schemas  # noqa: E402
from tests.test_partO_literature_row import RealListTests, _real_layer  # noqa: E402

WALKS = 8


def main() -> None:
    for dataset, target in RealListTests.TARGETS.items():
        layer, tuples, types, slices, node_type, edge_index, edge_type = _real_layer(dataset)
        start, end = slices[target]
        print(f"== {dataset}: target {target} ({end - start} nodes); kept relations "
              f"{layer.kept_names}; self-inverse {[layer.kept_names[c] for c in layer.self_inverse]}")
        half = build_half_edges(edge_index, edge_type, node_type.numel(), layer.self_inverse,
                                backward=True)
        generator = torch.Generator().manual_seed(0)
        anchors = torch.arange(start, end)
        for schema in resolve_schemas(LITERATURE_METAPATHS[dataset], layer, tuples):
            walks = sample_walks([hop_csr(half, r, d) for r, d in schema.hops], anchors, WALKS,
                                 generator)
            reached = walks.nodes[:, 0].unique().numel() if walks.nodes.numel() else 0
            hops = ", ".join(f"({layer.kept_names[r]}, {d:+d})" for r, d in schema.hops)
            print(f"  {schema.name:6s} {'-'.join(schema.node_types):40s} hops [{hops}]  "
                  f"targets with an instance in {WALKS} walks: {reached}/{end - start}; "
                  f"accepted walks {walks.nodes.size(0)}")


if __name__ == "__main__":
    main()
