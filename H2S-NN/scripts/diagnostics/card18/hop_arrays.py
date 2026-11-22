"""Card 18 report: every listed ACM hop on acm_pp, before (pre-card-18 `hop`) and after, on the baseline bundle's
graph (untied: all 8 relations) and on the same graph without paper-rel_1-paper (the tied view, where the fallback
runs); and H2G-former's walk-graph edge type of each schema hop on both. Per hop: relation, forward?, edge count,
md5 of the (src, dst) pairs, first 3 pairs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from tests.test_card18_metapath_direction import _graph, legacy_hop, without  # noqa: E402


def describe(graph, result) -> dict:
    (s, d), r, forward = result
    pairs = np.stack([s, d], axis=1).astype(np.int64)
    return {"relation": "-".join(graph.relations[r]), "forward": bool(forward), "edges": int(len(s)),
            "md5": hashlib.md5(np.ascontiguousarray(pairs).tobytes()).hexdigest(), "first": pairs[:3].tolist()}


def main() -> None:
    from src.baselines.adapters._h2gformer_encoders import listed_paths, walk_hop, with_reverse_edges

    untied, target = _graph("acm_pp")
    out = {"target": list(target)}
    for label, graph in (("bundle_untied", untied), ("tied_view", without(untied, ("paper", "rel_1", "paper")))):
        rows = []
        for name, hops in listed_paths("acm_pp"):
            for j, hop in enumerate(hops):
                rows.append({"path": name, "hop": j + 1, "listed": "-".join(hop),
                             "before": describe(graph, legacy_hop(graph, hop)),
                             "after": describe(graph, graph.hop(hop))})
        walk = with_reverse_edges({rel: torch.zeros(2, 0, dtype=torch.long) for rel in graph.relations})
        out[label] = {"relations": ["-".join(r) for r in graph.relations], "magnn_hops": rows,
                      "h2gformer_walk_hops": [{"path": n, "listed": "-".join(h),
                                               "walk_graph": "-".join(walk_hop(h, walk))}
                                              for n, hops in listed_paths("acm_pp") for h in hops]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
