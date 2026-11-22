"""Card 18: MAGNN's listed hops on a relation the task dropped (`TypedGraph.hop`, `_h2gformer_encoders.dropped_hop`).

(a) acm_pp (real data): the cited-by path PrPAP differs from PcPAP, and every PrPAP instance's first hop is a
    reversed citation (v1 cites v0 in paper-rel_0-paper). The baseline bundle is untied (`BASELINE_DATA_PINS`), so
    paper-rel_1-paper is stored and the hop reads it; on the same graph without rel 1 (the tied case, where
    `hop` falls back to the kept relation) the PrPAP instances are the same.
(b) dblp_ap, imdb_ma, freebase_bb (real data): every listed hop MAGNN reads gives the same edge arrays, relation id
    and direction as the pre-card-18 `hop` (copied below).
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DERIVED = PROJECT_ROOT / "data" / "LP" / "lp_derived"


def _graph(task: str):
    from src.baselines.adapters.magnn import TypedGraph
    from src.baselines.bundles import build_bundle
    from src.baselines.protocol import LINK_PREDICTION

    data = build_bundle(LINK_PREDICTION, task, overrides={"device": "cpu"}).data
    graph = TypedGraph.from_global(tuple(data.node_type_names), dict(data.node_slices),
                                   tuple(data.relation_tuples), data.edge_index, data.edge_type)
    return graph, tuple(data.target_relation)


def without(graph, relation):
    """`graph` with `relation` removed (the tied fact layer's view of an exact-reverse pair)."""
    from src.baselines.adapters.magnn import TypedGraph

    keep = [i for i, r in enumerate(graph.relations) if r != relation]
    return TypedGraph(graph.node_types, graph.slices, tuple(graph.relations[i] for i in keep),
                      [graph.edges[i] for i in keep])


def legacy_hop(graph, hop):
    """`TypedGraph.hop` before card 18 (magnn.py at 2e4881c, lines 195-210)."""
    a, rel, b = hop
    if hop in graph.relations:
        r = graph.relations.index(hop)
        return graph.edges[r], r, True
    if rel.startswith("rev_") and (b, rel[4:], a) in graph.relations:
        r = graph.relations.index((b, rel[4:], a))
        s, d = graph.edges[r]
        return (d, s), r, False
    between = [r for r, (s, _, d) in enumerate(graph.relations) if {s, d} == {a, b}]
    if len(between) == 1:
        r = between[0]
        s, d = graph.edges[r]
        return ((s, d), r, True) if graph.relations[r][0] == a else ((d, s), r, False)
    raise KeyError(hop)


@unittest.skipUnless((DERIVED / "acm_pp" / "split_meta.json").exists(), "data/LP/lp_derived/acm_pp not built")
class CitedByDirectionTests(unittest.TestCase):
    def test_prpap_differs_from_pcpap_and_walks_citations_backwards(self) -> None:
        from src.baselines.adapters._h2gformer_encoders import listed_paths
        from src.baselines.adapters.magnn import path_instances

        graph, _ = _graph("acm_pp")
        paths = dict(listed_paths("acm_pp"))
        instances = {name: path_instances([graph.hop(h)[0] for h in paths[name]], graph.num_nodes)
                     for name in ("PcPAP", "PrPAP")}
        pc = {tuple(row) for row in instances["PcPAP"].tolist()}
        pr = {tuple(row) for row in instances["PrPAP"].tolist()}
        self.assertGreater(len(pr), 0)
        self.assertNotEqual(pc, pr)
        cite = graph.relations.index(("paper", "rel_0", "paper"))
        src, dst = graph.edges[cite]
        cites = set(zip(src.tolist(), dst.tolist()))
        first = {(int(v0), int(v1)) for v0, v1 in instances["PrPAP"][:, :2].tolist()}
        self.assertTrue(all((v1, v0) in cites for v0, v1 in first))
        tied = without(graph, ("paper", "rel_1", "paper"))
        tied_pr = path_instances([tied.hop(h)[0] for h in paths["PrPAP"]], tied.num_nodes)
        self.assertEqual({tuple(row) for row in tied_pr.tolist()}, pr)
        tied_cite = tied.relations.index(("paper", "rel_0", "paper"))
        self.assertEqual(tied.hop(paths["PrPAP"][0])[1:], (tied_cite, False))
        self.assertEqual(tied.hop(paths["PcPAP"][0])[1:], (tied_cite, True))


class OtherTasksUnchangedTests(unittest.TestCase):
    def test_listed_hops_equal_the_pre_card18_hop(self) -> None:
        from src.baselines.adapters._h2gformer_encoders import listed_paths

        for task in ("dblp_ap", "imdb_ma", "freebase_bb"):
            if not (DERIVED / task / "split_meta.json").exists():
                self.skipTest(f"data/LP/lp_derived/{task} not built")
            with self.subTest(task=task):
                graph, _ = _graph(task)
                hops = {h for _, path in listed_paths(task) for h in path}
                self.assertGreater(len(hops), 0)
                for hop in sorted(hops):
                    (s_new, d_new), r_new, f_new = graph.hop(hop)
                    (s_old, d_old), r_old, f_old = legacy_hop(graph, hop)
                    self.assertEqual((r_new, f_new), (r_old, f_old), hop)
                    self.assertTrue(np.array_equal(s_new, s_old) and np.array_equal(d_new, d_old), hop)


if __name__ == "__main__":
    unittest.main()
