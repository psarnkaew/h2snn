"""Part M3b B4: the label-leak audit (`src/data/leak_audit.py`) on synthetic graphs.

A venue relation deliberately left in the message graph is reported (by name and by the probe); with it
removed the audit passes (negative control). A label-carrying relation with an innocent name is caught by the
probe alone; a random relation is not flagged.
"""

from __future__ import annotations

import unittest

import torch


def _graph(with_venue: bool, with_disguised: bool = False):
    from torch_geometric.data import HeteroData

    generator = torch.Generator().manual_seed(0)
    papers, venues, fields = 300, 6, 20
    data = HeteroData()
    venue_of = torch.randint(0, venues, (papers,), generator=generator)
    y = venue_of.clone()
    y[:30] = -1  # unlabelled papers
    data["paper"].x = torch.rand(papers, 4, generator=generator)
    data["paper"].y = y
    data["field"].x = torch.rand(fields, 4, generator=generator)
    data["paper", "PF_in", "field"].edge_index = torch.stack([
        torch.arange(papers).repeat(2), torch.randint(0, fields, (2 * papers,), generator=generator)])
    data["paper", "PP_cite", "paper"].edge_index = torch.randint(0, papers, (2, 900), generator=generator)
    if with_venue:
        data["venue"].num_nodes = venues
        data["paper", "PV_Journal", "venue"].edge_index = torch.stack([torch.arange(papers), venue_of])
        data["venue", "rev_PV_Journal", "paper"].edge_index = torch.stack([venue_of, torch.arange(papers)])
    if with_disguised:  # the label under an innocent name
        data["topic"].num_nodes = venues
        data["paper", "has", "topic"].edge_index = torch.stack([torch.arange(papers), venue_of])
    return data


class LeakAuditTests(unittest.TestCase):
    def test_venue_relation_left_in_is_reported(self) -> None:
        from src.data.leak_audit import audit

        result = audit(_graph(with_venue=True), "paper")
        self.assertFalse(result["passed"])
        self.assertIn("venue", result["inventory"]["label_source_names_present"])
        self.assertIn("paper|PV_Journal|venue", result["carriers"])
        self.assertIn("venue|rev_PV_Journal|paper", result["carriers"])
        probe = {"|".join(p.relation): p for p in result["probes"]}["paper|PV_Journal|venue"]
        self.assertEqual(probe.majority_share, 1.0)
        self.assertLess(probe.control_share, 0.5)

    def test_negative_control_without_the_venue_relation_passes(self) -> None:
        from src.data.leak_audit import CARRIER_SHARE, audit

        result = audit(_graph(with_venue=False), "paper")
        self.assertTrue(result["passed"], result["carriers"])
        self.assertEqual(result["inventory"]["label_source_names_present"], [])
        for probe in result["probes"]:
            self.assertLess(probe.majority_share, CARRIER_SHARE, probe)

    def test_label_carrier_with_an_innocent_name_is_caught_by_the_probe(self) -> None:
        from src.data.leak_audit import audit

        result = audit(_graph(with_venue=False, with_disguised=True), "paper")
        self.assertFalse(result["passed"])
        self.assertEqual(result["inventory"]["label_source_names_present"], [])
        self.assertEqual(result["carriers"], ["paper|has|topic"])

    def test_majority_share_on_a_known_example(self) -> None:
        from src.data.leak_audit import majority_share

        # hub 0: labels 1,1,2 -> 2 of 3; hub 1: labels 0,0 -> 2 of 2; hub 2: one neighbour (ignored)
        hub = torch.tensor([0, 0, 0, 1, 1, 2])
        label = torch.tensor([1, 1, 2, 0, 0, 3])
        share, hubs, incidences, keep = majority_share(hub, label)
        self.assertEqual((hubs, incidences), (2, 5))
        self.assertAlmostEqual(share, 4 / 5)
        self.assertEqual(keep.tolist(), [True] * 5 + [False])


if __name__ == "__main__":
    unittest.main()
