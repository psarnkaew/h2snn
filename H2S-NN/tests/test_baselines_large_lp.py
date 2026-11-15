"""Card baselines-h2gb C6 (slim mode): link-level neighbour sampling on the task's message graph never shows a
held-out fact. Synthetic large-LP task with a known held-out set (validation and test pairs of a same-type relation,
stored in both orientations, plus a parallel same-type relation carrying some held-out pairs); every sampled subgraph
of the training and evaluation links, at full fan-out and depth 1-3, holds no held-out fact in either orientation.

Card mutation: sampling on the full graph instead of the message graph -> this test fails.
The real data root is unreachable.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def synthetic_task(seed: int = 0):
    from torch_geometric.data import HeteroData

    from src.data.large_lp import split_pairs, unordered_pairs

    g = torch.Generator().manual_seed(seed)
    n_user, n_club = 40, 8
    raw = torch.stack([torch.randint(0, n_user, (120,), generator=g), torch.randint(0, n_user, (120,), generator=g)])
    raw = raw[:, raw[0] != raw[1]]
    pairs, _ = unordered_pairs(raw)
    train, valid, test = split_pairs(pairs, seed=3, val_frac=0.15, test_frac=0.15)
    both = torch.from_numpy(np.concatenate([pairs, pairs[:, ::-1]]).T.copy())
    data = HeteroData()
    data["user"].x = torch.randn(n_user, 4, generator=g)
    data["club"].x = torch.randn(n_club, 3, generator=g)
    data["user", "has_friend", "user"].edge_index = both
    # a second same-type relation: some held-out pairs appear here too (removed "whatever its relation")
    extra = torch.from_numpy(np.concatenate([valid[:3], test[:3]]).T.copy())
    data["user", "messaged", "user"].edge_index = torch.cat([extra, extra.flip(0)], dim=1)
    uc = torch.stack([torch.randint(0, n_user, (50,), generator=g), torch.randint(0, n_club, (50,), generator=g)])
    data["user", "joins", "club"].edge_index = uc
    data["club", "rev_joins", "user"].edge_index = uc.flip(0)
    files = {"split_train": train, "split_valid": valid, "split_test": test}
    return data, files


class HeldOutFactTests(unittest.TestCase):
    def test_no_held_out_fact_in_any_sampled_subgraph(self) -> None:
        from src.baselines.large_lp_minibatch import (assert_no_held_out_facts, held_out_pairs,
                                                      link_loader, message_graph)

        data, files = synthetic_task()
        held = held_out_pairs(files)
        from src.data.large_lp import held_out_fact_mask

        relation = ("user", "has_friend", "user")
        self.assertGreater(int(held_out_fact_mask(data[relation].edge_index, held).sum()), 0)  # the task has some
        graph = message_graph(data, "user", held)
        links = {name: torch.from_numpy(files[f"split_{name}"].T.copy()) for name in ("train", "valid", "test")}
        checked = 0
        for depth in (1, 2, 3):
            for name, seeds in links.items():
                loader = link_loader(graph, relation, seeds, [-1] * depth, batch_size=4, shuffle=name == "train")
                for batch in loader:
                    assert_no_held_out_facts(batch, "user", held)
                    checked += 1
        self.assertGreater(checked, 30)
        # the message graph keeps every training fact and nothing else changes
        kept = graph[relation].edge_index
        self.assertEqual(kept.size(1), 2 * files["split_train"].shape[0])
        self.assertTrue(torch.equal(graph["user", "joins", "club"].edge_index, data["user", "joins", "club"].edge_index))
        self.assertEqual(graph["user", "messaged", "user"].edge_index.size(1), 0)


if __name__ == "__main__":
    unittest.main()
