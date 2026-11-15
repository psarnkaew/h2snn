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


def _wired_adapters() -> list[str]:
    """Adapters with a large-LP code path (part 2 E4: GCN, GAT; their registry cells are a stop item, see the report)."""
    from src.baselines.registry import MODELS, get_adapter

    return sorted(spec.name for spec in MODELS
                  if getattr(__import__(spec.module, fromlist=["x"]), "LARGE_LP_DATASETS", None)
                  and hasattr(get_adapter(spec.name), "build_lp_encoder"))


class WiredAdapterLoaderTests(unittest.TestCase):
    """Part 2 E4: every adapter with a large-LP path samples through `large_lp_training` (training step: links as seeds of
    the link sampler; evaluation: each endpoint once by node-level sampling), both on the message graph; no sampled
    subgraph holds a held-out fact. The card mutation (full graph instead of the message graph) fails this test too."""

    def test_no_held_out_fact_in_wired_adapters_loaders(self) -> None:
        from src.baselines import large_lp_training as llp
        from src.baselines.large_lp_minibatch import assert_no_held_out_facts, held_out_pairs, message_graph

        names = _wired_adapters()
        self.assertTrue(names)
        data, files = synthetic_task()
        held = held_out_pairs(files)
        graph = message_graph(data, "user", held)
        relation = ("user", "has_friend", "user")
        for name in names:
            with self.subTest(adapter=name):
                train = torch.from_numpy(files["split_train"].T.copy())
                for depth in (1, 2, 3):
                    batch = llp.step_subgraph(graph, relation, train[:, :12], [-1] * depth)
                    assert_no_held_out_facts(batch, "user", held)
                    for split in ("valid", "test"):
                        nodes = torch.unique(torch.from_numpy(files[f"split_{split}"]).reshape(-1))
                        for batch in llp.endpoint_loader(graph, "user", nodes, [-1] * depth, batch_size=5):
                            assert_no_held_out_facts(batch, "user", held)

    def test_wired_adapters_one_lp_step_is_finite_and_moves(self) -> None:
        from src.baselines import large_lp_training as llp
        from src.baselines.heads import SharedLinkHead
        from src.baselines.large_lp_minibatch import held_out_pairs, message_graph
        from src.baselines.minibatch import batch_type_inputs, h2gb_input_dims, hgb_homogeneous_batch
        from src.baselines.registry import get_adapter

        data, files = synthetic_task()
        graph = message_graph(data, "user", held_out_pairs(files))
        relation = ("user", "has_friend", "user")
        node_types, counts, dims = h2gb_input_dims(graph)
        pos = torch.from_numpy(files["split_train"].T.copy())[:, :8]
        neg = torch.stack([pos[0], torch.randint(0, 40, (8,), generator=torch.Generator().manual_seed(1))])
        links = torch.cat([pos, neg], dim=1)
        for name in _wired_adapters():
            with self.subTest(adapter=name):
                torch.manual_seed(0)
                params = type("P", (dict,), {"report": None, "source": "test"})(
                    hidden_dim=8, num_layers=2, num_heads=2, dropout=0.0)
                encoder = get_adapter(name).build_lp_encoder(dims, params)
                head = SharedLinkHead("dot", num_relations=1, dim=8)
                module = torch.nn.ModuleList([encoder, head])
                before = [p.detach().clone() for p in module.parameters()]
                batch = llp.step_subgraph(graph, relation, links, [-1, -1])
                inputs = batch_type_inputs(batch, node_types, counts)
                _, edge_index, offsets = hgb_homogeneous_batch(batch, node_types)
                optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2)
                out = encoder.encode((inputs, edge_index))
                scores = head(out, batch[relation].edge_label_index + offsets["user"][0])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    scores, torch.cat([torch.ones(8), torch.zeros(8)]))
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                optimizer.step()
                self.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, module.parameters())))


if __name__ == "__main__":
    unittest.main()
