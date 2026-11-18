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


SAMPLERS = {
    "link, H2GB per-hop list (full)": ("link", [-1, -1, -1]),
    "link, own fan-out [10, 10] (Hetero2Net)": ("link", [10, 10]),
    "hgt, per-type lists seeded by endpoints (G-8)": ("hgt", {"user": [10 ** 6] * 3, "club": [10 ** 6] * 3}),
}


def _lp_data(graph):
    from src.baselines.large_lp_training import LargeLPData

    e = torch.empty(2, 0, dtype=torch.long)
    return LargeLPData("h-pokec_uu", graph, "user", ("user", "has_friend", "user"), e, e, e, e, e, e)


class WiredLoaderTests(unittest.TestCase):
    """Card 1c F7: every sampler configuration of the wired large-LP adapters (training step and evaluation endpoints)
    on the message graph holds no held-out fact. The card mutation (full graph instead of the message graph) fails it."""

    def test_no_held_out_fact_in_any_sampler_configuration(self) -> None:
        from src.baselines.large_lp_minibatch import assert_no_held_out_facts, held_out_pairs, message_graph
        from src.baselines.large_lp_training import Sampler

        data, files = synthetic_task()
        held = held_out_pairs(files)
        graph = message_graph(data, "user", held)
        relation = ("user", "has_friend", "user")
        train = torch.from_numpy(files["split_train"].T.copy())
        for label, (mode, fanout) in SAMPLERS.items():
            with self.subTest(sampler=label):
                sampler = Sampler(mode, fanout)
                batch, positions = sampler.step(graph, relation, "user", train[:, :12])
                assert_no_held_out_facts(batch, "user", held)
                self.assertEqual(tuple(positions.shape), (2, 12))
                for split in ("valid", "test"):
                    nodes = torch.unique(torch.from_numpy(files[f"split_{split}"]).reshape(-1))
                    for batch in sampler.endpoints(graph, "user", nodes, batch_size=5):
                        assert_no_held_out_facts(batch, "user", held)

    def test_each_wired_method_one_lp_step_is_finite_and_moves(self) -> None:
        import importlib

        from src.baselines.heads import SharedLinkHead
        from src.baselines.large_lp_minibatch import held_out_pairs, message_graph
        from src.baselines.large_lp_training import Sampler, step_scores
        from src.baselines.registry import LARGE_LP_WIRED_CELLS, get_adapter, model_spec

        data, files = synthetic_task()
        pos = torch.from_numpy(files["split_train"].T.copy())[:, :8]
        neg = torch.stack([pos[0], torch.randint(0, 40, (8,), generator=torch.Generator().manual_seed(1))])
        links = torch.cat([pos, neg], dim=1)
        names = sorted({name for name, _ in LARGE_LP_WIRED_CELLS})
        self.assertEqual(names, ["csnn", "gat", "gcn", "h2gformer", "h2sgnn", "han", "hetero2net", "hetsheaf", "nsd",
                                 "pshgcn"])
        params = type("P", (dict,), {"report": None, "source": "test"})(
            hidden_dim=8, num_layers=2, num_heads=2, dropout=0.0, embed_dim=8, hidden_channels=8, d=2, layers=2,
            gt_layers=2, attn_heads=2)
        for name in names:
            with self.subTest(method=name):
                graph = message_graph(data, "user", held_out_pairs(files))
                mode, fanout = {"han": SAMPLERS["hgt, per-type lists seeded by endpoints (G-8)"],
                                "h2gformer": SAMPLERS["hgt, per-type lists seeded by endpoints (G-8)"],
                                "hetero2net": SAMPLERS["link, own fan-out [10, 10] (Hetero2Net)"]}.get(
                                    name, ("link", [-1, -1]))
                torch.manual_seed(0)
                module = importlib.import_module(model_spec(name).module)
                builder = getattr(module, "build_large_lp", None) or getattr(get_adapter(name), "build_large_lp")
                model, sampler = builder(_lp_data(graph), params, torch.device("cpu"), Sampler(mode, fanout))
                head = SharedLinkHead("dot", num_relations=1, dim=model.embed_dim)
                scores = step_scores(model, head, sampler, graph, ("user", "has_friend", "user"), "user", links,
                                     torch.device("cpu"))
                from torch.nn.parameter import UninitializedParameter

                before = {k: v.detach().clone() for k, v in model.modules.named_parameters()
                          if not isinstance(v, UninitializedParameter)}
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    scores, torch.cat([torch.ones(8), torch.zeros(8)]))
                if model.aux_loss is not None:
                    loss = loss + model.aux_loss()
                self.assertTrue(torch.isfinite(loss).item())
                optimizer = torch.optim.AdamW(list(model.modules.parameters()) + list(head.parameters()), lr=1e-2)
                loss.backward()
                optimizer.step()
                after = dict(model.modules.named_parameters())
                self.assertTrue(any(not torch.equal(v, after[k]) for k, v in before.items()), "no parameter moved")


if __name__ == "__main__":
    unittest.main()
