"""Card 21: loader workers, the equal-hop fan-out, and the two speed-only changes of Amendment 2. CPU, synthetic graphs.

1. `LOADER_WORKERS`: default 6; `H2SNN_LOADER_WORKERS` overrides it.
2. (Amendment 1.3) `equal_hop_fanout` pads every hop list to the longest length keeping its values; 20 batches of
   `hgt_loader` on the B1 graph (`user` 4 hops, `item` 3) sample without error with a fixed seed.
3. Same seed, 2 workers: two training loaders give identical first-batch node ids; a 2-worker evaluation loader
   (`_PassLoader`) gives the 0-worker seed order on two passes.
4. (C2) ChebNetII `ForwardBases`: the logits of an evaluation pass with the once-per-pass bases equal the old
   per-batch recomputation (atol 1e-6), also after a training step changes the parameters.
5. (C1) HINormer sequence cache: miss and hit give the uncached sequences, and the first `random.random()` after the
   call is the uncached call's.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def b1_graph(seed: int = 0):
    """`user` (target, 400) and `item` (300); user-user friendships, user-item likes, reverses stored."""
    from torch_geometric.data import HeteroData

    g = torch.Generator().manual_seed(seed)
    n_user, n_item = 400, 300
    data = HeteroData()
    data["user"].x = torch.randn(n_user, 5, generator=g)
    data["item"].x = torch.randn(n_item, 3, generator=g)
    data["user"].y = torch.randint(0, 3, (n_user,), generator=g)
    order = torch.randperm(n_user, generator=g)
    for name, part in (("train", order[:240]), ("val", order[240:320]), ("test", order[320:])):
        mask = torch.zeros(n_user, dtype=torch.bool)
        mask[part] = True
        data["user"][f"{name}_mask"] = mask
    uu = torch.unique(torch.randint(0, n_user, (2, 3200), generator=g), dim=1)
    ui = torch.unique(torch.stack([torch.randint(0, n_user, (2000,), generator=g),
                                   torch.randint(0, n_item, (2000,), generator=g)]), dim=1)
    data["user", "friend", "user"].edge_index = torch.cat([uu, uu.flip(0)], dim=1)
    data["user", "likes", "item"].edge_index = ui
    data["item", "rev_likes", "user"].edge_index = ui.flip(0)
    return data


#: H-Pokec's shape (`Pokec-*.yaml`): the target type 4 hops, the others 3
B1_FANOUT = {"user": [16, 16, 8, 8], "item": [4, 4, 4]}


class LoaderWorkersTests(unittest.TestCase):
    def _workers(self, value: str | None) -> int:
        env = {k: v for k, v in os.environ.items() if k != "H2SNN_LOADER_WORKERS"}
        if value is not None:
            env["H2SNN_LOADER_WORKERS"] = value
        out = subprocess.run([sys.executable, "-c", "from src.baselines.minibatch import LOADER_WORKERS; "
                              "print(LOADER_WORKERS)"], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        return int(out.stdout.strip().splitlines()[-1])

    def test_1_default_and_environment_override(self) -> None:
        self.assertEqual(self._workers(None), 6)
        self.assertEqual(self._workers("3"), 3)
        self.assertEqual(self._workers("0"), 0)


class EqualHopTests(unittest.TestCase):
    def test_2_padding_and_sampling(self) -> None:
        from src.baselines.minibatch import equal_hop_fanout, hgt_loader

        padded = equal_hop_fanout(B1_FANOUT)
        self.assertEqual({len(v) for v in padded.values()}, {4})
        for key, values in B1_FANOUT.items():
            self.assertEqual(padded[key][:len(values)], values)
        self.assertEqual(padded["item"], [4, 4, 4, 4])
        equal = {"a": [5, 5], "b": [3, 3]}
        self.assertIs(equal_hop_fanout(equal), equal)

        data = b1_graph()
        torch.manual_seed(0)
        loader = hgt_loader(data, B1_FANOUT, "user", "train", 8, shuffle=True, steps=20, num_workers=0)
        batches = list(loader)
        self.assertEqual(len(batches), 20)
        self.assertTrue(all(int(b["user"].batch_size) == 8 for b in batches))


class DeterminismTests(unittest.TestCase):
    def test_3_same_seed_same_batches_with_workers(self) -> None:
        from src.baselines.minibatch import eval_loader, hgt_loader

        data = b1_graph()

        def first_batch_ids() -> dict[str, torch.Tensor]:
            torch.manual_seed(7)
            loader = hgt_loader(data, B1_FANOUT, "user", "train", 16, shuffle=True, steps=5, num_workers=2)
            batch = next(iter(loader))
            return {t: batch[t].n_id.clone() for t in batch.node_types}

        first, second = first_batch_ids(), first_batch_ids()
        self.assertEqual(first.keys(), second.keys())
        for node_type in first:
            self.assertTrue(torch.equal(first[node_type], second[node_type]), node_type)

        def seed_order(loader) -> torch.Tensor:
            return torch.cat([b["user"].n_id[:int(b["user"].batch_size)] for b in loader])

        reference = seed_order(eval_loader(data, B1_FANOUT, "user", "val", 16, num_workers=0))
        with_workers = eval_loader(data, B1_FANOUT, "user", "val", 16, num_workers=2)
        for _ in range(2):  # a pass starts its workers and ends them; the next pass restarts
            self.assertTrue(torch.equal(seed_order(with_workers), reference))
        self.assertEqual(sorted(reference.tolist()), data["user"].val_mask.nonzero().view(-1).tolist())


def _old_forward(bases_module, features, rows):
    """`ForwardBases.forward` before card 21 (every call over the whole graph)."""
    t0 = bases_module.projection(features)
    t1 = torch.sparse.mm(bases_module.laplacian, t0)
    out = [t0[rows], t1[rows]]
    for _ in range(1, bases_module.K):
        t2 = 2 * torch.sparse.mm(bases_module.laplacian, t1) - t0
        t0, t1 = t1, t2
        out.append(t2[rows])
    return out


class ChebNetIIEvalBasesTests(unittest.TestCase):
    def test_4_eval_pass_equals_per_batch_recomputation(self) -> None:
        from src.baselines.adapters import _chebnetii_h2gb as h2gb
        from tests.test_baselines_h2gb_classb import typed_graph

        data = typed_graph()  # `field` featureless: a sparse identity block
        for node_type in ("paper", "author"):
            data[node_type].x = data[node_type].x.float()
        features, edge_index, slices = h2gb.forward_inputs(data)
        n = sum(int(f.size(0)) for f in features)
        start, _ = slices["paper"]
        torch.manual_seed(0)
        model = torch.nn.ModuleDict({"bases": h2gb.ForwardBases([int(f.size(1)) for f in features], 8, edge_index, n,
                                                                3),
                                     "chebnet": h2gb.build_model(8, 3, {"K": 3, "hidden": 16})})
        index = data["paper"].val_mask.nonzero().view(-1)

        def eval_pass(old: bool) -> torch.Tensor:
            model.eval()
            with torch.no_grad():
                out = []
                for batch in index.split(3):
                    rows = batch + start
                    bases = _old_forward(model["bases"], features, rows) if old else model["bases"](features, rows)
                    out.append(model["chebnet"](bases, 0, batch.numel()))
            return torch.cat(out)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)
        for step in range(2):
            new, old = eval_pass(False), eval_pass(True)
            self.assertIsNotNone(model["bases"]._eval_cache)
            torch.testing.assert_close(new, old, atol=1e-6, rtol=0)
            model.train()  # a training step changes the projection: the next pass must not reuse the cache
            idx = data["paper"].train_mask.nonzero().view(-1)
            loss = torch.nn.functional.nll_loss(model["chebnet"](model["bases"](features, idx + start), 0, idx.numel()),
                                                data["paper"].y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.assertIsNone(model["bases"]._eval_cache)


class HINormerSequenceCacheTests(unittest.TestCase):
    def test_5_cache_keeps_sequences_and_random_state(self) -> None:
        import dgl

        from src.baselines.adapters import hinormer

        g = torch.Generator().manual_seed(0)
        edges = torch.unique(torch.randint(0, 40, (2, 160), generator=g), dim=1)
        graph = dgl.graph((edges[0], edges[1]), num_nodes=40)
        nodes = torch.arange(5, 25)

        random.seed(3)
        expected = hinormer.build_node_sequences_for(graph, nodes, 6)
        expected_next = random.random()
        with tempfile.TemporaryDirectory() as tmp:
            for status in ("miss", "hit"):
                random.seed(3)
                node_seq, got = hinormer.cached_node_sequences(graph, nodes, 6, dataset="toy", cache_dir=tmp)
                self.assertEqual(got, status)
                self.assertTrue(torch.equal(node_seq, expected))
                self.assertEqual(node_seq.dtype, torch.long)
                self.assertEqual(random.random(), expected_next)
            random.seed(4)  # another state at the call: another key, computed afresh
            _, got = hinormer.cached_node_sequences(graph, nodes, 6, dataset="toy", cache_dir=tmp)
            self.assertEqual(got, "miss")


if __name__ == "__main__":
    unittest.main()
