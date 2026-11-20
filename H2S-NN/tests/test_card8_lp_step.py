"""Card 8 C2b: baseline small-LP training in batches of 8,192 positives per step, and DistMult as HGB.

* Shared loop (`training.train_link_prediction`) on a synthetic task with P = 20,000 training positives: one epoch
  takes ceil(P / 8,192) = 3 optimizer steps, one encode per step (plus the evaluation's).
* An adapter (GCN, `hgb_gnn`) on a tiny derived LP task end to end: steps per epoch = ceil(P / B) for a published
  B (`lp_batch_size`) and ceil(P / 8,192) otherwise. Every small-LP adapter trains through the shared loop with the
  budget's step size (no adapter has its own small-LP loop; static check).
* DistMult: a full `d x d` matrix per relation (shape [R, d, d], HGB's init), `score(e1, e2) = W[0, 1] != 0` and
  `score(u, v) != score(v, u)` for a non-symmetric W; equal to HGB's `bmm(bmm(u, W[r]), v)`.

Card mutations: one step per epoch -> `test_shared_loop_steps_per_epoch` fails; diagonal DistMult weights ->
`test_distmult_is_a_full_matrix` fails. The real data root is unreachable.
"""

from __future__ import annotations

import math
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def _synthetic_link_data(num_pos: int = 20_000) -> SimpleNamespace:
    g = torch.Generator().manual_seed(0)
    heads = torch.randint(0, 100, (num_pos,), generator=g)
    tails = torch.randint(100, 300, (num_pos,), generator=g)
    small = torch.stack([torch.arange(10), torch.arange(100, 110)])
    return SimpleNamespace(
        train_pos_edge_index=torch.stack([heads, tails]), train_pos_edge_relation=torch.zeros(num_pos, dtype=torch.long),
        neg_sampling_mode="fixed", run_seed=0, neg_sampling_ratio=1, dst_node_offset=100, num_dst_nodes=200,
        val_pos_edge_index=small, val_neg_edge_index=small.flip(0) % 300 + torch.tensor([[0], [0]]),
        val_pos_edge_relation=torch.zeros(10, dtype=torch.long), val_neg_edge_relation=torch.zeros(10, dtype=torch.long))


class SharedLoopStepTests(unittest.TestCase):
    def test_shared_loop_steps_per_epoch(self) -> None:
        from src.baselines.heads import SharedLinkHead
        from src.baselines.training import train_link_prediction

        data = _synthetic_link_data()
        torch.manual_seed(0)
        embedding = torch.nn.Parameter(torch.randn(300, 8))
        model = torch.nn.Module()
        model.register_parameter("emb", embedding)
        head = SharedLinkHead("dot", num_relations=1, dim=8)
        optimizer = torch.optim.SGD([embedding], lr=0.01)
        calls = {"embed": 0, "step": 0}
        original = optimizer.step

        def step(*a, **k):
            calls["step"] += 1
            return original(*a, **k)

        def embed():
            calls["embed"] += 1
            return embedding

        optimizer.step = step
        with mock.patch("src.utils.misc.lp_test_set_metrics", lambda score_pairs, d: {}):
            record = train_link_prediction(model=model, head=head, embed=embed, data=data, optimizer=optimizer,
                                           device=torch.device("cpu"), epochs=1, patience=1)
        self.assertEqual(calls["step"], math.ceil(20_000 / 8192))
        self.assertEqual(record["optimizer_steps_per_epoch"], 3.0)
        self.assertEqual(calls["embed"], 3 + 1)  # one encode per step, one for the evaluation

    def test_step_batches_cover_every_positive_once(self) -> None:
        from src.baselines.training import link_step_batches

        torch.manual_seed(0)
        batches = link_step_batches(20_000, 20_000, 8192, torch.device("cpu"))
        self.assertEqual([p.numel() for p, _ in batches], [8192, 8192, 3616])
        self.assertTrue(torch.equal(torch.sort(torch.cat([p for p, _ in batches])).values, torch.arange(20_000)))
        self.assertEqual(sum(n.numel() for _, n in batches), 20_000)


class AdapterStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from tests.test_partP_derived_lp import build_tiny_task, tiny_bundle

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        build_tiny_task(root, "acm_pa")
        cls.bundle = tiny_bundle(root, "acm_pa")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_gcn_adapter_steps_per_epoch(self) -> None:
        from src.baselines.protocol import LINK_PREDICTION, BaselineParams
        from src.baselines.registry import get_adapter

        positives = int(self.bundle.data.train_pos_edge_index.size(1))
        self.assertGreater(positives, 16)
        for values, source, batch in (({"epochs": 1, "early_stopping": 1, "lp_batch_size": 16}, "published", 16),
                                      ({"epochs": 1, "early_stopping": 1}, "tuned", 8192)):
            with self.subTest(source=source):
                record = get_adapter("gcn").fit_task(self.bundle, BaselineParams(values, source=source),
                                                     task=LINK_PREDICTION, seed=0, device=torch.device("cpu"))
                self.assertEqual(record["optimizer_steps_per_epoch"], float(math.ceil(positives / batch)))
                self.assertEqual(record["lp_positives_per_step"], float(batch))

    def test_every_small_lp_adapter_uses_the_shared_loop(self) -> None:
        """No adapter has its own small-LP epoch loop: each LP fit calls `train_link_prediction` with the budget's
        keyword arguments (which carry the step size); training negatives are drawn only in the shared loop."""
        adapters = PROJECT_ROOT / "src/baselines/adapters"
        calls = {}
        for path in sorted(adapters.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("sample_train_negatives(", text, path.name)
            self.assertNotIn("hgb_epoch_train_negatives(", text, path.name)
            for match in re.finditer(r"train_link_prediction\(", text):
                if text[max(0, match.start() - 4):match.start()] == "def ":
                    continue
                depth, end = 0, match.end() - 1
                for end in range(match.end() - 1, len(text)):  # the call's closing parenthesis
                    depth += {"(": 1, ")": -1}.get(text[end], 0)
                    if depth == 0:
                        break
                calls.setdefault(path.name, []).append(
                    "budget_kwargs(LINK_PREDICTION, params)" in text[match.start():end + 1])
        expected = {"_homogeneous_lp.py", "h2gformer.py", "h2sgnn.py", "hetero2net.py", "hetsheaf.py", "hgb_gnn.py",
                    "hgb_han.py", "pshgcn.py"}
        self.assertTrue(expected <= set(calls), sorted(calls))
        for name, flags in calls.items():
            self.assertTrue(all(flags), f"{name}: a train_link_prediction call without budget_kwargs(LINK_PREDICTION)")


class DistMultTests(unittest.TestCase):
    def test_distmult_is_a_full_matrix(self) -> None:
        from src.baselines.heads import SharedLinkHead

        head = SharedLinkHead("distmult", num_relations=2, dim=4)
        self.assertEqual(tuple(head.decoder.weights.shape), (2, 4, 4))
        with torch.no_grad():
            head.decoder.weights.zero_()
            head.decoder.weights[0, 0, 1] = 0.7
            head.decoder.weights[0, 1, 0] = -0.3
        emb = torch.eye(4)
        pairs = torch.tensor([[0, 1], [1, 0]])  # (e1 -> e2), (e2 -> e1)
        scores = head(emb, pairs, torch.zeros(2, dtype=torch.long))
        self.assertAlmostEqual(float(scores[0]), 0.7, places=6)
        self.assertAlmostEqual(float(scores[1]), -0.3, places=6)
        self.assertNotEqual(float(scores[0]), float(scores[1]))

    def test_distmult_equals_hgbs_bmm_and_init(self) -> None:
        from src.baselines.heads import SharedLinkHead

        torch.manual_seed(0)
        head = SharedLinkHead("distmult", num_relations=3, dim=6)
        u, v = torch.randn(20, 6), torch.randn(20, 6)
        rel = torch.randint(0, 3, (20,))
        emb = torch.cat([u, v])
        ours = head(emb, torch.stack([torch.arange(20), torch.arange(20, 40)]), rel)
        w = head.decoder.weights[rel]
        hgb = torch.bmm(torch.bmm(u.unsqueeze(1), w), v.unsqueeze(2)).squeeze()  # HGB GNN.py:16-20
        self.assertTrue(torch.allclose(ours, hgb, atol=1e-5))
        # HGB's init: xavier_normal_ with gain 1.414 on (R, d, d): std = 1.414 * sqrt(2 / (d*d + R*d))
        big = SharedLinkHead("distmult", num_relations=4, dim=64).decoder.weights
        expected = 1.414 * math.sqrt(2.0 / (64 * 64 + 4 * 64))
        self.assertAlmostEqual(float(big.std()), expected, delta=0.05 * expected)


if __name__ == "__main__":
    unittest.main()
