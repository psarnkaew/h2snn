"""Card 3 Part B (B1, Slim): SeHGNN, PSHGCN, H2SGNN on Freebase NC; PSHGCN / H2SGNN on H2GB NC and large LP
(decisions F-1, G-2, G-3; `src/baselines/adapters/_hgb_generic.py`). Synthetic data only.

* equivalence: on a synthetic ACM-shaped HGB release the generic construction gives exactly the dictionary their ACM
  branch builds (keys, order and values; ACM's `PP` merge incl.), for PSHGCN (`adjs`, `lis`, `lis_t`) and H2SGNN
  (`adjs`, meta-path names and operators);
* the relation plan: parallel relations merged, missing reverses completed, a no-op where reverses are stored;
* one step finite and moves per adapter path (Freebase-shaped release: 8 types, HGB's 36 relations, plus a parallel
  relation for the generic branch; H2GB: a sampled batch / target rows on the shared loops);
* large LP: PSHGCN's per-node precomputation reads the message graph only (held-out facts absent; negative control).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.partP_freebase_fixture import write_freebase_release


def _write_acm_shaped_release(root: Path) -> Path:
    """HGB format (`node.dat`, `link.dat`, `label.dat`, `label.dat.test`) with ACM's layout: types P (target, with
    attributes), A, C, K; links 0 P-P cite, 1 P-P ref (= cite transposed), 2 P-A, 3 A-P, 4 P-C, 5 C-P, 6 P-K, 7 K-P."""
    from src.data.data_layout import hgb_nc_raw_dir

    folder = hgb_nc_raw_dir(root, "acm")
    folder.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    counts = [20, 12, 4, 6]
    shift = [0, 20, 32, 36]
    nodes = []
    for t, n in enumerate(counts):
        for i in range(n):
            node = shift[t] + i
            attr = "\t" + ",".join(f"{v:.4f}" for v in torch.rand(5, generator=g).tolist()) if t == 0 else ""
            nodes.append(f"{node}\tn{node}\t{t}{attr}\n")
    (folder / "node.dat").write_text("".join(nodes), encoding="utf-8")

    def pairs(src: int, dst: int, k: int) -> list[tuple[int, int]]:
        out = set()
        while len(out) < k:
            u = shift[src] + int(torch.randint(counts[src], (1,), generator=g))
            v = shift[dst] + int(torch.randint(counts[dst], (1,), generator=g))
            if u != v:
                out.add((u, v))
        return sorted(out)

    cite, pa, pc, pk = pairs(0, 0, 30), pairs(0, 1, 30), pairs(0, 2, 15), pairs(0, 3, 25)
    links = []
    for r, edges in enumerate([cite, [(v, u) for u, v in cite], pa, [(v, u) for u, v in pa], pc,
                               [(v, u) for u, v in pc], pk, [(v, u) for u, v in pk]]):
        links += [f"{u}\t{v}\t{r}\t1.0\n" for u, v in edges]
    (folder / "link.dat").write_text("".join(links), encoding="utf-8")
    papers = list(range(20))
    (folder / "label.dat").write_text("".join(f"{p}\tn{p}\t0\t{p % 3}\n" for p in papers[:14]), encoding="utf-8")
    (folder / "label.dat.test").write_text("".join(f"{p}\tn{p}\t0\t{p % 3}\n" for p in papers[14:]),
                                           encoding="utf-8")
    return folder


def _dense_equal(case: unittest.TestCase, theirs: dict, ours: dict) -> None:
    case.assertEqual(list(theirs), list(ours))
    for key in theirs:
        case.assertTrue(torch.equal(theirs[key].to_dense(), ours[key].to_dense()), key)


def _add_parallel_relation(folder: Path) -> None:
    """A second BOOK -> FILM relation (link id 36) on the Freebase-shaped release, parallel to link 1 (the fixture's
    ids: books 0-39, films 40-51)."""
    with (folder / "link.dat").open("a", encoding="utf-8") as handle:
        handle.write("".join(f"{i}\t{40 + i % 12}\t36\t1.0\n" for i in range(5)))


def _step_moves(case: unittest.TestCase, model: torch.nn.Module, loss: torch.Tensor,
                optimizer: torch.optim.Optimizer) -> None:
    case.assertTrue(torch.isfinite(loss).item())
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    case.assertTrue(any(not torch.equal(b, a) for b, a in zip(before, model.parameters())), "no parameter moved")


class EquivalenceTests(unittest.TestCase):
    """The generic construction equals their ACM branch (card B1 equivalence)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.folder = _write_acm_shaped_release(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _generic(self, vendor: str) -> dict:
        from src.baselines.adapters._hgb_generic import hgb_generic_inputs, read_hgb_raw

        raw = read_hgb_raw(vendor, self.folder)
        return raw, hgb_generic_inputs(raw, {"0": "P", "1": "A", "2": "C", "3": "K"})

    def test_pshgcn_generic_branch_equals_their_acm_branch(self) -> None:
        from src.baselines.adapters._hgb_vendor import load_vendor_dataset
        from src.baselines.adapters._polynomial_lp import build_term_lists

        args = SimpleNamespace(dataset="ACM", device=torch.device("cpu"), K=3)
        _, adjs, lis, lis_t, labels, _, _, _, test_idx = load_vendor_dataset("pshgcn", "acm", args, self.root)
        raw, generic = self._generic("pshgcn")
        _dense_equal(self, adjs, generic["adjs"])
        ours_lis, ours_lis_t = build_term_lists(generic["adjs"], 3, enforce_type_chaining=True)
        self.assertEqual(lis, ours_lis)
        self.assertEqual(lis_t, ours_lis_t)
        self.assertTrue(torch.equal(labels, raw["labels"]))
        self.assertEqual(list(test_idx), list(raw["test_idx"]))

    def test_h2sgnn_generic_branch_equals_their_acm_branch(self) -> None:
        from src.baselines.adapters._hgb_vendor import load_vendor_dataset

        args = SimpleNamespace(dataset="ACM", device=torch.device("cpu"))
        out = load_vendor_dataset("h2sgnn", "acm", args, self.root)
        adjs, lis, adjs_list = out[1], out[2], out[9]
        _, generic = self._generic("h2sgnn")
        _dense_equal(self, adjs, generic["adjs"])
        self.assertEqual(lis, generic["metapath_names"])
        # Their ACM crop is the constant `target_max = 3024` (`H2SGNN/processing.py:87`, real ACM: 3,025 papers); ours
        # is `target_count - 1`, the same on real ACM. On this 20-paper release their crop keeps every row, so compare
        # on the target block (their extra entries are the non-target nodes' `set_diag` self-loops in PPPP).
        n = int((torch.tensor([int(line.split("\t")[2]) for line in
                               (self.folder / "node.dat").read_text().splitlines()]) == 0).sum())
        for theirs, ours in zip(adjs_list, generic["metapath_operators"], strict=True):
            self.assertEqual(tuple(ours.sizes()), (n, n))
            self.assertTrue(torch.equal(theirs.to_dense()[:n, :n], ours.to_dense()))


class RelationPlanTests(unittest.TestCase):
    def test_parallel_relations_merged_and_missing_reverses_completed(self) -> None:
        from src.baselines.adapters._hgb_generic import plan_relations

        e = lambda *pairs: torch.tensor(pairs, dtype=torch.long).T  # noqa: E731
        plan = plan_relations([("0", "1", e((0, 1), (2, 0))), ("0", "1", e((0, 1), (3, 2))),  # parallel 0 -> 1
                               ("0", "0", e((0, 1))), ("2", "0", e((1, 3)))])
        self.assertEqual(list(plan), [("0", "1"), ("0", "0"), ("2", "0"), ("1", "0"), ("0", "2")])
        self.assertEqual(sorted(map(tuple, plan[("0", "1")].T.tolist())), [(0, 1), (2, 0), (3, 2)])
        self.assertEqual(sorted(map(tuple, plan[("1", "0")].T.tolist())), [(0, 2), (1, 0), (2, 3)])
        self.assertEqual(sorted(map(tuple, plan[("0", "0")].T.tolist())), [(0, 1), (1, 0)])
        self.assertEqual(sorted(map(tuple, plan[("0", "2")].T.tolist())), [(3, 1)])

    def test_stored_reverses_leave_the_plan_unchanged(self) -> None:
        from src.baselines.adapters._hgb_generic import plan_relations

        forward = torch.tensor([[0, 1, 2], [1, 0, 0]])
        plan = plan_relations([("a", "b", forward), ("b", "a", torch.tensor([[5], [5]])),
                               ("a", "a", torch.tensor([[0, 1], [1, 0]]))])
        self.assertEqual(list(plan), [("a", "b"), ("b", "a"), ("a", "a")])
        self.assertEqual(plan[("b", "a")].tolist(), [[5], [5]])  # stored as is, even if not the transpose
        self.assertEqual(plan[("a", "a")].size(1), 2)

    def test_one_direction_toy_graph_gets_both_directions_and_a_symmetric_same_type_key(self) -> None:
        """2 relations, each stored one way (P -> A, P -> P): the generic branch's operators hold A -> P as the
        transpose of P -> A, and a symmetric P -> P with self-loops (author decision 2026-11-18; deviations.md D1)."""
        from src.baselines.adapters._hgb_generic import hgb_operators, plan_relations

        e = lambda *pairs: torch.tensor(pairs, dtype=torch.long).T  # noqa: E731
        plan = plan_relations([("p", "a", e((0, 0), (1, 2), (3, 1))), ("p", "p", e((0, 1), (2, 3), (3, 0)))])
        adjs = hgb_operators(plan, ["p", "a"], {"p": 4, "a": 3}, {"p": "P", "a": "A"}, "p")
        self.assertEqual(sorted(adjs), ["AP", "PA", "PP"])
        pattern = {k: (v.to_dense() != 0) for k, v in adjs.items()}
        self.assertTrue(torch.equal(pattern["AP"], pattern["PA"].T))
        self.assertTrue(pattern["AP"][4:, :4].any())
        self.assertTrue(torch.equal(pattern["PP"], pattern["PP"].T))
        self.assertTrue(pattern["PP"][:4, :4].diagonal().all())
        self.assertTrue(pattern["PP"][1, 0] and pattern["PP"][0, 3])  # stored (0,1), (3,0) and their transposes

    def test_h2sgnn_names_from_the_schema_equal_the_operators_names(self) -> None:
        from src.baselines.adapters._hgb_generic import h2sgnn_hetero_operators, h2sgnn_metapath_names
        from tests.test_baselines_minibatch import synthetic_hetero

        data = synthetic_hetero()
        del data["b", "rev_to", "a"]  # a missing reverse
        names, _ = h2sgnn_hetero_operators(data, "a")
        self.assertEqual(names, h2sgnn_metapath_names(list(data.node_types), list(data.edge_types), "a"))
        self.assertIn("ABBA", names)


class FreebaseStepTests(unittest.TestCase):
    """One step finite and moves per adapter on a Freebase-shaped release (8 types, HGB's 36 relations)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.folder = write_freebase_release(Path(cls._tmp.name) / "plain")
        cls.parallel = write_freebase_release(Path(cls._tmp.name) / "parallel")
        _add_parallel_relation(cls.parallel)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _raw(self, vendor: str):
        from src.baselines.adapters._hgb_generic import read_hgb_raw

        raw = read_hgb_raw(vendor, self.parallel)
        self.assertEqual(sum(1 for s, d, _ in raw["relations"] if (s, d) == ("0", "1")), 2)  # the parallel pair
        return raw

    def test_pshgcn_generic_branch_step(self) -> None:
        from src.baselines.adapters._hgb_generic import hgb_features, hgb_generic_inputs
        from src.baselines.adapters._polynomial_lp import build_term_lists
        from src.baselines.adapters.pshgcn import _vendor_args
        from src.baselines.protocol import BaselineParams
        from src.baselines.runtime import pushd_sys_path

        raw = self._raw("pshgcn")
        adjs = hgb_generic_inputs(raw)["adjs"]
        self.assertEqual(len(adjs), 64)  # 8 same-type + 28 cross-type pairs and their completed reverses
        args = _vendor_args("freebase", BaselineParams({"K": 2, "hidden": 8, "emb_dim": 8}), device=torch.device("cpu"))
        lis, lis_t = build_term_lists(adjs, args.K, enforce_type_chaining=True)
        features = hgb_features(raw)
        torch.manual_seed(0)
        with pushd_sys_path("pshgcn"):
            from model import PSHGCN  # type: ignore[import-not-found]

            model = PSHGCN([f.shape[1] for f in features], raw["num_classes"], lis, lis_t, args)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        train = torch.arange(10)
        loss = torch.nn.functional.cross_entropy(model(adjs, features)[train], raw["labels"][train])
        _step_moves(self, model, loss, optimizer)

    def test_h2sgnn_generic_branch_step(self) -> None:
        from src.baselines.adapters._hgb_generic import hgb_features, hgb_generic_inputs
        from src.baselines.adapters.h2sgnn import _optimizer, _vendor_args, _vendor_model
        from src.baselines.protocol import BaselineParams

        raw = self._raw("h2sgnn")
        generic = hgb_generic_inputs(raw)
        self.assertEqual(len(generic["metapath_names"]), 8)  # BOOK-X-BOOK over the 8 types (reverses completed)
        args = _vendor_args("freebase", BaselineParams({"K": 2, "hidden": 8, "emb_dim": 8}), device=torch.device("cpu"))
        features = hgb_features(raw)
        torch.manual_seed(0)
        model = _vendor_model(int(features[0].shape[1]), raw["num_classes"], generic["metapath_names"], args)
        ops = generic["metapath_operators"]
        train = torch.arange(10)
        loss = torch.nn.functional.cross_entropy(model(ops, features, ops)[train], raw["labels"][train])
        _step_moves(self, model, loss, _optimizer(model, args))

    def test_sehgnn_freebase_branch_step(self) -> None:
        from src.baselines.adapters.sehgnn import VENDORS, _freebase_inputs, _vendor_args
        from src.baselines.protocol import BaselineParams
        from src.baselines.runtime import pushd_sys_path

        params = BaselineParams({"num_hops": 2, "num_label_hops": 3, "hidden": 8, "embed_size": 8,
                                 "n_task_layers": 2, "residual": True})
        args = _vendor_args("freebase", params, self.folder.parents[3])
        num_target = 40
        labels = torch.arange(num_target) % 7
        onehot = torch.zeros(num_target, 7)
        onehot[:10] = torch.nn.functional.one_hot(labels[:10], 7).float()
        feats, label_feats, data_size = _freebase_inputs(self.folder, args, onehot, torch.device("cpu"))
        self.assertIn("0", feats)
        self.assertTrue(set(label_feats) <= {"00", "000", "0000", "0010", "0030", "0040", "0050", "0060", "0070"})
        torch.manual_seed(0)
        with pushd_sys_path(*VENDORS):
            from model import SeHGNN  # type: ignore[import-not-found]

            model = SeHGNN("Freebase", 8, 8, 7, feats.keys(), label_feats.keys(), "0", 0.0, 0.0, 0.0, 2, 2, "none",
                           True, data_size=data_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        batch = torch.arange(10)
        out = model(batch, {k: v[batch] for k, v in feats.items()}, {k: v[batch] for k, v in label_feats.items()})
        _step_moves(self, model, torch.nn.functional.cross_entropy(out, labels[batch]), optimizer)


class H2GBStepTests(unittest.TestCase):
    """H2GB NC on the shared loops' units: PSHGCN target rows of the precomputed inputs, H2SGNN a sampled batch."""

    def test_pshgcn_decoupled_rows_step_and_batching_is_exact(self) -> None:
        from src.baselines.adapters._hgb_generic import build_decoupled_model, decoupled_forward, decoupled_inputs
        from src.baselines.protocol import BaselineParams
        from tests.test_baselines_minibatch import synthetic_hetero

        data = synthetic_hetero()
        data["c"].num_nodes = 7  # a featureless type: their random-uniform inputs
        data["a", "has", "c"].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 6]])
        inputs = decoupled_inputs(data, "a", seed=0)
        self.assertIn("P", inputs["feats"])
        torch.manual_seed(0)
        model = build_decoupled_model(inputs, 3, BaselineParams({"emb_dim": 8, "hidden_x": 8, "layers_x": 2,
                                                                 "dropout": 0.0, "input_drop": 0.0}))
        feats = inputs["feats"]
        model.eval()
        full = decoupled_forward(model, dict(feats))
        rows = torch.tensor([3, 0, 7])
        self.assertTrue(torch.allclose(decoupled_forward(model, {k: v[rows] for k, v in feats.items()}), full[rows],
                                       atol=1e-6))
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        loss = torch.nn.functional.cross_entropy(decoupled_forward(model, {k: v[rows] for k, v in feats.items()}),
                                                 data["a"].y[rows])
        _step_moves(self, model, loss, optimizer)

    def test_h2sgnn_sampled_batch_step(self) -> None:
        from src.baselines.adapters._hgb_generic import h2sgnn_metapath_names
        from src.baselines.adapters.h2sgnn import _optimizer, _vendor_args, _vendor_model, batch_forward
        from src.baselines.protocol import BaselineParams
        from tests.test_baselines_h2gb_classa import _first_batch
        from tests.test_baselines_minibatch import synthetic_hetero

        data = synthetic_hetero()
        names = h2sgnn_metapath_names(list(data.node_types), list(data.edge_types), "a")
        batch = _first_batch(data)
        size = int(batch["a"].batch_size)
        args = _vendor_args("oag-cs", BaselineParams({"K": 2, "hidden": 8, "emb_dim": 8}), device=torch.device("cpu"))
        torch.manual_seed(0)
        model = _vendor_model(5, 3, names, args)
        logits = batch_forward(model, batch, "a", names, torch.device("cpu"))[:size]
        self.assertEqual(tuple(logits.shape), (size, 3))
        loss = torch.nn.functional.cross_entropy(logits, batch["a"].y[:size])
        _step_moves(self, model, loss, _optimizer(model, args))


class LargeLPHeldOutTests(unittest.TestCase):
    def test_pshgcn_precomputation_reads_the_message_graph_only(self) -> None:
        """Every user-user edge PSHGCN propagates over is a message-graph edge: no held-out fact in either
        orientation; built on the full graph instead (negative control), held-out facts appear."""
        from src.baselines.adapters._hgb_generic import decoupled_inputs
        from src.baselines.large_lp_minibatch import held_out_pairs, message_graph
        from src.data.large_lp import held_out_fact_mask
        from tests.test_baselines_large_lp import synthetic_task

        data, files = synthetic_task()
        held = held_out_pairs(files)
        for graph, expect_clean in ((message_graph(data, "user", held), True), (data, False)):
            plan = decoupled_inputs(graph, "user", seed=0)["plan"]
            facts = sum(int(held_out_fact_mask(edges, held).sum()) for (s, d), edges in plan.items()
                        if s == d == "user")
            if expect_clean:
                self.assertEqual(facts, 0)
            else:
                self.assertGreater(facts, 0)


if __name__ == "__main__":
    unittest.main()
