"""Part P, P2 (`docs/H2S-NN_partP_card.md`): the derived LP tasks for every LP baseline.

On tiny derived-format releases built by the project's own pipeline (`tests.test_lp_derived`
release writer, `build_derived_lp.build_task`, `apply_feature_rule.apply_rule`), loaded by name
through `build_bundle`:
- every LP adapter's `fit_task` runs one epoch and returns both test sets' keys (an adapter whose
  vendor clone is empty, or that needs a C++ compiler absent here, is skipped with that reason);
- no held-out pair (validation or test) reaches the message graph of any adapter, in either
  orientation: the bundle's graph, HGB's homogeneous view, the typed edge dicts and the polynomial
  relation operators;
- PSHGCN/H2SGNN relation keys on the tiny release equal a hand list (stored reverses dropped);
- a task whose head type has no 2-hop path to the tail type (DBLP-shaped author->paper) yields
  `None` for the 2-hop metrics with the reason, never 1 or NaN.
Card mutation: the reverse copy of a test fact kept in the message graph -> a test fails.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from scripts.lp import build_derived_lp as bd
from scripts.lp.apply_feature_rule import apply_rule
from src.baselines.protocol import LINK_PREDICTION, BaselineParams
from src.data.data_layout import lp_derived_root
from tests.test_lp_derived import write_release

LP_ADAPTERS = ("gcn", "gat", "nsd", "csnn", "han", "pshgcn", "h2gformer", "hetero2net", "h2sgnn")
#: The vendor each adapter's LP fit imports (src/baselines/runtime/vendor_paths.py).
NEEDS_VENDOR = {"nsd": "nsd", "pshgcn": "pshgcn_lp", "h2sgnn": "h2sgnn"}


def environment_blocker(name: str) -> str | None:
    """Why `name` cannot run on this machine (an empty vendor clone, no C++ compiler), or None."""
    from src.baselines.runtime.vendor_paths import VENDOR_ROOTS

    key = NEEDS_VENDOR.get(name)
    if key is not None:
        root = VENDOR_ROOTS[key]
        if not root.is_dir() or not any(root.rglob("*.py")):
            return f"vendor clone {root} is empty on this machine"
    if name == "csnn" and sys.platform == "win32" and shutil.which("cl") is None:
        return "CSNN's torch_householder JIT-compiles with MSVC; no `cl` on this machine"
    return None


def write_dblp_like(folder: Path, seed: int = 0) -> None:
    """HGB-DBLP-shaped release: 0 author, 1 paper, 2 term, 3 venue; relation 0 author->paper
    (target) with its stored reverse 3, paper->term 1 / 4, paper->venue 2 / 5."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    authors, papers, terms, venues = 30, 40, 5, 3
    a0, p0, t0, v0 = 0, 30, 70, 75
    nodes = [f"{a0 + i}\tauthor{i}\t0\t0.1,0.{i % 10}\n" for i in range(authors)]
    nodes += [f"{p0 + i}\tpaper{i}\t1\t0.2,0.{i % 10}\n" for i in range(papers)]
    nodes += [f"{t0 + i}\tterm{i}\t2\t0.3,0.4\n" for i in range(terms)]
    nodes += [f"{v0 + i}\tvenue{i}\t3\n" for i in range(venues)]
    (folder / "node.dat").write_bytes("".join(nodes).encode())
    writes = rng.sample([(a, p) for a in range(authors) for p in range(papers)], 120)
    links = [f"{a0 + a}\t{p0 + p}\t0\t1.0\n" for a, p in writes]
    links += [f"{p0 + p}\t{t0 + p % terms}\t1\t1.0\n" for p in range(papers)]
    links += [f"{p0 + p}\t{v0 + p % venues}\t2\t1.0\n" for p in range(papers)]
    links += [f"{p0 + p}\t{a0 + a}\t3\t1.0\n" for a, p in writes]
    links += [f"{t0 + p % terms}\t{p0 + p}\t4\t1.0\n" for p in range(papers)]
    links += [f"{v0 + p % venues}\t{p0 + p}\t5\t1.0\n" for p in range(papers)]
    (folder / "link.dat").write_bytes("".join(links).encode())
    (folder / "info.dat").write_text(json.dumps({"synthetic": True}), encoding="utf-8")


def build_tiny_task(root: Path, task: str) -> None:
    """Build `task` from a synthetic release with the project's derived-task pipeline."""
    if task == "acm_pa":
        write_release(root / "release_acm")
        bd.build_task(root / "release_acm", lp_derived_root(root) / task, 2, seed=42,
                      val_frac=0.09, test_frac=0.10)
    elif task == "dblp_ap":
        write_dblp_like(root / "release_dblp")
        bd.build_task(root / "release_dblp", lp_derived_root(root) / task, 0, seed=42,
                      val_frac=0.09, test_frac=0.10)
    else:
        raise KeyError(task)
    apply_rule(task, lp_derived_root(root) / task)


def tiny_bundle(root: Path, task: str):
    from src.baselines.bundles import build_bundle

    return build_bundle(LINK_PREDICTION, task, overrides={"device": "cpu",
                                                          "data": {"root": str(root)}})


class _TinyTask(unittest.TestCase):
    TASK = "acm_pa"

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        build_tiny_task(cls.root, cls.TASK)
        cls.bundle = tiny_bundle(cls.root, cls.TASK)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()


def _held_out(bundle) -> set[tuple[int, int]]:
    data = bundle.data
    pairs = set()
    for name in ("val_pos_edge_index", "test_pos_edge_index"):
        pairs |= {tuple(p) for p in getattr(data, name).t().tolist()}
    return pairs | {(v, u) for u, v in pairs}


class FitTests(_TinyTask):
    def test_every_lp_adapter_runs_one_epoch(self) -> None:
        from src.baselines.registry import get_adapter, unsupported_reason

        for name in LP_ADAPTERS:
            with self.subTest(adapter=name):
                self.assertIsNone(unsupported_reason(name, LINK_PREDICTION, self.TASK))
                adapter = get_adapter(name)
                self.assertTrue(adapter.supports(LINK_PREDICTION, self.TASK))
                blocker = environment_blocker(name)
                if blocker is not None:
                    self.skipTest(blocker)
                record = adapter.fit_task(self.bundle, BaselineParams(
                    {"epochs": 1, "early_stopping": 1}), task=LINK_PREDICTION, seed=0,
                    device=torch.device("cpu"))
                for prefix in ("test_2hop_", "test_random_"):
                    self.assertIn(prefix + "auroc", record)
                    self.assertIn(prefix + "aupr", record)
                for key in ("test_random_auroc", "test_random_aupr", "test_2hop_auroc"):
                    self.assertTrue(math.isfinite(float(record[key])), key)


class MessageGraphTests(_TinyTask):
    def test_no_held_out_pair_reaches_any_message_graph(self) -> None:
        from src.baselines.adapters._polynomial_lp import build_relation_adjacencies
        from src.baselines.adapters._target_block import build_edge_index_dict

        held = _held_out(self.bundle)
        self.assertGreater(len(held), 0)
        data = self.bundle.data
        graphs = {"bundle": {tuple(p) for p in data.edge_index.t().tolist()}}
        view = self.bundle.homogeneous(feats_type=0)
        graphs["homogeneous view"] = {tuple(p) for p in view.edge_index.t().tolist()}
        typed = set()
        slices = dict(data.node_slices)
        for (source, _, destination), index in build_edge_index_dict(
                self.bundle, torch.device("cpu")).items():
            typed |= {(u + slices[source][0], v + slices[destination][0])
                      for u, v in index.t().tolist()}
        graphs["typed edge dict"] = typed
        adjacencies, _ = build_relation_adjacencies(self.bundle, torch.device("cpu"))
        polynomial = set()
        for adjacency in adjacencies.values():
            row, col, _ = adjacency.coo()
            polynomial |= set(zip(row.tolist(), col.tolist()))
        graphs["polynomial operators"] = polynomial
        for name, edges in graphs.items():
            with self.subTest(graph=name):
                self.assertEqual(edges & held, set())


class RelationKeyTests(_TinyTask):
    def test_polynomial_keys_equal_the_hand_list(self) -> None:
        from src.baselines.adapters._polynomial_lp import (
            _plan_relation_keys,
            build_relation_adjacencies,
        )

        data = self.bundle.data
        self.assertEqual(data.node_type_names[:3], ("paper", "author", "subject"))
        keys, self_loops, chaining = _plan_relation_keys(self.bundle)
        # by hand: one same-type key for cite/ref (symmetrised, self-loops), forward and
        # transposed keys for paper-author and paper-subject; the stored reverse of each pair
        # adds no key (whichever of the two the bundle lists first is kept)
        self.assertTrue(chaining)
        self.assertEqual(set(keys), {"AA", "AB", "BA", "AC", "CA"})
        self.assertEqual(self_loops, {"AA": True, "AB": False, "BA": False, "AC": False,
                                      "CA": False})
        relations = list(data.relation_tuples)

        def edges(relation):
            mask = data.edge_type == relations.index(relation)
            return {tuple(p) for p in data.edge_index[:, mask].t().tolist()}

        writes, subject = edges(("paper", "rel_2", "author")), edges(("paper", "rel_4", "subject"))
        cite = edges(("paper", "rel_0", "paper"))
        self.assertEqual(edges(("author", "rel_3", "paper")), {(v, u) for u, v in writes})
        adjacencies, _ = build_relation_adjacencies(self.bundle, torch.device("cpu"))

        def pattern(key):
            row, col, _ = adjacencies[key].coo()
            return set(zip(row.tolist(), col.tolist()))

        self.assertEqual(pattern("AB"), writes)
        self.assertEqual(pattern("BA"), {(v, u) for u, v in writes})
        self.assertEqual(pattern("AC"), subject)
        self.assertEqual(pattern("CA"), {(v, u) for u, v in subject})
        # set_diag() over the whole flattened matrix (their processing.py; unchanged code)
        everyone = range(int(data.x.size(0)))
        self.assertEqual(pattern("AA"), cite | {(v, u) for u, v in cite} | {(n, n) for n in everyone})

    def test_parallel_relations_still_raise(self) -> None:
        from types import SimpleNamespace

        from src.baselines.adapters._polynomial_lp import _plan_relation_keys

        data = SimpleNamespace(
            node_type_names=("a", "b"),
            relation_tuples=(("a", "r0", "b"), ("a", "r1", "b")),
            edge_index=torch.tensor([[0, 0], [2, 3]]), edge_type=torch.tensor([0, 1]))
        with self.assertRaises(NotImplementedError):
            _plan_relation_keys(SimpleNamespace(data=data, dataset="toy"))


class EmptyTwoHopTests(_TinyTask):
    TASK = "dblp_ap"

    def test_empty_two_hop_set_yields_none_with_a_reason(self) -> None:
        from src.baselines.registry import get_adapter

        self.assertEqual(self.bundle.data.test_neg_edge_index.size(1), 0)  # HGB's 2-hop set
        self.assertTrue(self.bundle.data.test_2hop_empty_reason)
        record = get_adapter("gcn").fit_task(self.bundle, BaselineParams(
            {"epochs": 1, "early_stopping": 1}), task=LINK_PREDICTION, seed=0,
            device=torch.device("cpu"))
        for key in ("test_2hop_auroc", "test_2hop_aupr"):
            self.assertIsNone(record[key], key)
        self.assertEqual(record["test_2hop_reason"], self.bundle.data.test_2hop_empty_reason)
        self.assertTrue(math.isfinite(float(record["test_random_auroc"])))


if __name__ == "__main__":
    unittest.main()
