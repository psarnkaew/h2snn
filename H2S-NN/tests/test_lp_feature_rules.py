"""J5.1 (task card Part J, spec v1.9 F12): the rule search of `scripts/lp/lp_feature_rules.py`
finds the known rules on toy releases and finds none where features are a node's own."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "lp"))

import lp_feature_rules as fr  # noqa: E402

N_PAPERS, N_AUTHORS, N_KEYWORDS, DIM = 40, 24, 6, 8
KEYWORD_NAMES = ["deep net", "graph", "net graph", "sheaf", "deep", "laplacian sheaf"]


def write_release(folder: Path, author_rule: str) -> None:
    """Papers (type 0, own random 0/1 features over DIM columns), authors (type 1), keywords
    (type 2, featureless, named). Relations: 0 paper->author (target), 1 author->paper,
    2 paper->keyword, 3 keyword->paper."""
    rng = np.random.default_rng(3)
    paper_x = (rng.random((N_PAPERS, DIM)) < 0.25).astype(int)
    paper_x[:, 0] = 1  # no all-zero paper
    pa = [(p, a) for p in range(N_PAPERS) for a in range(N_AUTHORS) if (p * 7 + a * 3) % 11 == 0]
    pk = [(p, k) for p in range(N_PAPERS) for k in range(N_KEYWORDS) if (p + 2 * k) % 4 == 0]
    authors = {a: [p for p, b in pa if b == a] for a in range(N_AUTHORS)}
    keywords = {p: [k for q, k in pk if q == p] for p in range(N_PAPERS)}
    vocab = sorted({t for name in KEYWORD_NAMES for t in name.split()})

    def author_feature(a: int) -> list[str]:
        ps = authors[a]
        if author_rule == "or":
            v = np.zeros(DIM, dtype=int)
            for p in ps:
                v |= paper_x[p]
            return [str(int(t)) for t in v]
        if author_rule == "or_permuted":
            v = np.zeros(DIM, dtype=int)
            for p in ps:
                v |= paper_x[p]
            return [str(int(t)) for t in v[[3, 1, 4]]]  # a permuted subset: 3 of 5 columns
        if author_rule == "mean_tokens":
            v = np.zeros(len(vocab))
            for p in ps:
                for k in keywords[p]:
                    for t in KEYWORD_NAMES[k].split():
                        v[vocab.index(t)] += 1.0
            if ps:
                v = v / len(ps)
            return [repr(float(t)) for t in v]
        if author_rule == "own":
            return [str(int(t)) for t in rng.random(DIM) < 0.5]
        raise ValueError(author_rule)

    folder.mkdir(parents=True)
    rows = []
    for p in range(N_PAPERS):
        rows.append(f"{p}\tp{p}\t0\t{','.join(str(int(t)) for t in paper_x[p])}\n")
    for a in range(N_AUTHORS):
        rows.append(f"{N_PAPERS + a}\ta{a}\t1\t{','.join(author_feature(a))}\n")
    for k in range(N_KEYWORDS):
        rows.append(f"{N_PAPERS + N_AUTHORS + k}\t{KEYWORD_NAMES[k].replace(' ', '_')}\t2\t\n")
    (folder / "node.dat").write_text("".join(rows), encoding="utf-8")
    links = [f"{p}\t{N_PAPERS + a}\t0\t1\n" for p, a in pa]
    links += [f"{N_PAPERS + a}\t{p}\t1\t1\n" for p, a in pa]
    links += [f"{p}\t{N_PAPERS + N_AUTHORS + k}\t2\t1\n" for p, k in pk]
    links += [f"{N_PAPERS + N_AUTHORS + k}\t{p}\t3\t1\n" for p, k in pk]
    (folder / "link.dat").write_text("".join(links), encoding="utf-8")
    info = {"node.dat": {"node type": {"0": "paper", "1": "author", "2": "keyword"}},
            "link.dat": {"link type": {
                "0": {"start": 0, "end": 1, "meaning": "paper-author"},
                "1": {"start": 1, "end": 0, "meaning": "author-paper"},
                "2": {"start": 0, "end": 2, "meaning": "paper-keyword"},
                "3": {"start": 2, "end": 0, "meaning": "keyword-paper"}}}}
    (folder / "info.dat").write_text(json.dumps(info), encoding="utf-8")


class FeatureRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def identify(self, rule: str, keyword_type: int | None = None) -> dict:
        folder = self.root / rule
        write_release(folder, rule)
        return fr.identify(fr.TaskSpec(rule, folder, folder, 0, keyword_type))

    def test_or_of_neighbour_features_is_found_exactly(self) -> None:
        author = self.identify("or")["types"]["1"]
        self.assertTrue(author["affected"])
        self.assertIn("V1/or", author["exact_rules"])

    def test_permuted_vocabulary_is_matched_by_column(self) -> None:
        author = self.identify("or_permuted")["types"]["1"]
        # dimensions differ (3 vs DIM), so the columns are matched (pre-registration J5.1)
        self.assertTrue(any(r.startswith("V1") for r in author["exact_rules"]), author)

    def test_mean_of_keyword_token_counts_is_found(self) -> None:
        author = self.identify("mean_tokens", keyword_type=2)["types"]["1"]
        self.assertIn("V3/mean", author["exact_rules"])

    def test_own_features_have_no_rule(self) -> None:
        rep = self.identify("own")
        self.assertEqual(rep["types"]["1"]["exact_rules"], [])
        self.assertEqual(rep["types"]["0"]["exact_rules"], [])  # papers' features are their own
        self.assertEqual(rep["types"]["2"]["status"], "no features")


if __name__ == "__main__":
    unittest.main()
