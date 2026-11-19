"""Toy HGB releases in HGB's own file format (card 7 T3/T4): `write_toy_hgb(root, "dblp" | "acm")` writes
`<root>/NC/<name>/raw/<NAME>/{info,node,link,label}.dat, label.dat.test` shaped like the real release (same node
and link type order, same feature-less types, every target labelled), small enough for a unit test. Targets and
their neighbours are drawn by community so the toy starts homophilous (its h lies between the levels 0.2 and 0.6).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPECS = {
    "dblp": {
        "name": "DBLP",
        "types": ["author", "paper", "term", "venue"],
        "featureless": {"venue"},
        "links": [(0, 1, "author-paper"), (1, 2, "paper-term"), (1, 3, "paper-venue"), (1, 0, "paper-author"),
                  (2, 1, "term-paper"), (3, 1, "venue-paper")],
        "mirror": {0: 3, 1: 4, 2: 5},
        "classes": ["Database", "Data Mining", "AI", "Information Retrieval"],
    },
    "acm": {
        "name": "ACM",
        "types": ["paper", "author", "subject", "term"],
        "featureless": {"term"},
        "links": [(0, 0, "paper-cite-paper"), (0, 0, "paper-ref-paper"), (0, 1, "paper-author"),
                  (1, 0, "author-paper"), (0, 2, "paper-subject"), (2, 0, "subject-paper"), (0, 3, "paper-term"),
                  (3, 0, "term-paper")],
        "mirror": {0: 1, 2: 3, 4: 5, 6: 7},
        "classes": ["0", "1", "2"],
    },
}


def _toy_edges(dataset: str, rng: np.random.Generator) -> tuple[dict[str, int], dict[int, list[tuple[int, int]]],
                                                                 np.ndarray]:
    """(node counts per type, local edges per forward link id, target classes)."""
    if dataset == "dblp":
        counts = {"author": 120, "paper": 200, "term": 30, "venue": 6}
        classes = rng.integers(4, size=counts["author"])
        classes[:4] = np.arange(4)
        paper_comm = rng.integers(4, size=counts["paper"])
        author_paper, paper_term, paper_venue = set(), set(), set()
        for p in range(counts["paper"]):
            for _ in range(int(rng.integers(1, 4))):
                pool = np.flatnonzero(classes == paper_comm[p]) if rng.random() < 0.7 else np.arange(counts["author"])
                author_paper.add((int(rng.choice(pool)), p))
            for _ in range(2):
                paper_term.add((p, int(rng.integers(counts["term"]))))
            paper_venue.add((p, int(paper_comm[p] + 4 * rng.integers(0, 2)) % counts["venue"]))
        return counts, {0: sorted(author_paper), 1: sorted(paper_term), 2: sorted(paper_venue)}, classes
    counts = {"paper": 150, "author": 60, "subject": 5, "term": 20}
    classes = rng.integers(3, size=counts["paper"])
    classes[:3] = np.arange(3)
    author_comm = rng.integers(3, size=counts["author"])
    cite, writes, subject, term = set(), set(), set(), set()
    for p in range(counts["paper"]):
        for _ in range(2):
            pool = np.flatnonzero(classes == classes[p]) if rng.random() < 0.8 else np.arange(counts["paper"])
            q = int(rng.choice(pool))
            if q != p:
                cite.add((p, q))
        for _ in range(2):
            pool = np.flatnonzero(author_comm == classes[p]) if rng.random() < 0.8 else np.arange(counts["author"])
            writes.add((p, int(rng.choice(pool))))
        subject.add((p, int(classes[p]) if rng.random() < 0.7 else int(rng.integers(counts["subject"]))))
        for _ in range(3):
            term.add((p, int(rng.integers(counts["term"]))))
    return counts, {0: sorted(cite), 2: sorted(writes), 4: sorted(subject), 6: sorted(term)}, classes


def write_toy_hgb(root: Path, dataset: str, seed: int = 0) -> Path:
    """Write the toy release; returns its raw folder."""
    spec = SPECS[dataset]
    rng = np.random.default_rng(seed)
    counts, forward, classes = _toy_edges(dataset, rng)
    raw = Path(root) / "NC" / dataset / "raw" / spec["name"]
    raw.mkdir(parents=True, exist_ok=True)
    types = spec["types"]
    offsets, total = {}, 0
    for t in types:
        offsets[t] = total
        total += counts[t]
    info = {"node.dat": {"node type": {str(i): t for i, t in enumerate(types)}},
            "link.dat": {"link type": {str(i): {"start": s, "end": e, "meaning": m}
                                       for i, (s, e, m) in enumerate(spec["links"])}},
            "label.dat": {"node type": {"0": {str(i): c for i, c in enumerate(spec["classes"])}}}}
    (raw / "info.dat").write_text(json.dumps(info, indent=4), encoding="utf-8")
    with (raw / "node.dat").open("w", encoding="utf-8") as handle:
        for i, t in enumerate(types):
            for local in range(counts[t]):
                line = f"{offsets[t] + local}\tn{offsets[t] + local}\t{i}"
                if t not in spec["featureless"]:
                    line += "\t" + ",".join(f"{v:.3f}" for v in rng.normal(size=4))
                handle.write(line + "\n")
    lines = []
    for k, edges in forward.items():
        s_type, e_type = types[spec["links"][k][0]], types[spec["links"][k][1]]
        mirror = spec["mirror"][k]
        for a, b in edges:
            lines.append((offsets[s_type] + a, offsets[e_type] + b, k))
        for a, b in edges:
            lines.append((offsets[e_type] + b, offsets[s_type] + a, mirror))
    lines.sort(key=lambda r: (r[2], r[0], r[1]))
    with (raw / "link.dat").open("w", encoding="utf-8") as handle:
        for a, b, k in lines:
            handle.write(f"{a}\t{b}\t{k}\t1.0\n")
    target = types[0]
    order = rng.permutation(counts[target])
    cut = counts[target] * 3 // 10
    for name, ids in (("label.dat", order[:cut]), ("label.dat.test", order[cut:])):
        with (raw / name).open("w", encoding="utf-8") as handle:
            for local in sorted(ids.tolist()):
                handle.write(f"{offsets[target] + local}\tn{local}\t0\t{int(classes[local])}\n")
    return raw
