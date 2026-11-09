"""Derived link-prediction tasks in HGB link-prediction format (task card Part J, unit J1.2).

Source: an HGB node-classification release (node.dat, link.dat, info.dat). The target relation `r`
is split by fact into 81/9/10 with a recorded seed. The fixed split is the files themselves (md5
values in `split_meta.json`). See `../docs/lp_evidence/preregistration.md`.

- `node.dat`, `info.dat`: byte copies of the release (ids and features unchanged). Then
  `scripts/lp/rebuild_lp_features.py` (J5.3, spec v1.9 F12) must run on the built task: it
  rewrites, in node.dat only, the features of a type that encodes the target relation (rebuilt
  from training facts, or dropped). Re-running this build restores the release copy, so the
  rebuild runs again after it.
- `link.dat`: the release's rows in their original order and bytes, minus the rows of `r` whose pair
  is a test fact and the rows of `r' = rho(r)` whose reversed pair is a test fact.
- `link.dat.test`: the release's `r` rows of the test facts.
- `link.dat.valid`: the release's `r` rows of the validation facts. These facts stay in `link.dat`,
  as in HGB's format, where `link.dat` holds training and validation together.

Freebase BOOK-and-BOOK (`freebase_bb`, spec v1.8 F10; task card J4.3) is split by **unordered
pair** instead (`build_task_by_pair`): the relation has no stored reverse relation, yet 64 % of its
non-loop rows have their reverse stored in the same relation, so a split by directed row would put
(u, v) in test and (v, u) in training. Every stored row of a pair goes to the pair's part, the
test rows leave `link.dat` together, and self-loops stay in `link.dat` in no part.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Pair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class DerivedTask:
    source: str  # release folder relative to the data root
    relation: int  # target relation id in the release's link.dat (stored orientation)
    unit: str = "fact"  # "fact" (D7, rho-tied) or "unordered_pair" (spec v1.8 F10)


#: The spec v1.7 §6 tasks: the relations of each dataset's standard meta-path (PAP, APA, MAM),
#: and Freebase BOOK-and-BOOK (spec v1.8 F10, the author's choice after the J1.1 inventory).
DERIVED_TASKS: dict[str, DerivedTask] = {
    "acm_pa": DerivedTask("acm/acm/raw/ACM", 2),     # paper -> author
    "dblp_ap": DerivedTask("dblp/dblp/raw/DBLP", 0),  # author -> paper
    "imdb_ma": DerivedTask("imdb/imdb/raw/IMDB", 2),  # movie -> actor
    "freebase_bb": DerivedTask("freebase/freebase/raw/Freebase", 0, "unordered_pair"),  # book-book
}


@dataclass(frozen=True, slots=True)
class LinkRow:
    src: int
    dst: int
    relation: int
    raw: bytes  # the full line, newline included


def read_link_rows(path: Path) -> list[LinkRow]:
    rows = []
    for raw in path.read_bytes().splitlines(keepends=True):
        fields = raw.rstrip(b"\r\n").split(b"\t")
        if len(fields) != 4:
            raise ValueError(f"{path}: expected 4 tab-separated fields, got {raw[:80]!r}.")
        rows.append(LinkRow(int(fields[0]), int(fields[1]), int(fields[2]), raw))
    return rows


def exact_reverse(rows: list[LinkRow], relation: int) -> int:
    """`rho(relation)` by the D7 definition (`src/data/fact_layer.py:90-122`): the unique relation
    whose pair set equals the reversed pair set of `relation`."""
    pairs: dict[int, set[Pair]] = {}
    for row in rows:
        pairs.setdefault(row.relation, set()).add((row.src, row.dst))
    reversed_target = {(dst, src) for src, dst in pairs[relation]}
    partners = [rel for rel, edges in pairs.items() if edges == reversed_target]
    if len(partners) != 1:
        raise ValueError(f"relation {relation} has {len(partners)} exact reverses: {partners}.")
    return partners[0]


def split_facts(facts: list[Pair], seed: int, val_frac: float,
                test_frac: float) -> tuple[list[Pair], list[Pair], list[Pair]]:
    """(train, validation, test): `facts` sorted, then permuted with numpy's default_rng(seed)."""
    ordered = sorted(facts)
    n = len(ordered)
    n_test, n_val = round(test_frac * n), round(val_frac * n)
    perm = np.random.default_rng(seed).permutation(n)
    test = [ordered[i] for i in perm[:n_test]]
    val = [ordered[i] for i in perm[n_test:n_test + n_val]]
    train = [ordered[i] for i in perm[n_test + n_val:]]
    return train, val, test


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows_of(rows: list[LinkRow], relation: int, keep: set[Pair]) -> bytes:
    """The release's rows of `relation` whose pair is in `keep`, first occurrence, original order."""
    out, seen = [], set()
    for row in rows:
        pair = (row.src, row.dst)
        if row.relation == relation and pair in keep and pair not in seen:
            seen.add(pair)
            line = row.raw if row.raw.endswith(b"\n") else row.raw + b"\n"
            out.append(line)
    return b"".join(out)


def build_task(source: Path, out_dir: Path, relation: int, seed: int, val_frac: float,
               test_frac: float) -> dict[str, object]:
    rows = read_link_rows(source / "link.dat")
    reverse = exact_reverse(rows, relation)
    facts = sorted({(row.src, row.dst) for row in rows if row.relation == relation})
    train, val, test = split_facts(facts, seed, val_frac, test_frac)
    test_set, val_set = set(test), set(val)
    reversed_test = {(dst, src) for src, dst in test_set}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("node.dat", "info.dat"):
        shutil.copyfile(source / name, out_dir / name)
    kept, removed = [], 0
    for row in rows:
        pair = (row.src, row.dst)
        if (row.relation == relation and pair in test_set) or \
                (row.relation == reverse and pair in reversed_test):
            removed += 1
            continue
        kept.append(row.raw if row.raw.endswith(b"\n") else row.raw + b"\n")
    (out_dir / "link.dat").write_bytes(b"".join(kept))
    (out_dir / "link.dat.test").write_bytes(_rows_of(rows, relation, test_set))
    (out_dir / "link.dat.valid").write_bytes(_rows_of(rows, relation, val_set))

    meta: dict[str, object] = {
        "source": str(source).replace("\\", "/"),
        "source_md5": {name: md5(source / name) for name in ("node.dat", "link.dat", "info.dat")},
        "target_relation": relation, "reverse_relation": reverse,
        "seed": seed, "val_frac": val_frac, "test_frac": test_frac,
        "permutation": f"numpy {np.__version__} default_rng(seed).permutation over sorted facts",
        "facts": len(facts), "train": len(train), "valid": len(val), "test": len(test),
        "source_link_rows": len(rows), "link_rows_removed": removed,
        "output_md5": {name: md5(out_dir / name) for name in
                       ("node.dat", "info.dat", "link.dat", "link.dat.test", "link.dat.valid")},
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def unordered(pair: Pair) -> Pair:
    return (min(pair), max(pair))


def build_task_by_pair(source: Path, out_dir: Path, relation: int, seed: int, val_frac: float,
                       test_frac: float) -> dict[str, object]:
    """Split `relation` by unordered pair (spec v1.8 F10): the non-loop pairs {u, v}, sorted, are
    permuted exactly as `split_facts` permutes facts; each part holds every stored row of its pairs
    (one or two per pair); test rows leave `link.dat` in both orientations; self-loops stay in
    `link.dat` and are in no part (positives exclude them)."""
    rows = read_link_rows(source / "link.dat")
    pairs_by_relation: dict[int, set[Pair]] = {}
    for row in rows:
        pairs_by_relation.setdefault(row.relation, set()).add((row.src, row.dst))
    target = pairs_by_relation[relation]
    reversed_target = {(dst, src) for src, dst in target}
    if any(edges == reversed_target for rel, edges in pairs_by_relation.items() if rel != relation):
        raise ValueError(f"relation {relation} has a stored reverse relation: split it by fact.")
    loops = {pair for pair in target if pair[0] == pair[1]}
    pairs = sorted({unordered(pair) for pair in target - loops})
    train, val, test = split_facts(pairs, seed, val_frac, test_frac)
    test_set, val_set = set(test), set(val)
    test_rows = {pair for pair in target if pair[0] != pair[1] and unordered(pair) in test_set}
    val_rows = {pair for pair in target if pair[0] != pair[1] and unordered(pair) in val_set}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("node.dat", "info.dat"):
        shutil.copyfile(source / name, out_dir / name)
    kept, removed = [], 0
    for row in rows:
        if row.relation == relation and (row.src, row.dst) in test_rows:
            removed += 1
            continue
        kept.append(row.raw if row.raw.endswith(b"\n") else row.raw + b"\n")
    (out_dir / "link.dat").write_bytes(b"".join(kept))
    (out_dir / "link.dat.test").write_bytes(_rows_of(rows, relation, test_rows))
    (out_dir / "link.dat.valid").write_bytes(_rows_of(rows, relation, val_rows))

    meta: dict[str, object] = {
        "source": str(source).replace("\\", "/"),
        "source_md5": {name: md5(source / name) for name in ("node.dat", "link.dat", "info.dat")},
        "target_relation": relation, "reverse_relation": relation, "split_unit": "unordered_pair",
        "seed": seed, "val_frac": val_frac, "test_frac": test_frac,
        "permutation": f"numpy {np.__version__} default_rng(seed).permutation over sorted pairs",
        "facts": len(pairs), "train": len(train), "valid": len(val), "test": len(test),
        "rows": {"train": len(target - loops) - len(test_rows) - len(val_rows),
                 "valid": len(val_rows), "test": len(test_rows), "self_loops_in_link_dat": len(loops)},
        "source_link_rows": len(rows), "link_rows_removed": removed,
        "output_md5": {name: md5(out_dir / name) for name in
                       ("node.dat", "info.dat", "link.dat", "link.dat.test", "link.dat.valid")},
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


@dataclass(frozen=True, slots=True)
class SplitFacts:
    train: set[Pair]
    valid: set[Pair]
    test: set[Pair]
    link_pairs_by_relation: dict[int, set[Pair]]


def read_split(task_dir: Path, relation: int) -> SplitFacts:
    """The three parts of a built task: test and validation from their files, train = the target
    relation's pairs in `link.dat` minus validation."""
    link = read_link_rows(task_dir / "link.dat")
    by_relation: dict[int, set[Pair]] = {}
    for row in link:
        by_relation.setdefault(row.relation, set()).add((row.src, row.dst))
    test = {(r.src, r.dst) for r in read_link_rows(task_dir / "link.dat.test")}
    valid = {(r.src, r.dst) for r in read_link_rows(task_dir / "link.dat.valid")}
    return SplitFacts(by_relation.get(relation, set()) - valid, valid, test, by_relation)


def test_fact_leaks(task_dir: Path, relation: int, reverse: int) -> int:
    """Test facts present in `link.dat` in either orientation (as `relation` or reversed as
    `reverse`). The file-level guard behind route 2 of the pre-registration."""
    split = read_split(task_dir, relation)
    forward = split.link_pairs_by_relation.get(relation, set())
    backward = split.link_pairs_by_relation.get(reverse, set())
    return sum((u, v) in forward or (v, u) in backward for u, v in split.test)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "data" / "lp_derived")
    parser.add_argument("--tasks", nargs="+", default=list(DERIVED_TASKS), choices=list(DERIVED_TASKS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.09)
    parser.add_argument("--test-frac", type=float, default=0.10)
    args = parser.parse_args()
    for name in args.tasks:
        task = DERIVED_TASKS[name]
        build = build_task if task.unit == "fact" else build_task_by_pair
        meta = build(args.data_root / task.source, args.out_root / name, task.relation,
                     args.seed, args.val_frac, args.test_frac)
        print(json.dumps({"task": name, **{k: meta[k] for k in (
            "target_relation", "reverse_relation", "facts", "train", "valid", "test",
            "source_link_rows", "link_rows_removed")}, "output_md5": meta["output_md5"]}))


if __name__ == "__main__":
    main()
