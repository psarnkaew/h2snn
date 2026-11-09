"""Reviewer probe (read-only on the repo): does the LOADER catch a reciprocal leak on freebase_bb-like
data? Builds the synthetic release in the scratchpad, then (A) puts the reverse row of one test pair
back into link.dat, (B) lists only one orientation of a two-row validation pair in link.dat.valid."""
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Acer\Desktop\h2snn\H2S-NN")
import shutil
from scripts.lp import build_derived_lp as bd
from tests.test_lp_derived import write_freebase_like, derived_config
from src.data.link_prediction_dataset import build_link_prediction_dataset

ROOT = Path(__file__).resolve().parent / "probe_data"
TARGET = ("book", "rel_0", "book")


def fresh():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    write_freebase_like(ROOT / "release")
    bd.build_task_by_pair(ROOT / "release", ROOT / "lp_derived" / "freebase_bb", 0, seed=42,
                          val_frac=0.09, test_frac=0.10)
    return ROOT / "lp_derived" / "freebase_bb"


def load(tag):
    try:
        d = build_link_prediction_dataset(derived_config(ROOT, "freebase_bb", TARGET))
        msg = {tuple(p) for p in d.edge_index.t().tolist()}
        held = {tuple(p) for p in d.test_pos_edge_index.t().tolist()} | \
            {tuple(p) for p in d.val_pos_edge_index.t().tolist()}
        rev = len({(v, u) for u, v in held} & msg)
        print(f"{tag}: LOADED, no error; held-out rows present reversed in message graph: {rev}")
    except Exception as e:  # noqa: BLE001
        print(f"{tag}: RAISED {type(e).__name__}: {str(e)[:120]}")


task = fresh()
load("baseline")
# (A) one-row test pair: add its reverse row back to link.dat
test_rows = [tuple(map(int, l.split("\t")[:2])) for l in (task / "link.dat.test").read_text().splitlines()]
rows = set(test_rows)
single = next((u, v) for u, v in test_rows if (v, u) not in rows)
with (task / "link.dat").open("a") as f:
    f.write(f"{single[1]}\t{single[0]}\t0\t1.0\n")
load("(A) reverse of a test pair re-added to link.dat")

task = fresh()
# (B) drop one orientation of a two-row validation pair from link.dat.valid
lines = (task / "link.dat.valid").read_text().splitlines(keepends=True)
pairs = [tuple(map(int, l.split("\t")[:2])) for l in lines]
ps = set(pairs)
u, v = next((u, v) for u, v in pairs if (v, u) in ps and u != v)
kept = [l for l, p in zip(lines, pairs) if p != (v, u)]
(task / "link.dat.valid").write_text("".join(kept))
load("(B) validation file lists one orientation of a two-row pair")
shutil.rmtree(ROOT)
