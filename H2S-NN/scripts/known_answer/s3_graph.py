"""Card 4 K4: the S3 known-answer graph (pre-registration `docs/known_answer_preregistration_draft.md`,
S3, FROZEN; decision B-9 (1)): can `g` learn instance-level relevance that no fact window and no
endpoint pair can see?

Structure (one schema, five types, four distinct relations r1..r4, every intermediate inside one target):

    P -r1- A -r2- B -r3- C -r4- Q        per target P: 4 A, each A 4 B, each B 4 C, each C one Q
                                         = 64 length-4 cells P-A-B-C-Q per target (a tree).

Tags (1 bit, features carry +-1): t_P on P, t_B on B, t_Q on Q; A and C carry no tag. Relevance of a
cell: t_P xor t_B xor t_Q = 0. Exactly one cell per target is relevant (its Q carries the class y); the
other 63 have parity 1 and an independent random class.

Tag balance (exact, by a balanced design over the targets): every A has two B of each tag (so the mean
B tag around an A is 0); the targets are split evenly over (t_P, tag of the relevant B) in {0,1}^2 and y
is balanced inside each of the four cells of that design; the relevant B is drawn among its A's two B
with that tag. Then, for every node type and relation, the context tag means (and the node's own tag)
have the same distribution on the relevant cell's nodes as on the others (checked by `s3_audit.py`):
e.g. at C the pair (t_B, t_Q) is uniform over {0,1}^2 both for the relevant C and for the others.
A fact window (u's neighbours, u, v, v's neighbours) spans three edges, so it holds at most two of
{P, B, Q}; any two of the three tags are independent of relevance. `endpoints` sees P and Q only.

Features: P [t_P, 4 noise]; A [4 noise]; B [t_B, 4 noise]; C [4 noise]; Q [one-hot class (2), t_Q,
4 noise]. Split 50/25/25 (`h5_synthetic._split`, seed 1). Generator seed 0 (fixed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "docs" / "WISP_evidence"))
import h5_synthetic as h5  # noqa: E402  (the shared HeteroData builder and split)

NOISE = 4
FAN = 4          # A per P, B per A, C per B
TARGET = "p"
CARRIER = "q"


def _pm(bits: torch.Tensor) -> torch.Tensor:
    return bits.float().unsqueeze(1) * 2 - 1


def graph_s3(n: int = 2000, seed: int = 0):
    if n % 8:
        raise ValueError("n must be a multiple of 8 (balanced 2 x 2 x 2 design).")
    g = torch.Generator().manual_seed(seed)
    num_a, num_b, num_c = n * FAN, n * FAN ** 2, n * FAN ** 3
    # design over targets: (t_P, relevant-B tag, y), each of the 8 combinations n/8 times
    design = torch.arange(8).repeat_interleave(n // 8)[torch.randperm(n, generator=g)]
    t_p = (design >> 2) & 1
    rel_b_tag = (design >> 1) & 1
    y = design & 1
    # B tags: each A has two B of each tag, in random order
    t_b = torch.stack([torch.tensor([0, 0, 1, 1])[torch.randperm(FAN, generator=g)]
                       for _ in range(num_a)]).reshape(-1)
    a_of_b = torch.arange(num_b) // FAN
    p_of_a = torch.arange(num_a) // FAN
    b_of_c = torch.arange(num_c) // FAN
    p_of_c = p_of_a[a_of_b[b_of_c]]
    # the relevant chain of each target: an A, one of its two B with the design's tag, a C under it
    pick_a = torch.randint(FAN, (n,), generator=g)
    a_rel = torch.arange(n) * FAN + pick_a
    b_children = a_rel.unsqueeze(1) * FAN + torch.arange(FAN)              # [n, 4]
    matches = t_b[b_children] == rel_b_tag.unsqueeze(1)                   # two True per row
    choice = torch.randint(2, (n,), generator=g)
    b_rel = torch.stack([b_children[i][matches[i]][choice[i]] for i in range(n)])
    c_rel = b_rel * FAN + torch.randint(FAN, (n,), generator=g)
    relevant_c = torch.zeros(num_c, dtype=torch.bool)
    relevant_c[c_rel] = True
    # Q tags and classes (Q i hangs under C i)
    t_b_of_c = t_b[b_of_c]
    t_q = 1 ^ t_p[p_of_c] ^ t_b_of_c                                       # parity 1
    t_q[c_rel] = t_p ^ t_b[b_rel]                                         # parity 0
    q_class = torch.randint(2, (num_c,), generator=g)
    q_class[c_rel] = y
    assert int(((t_p[p_of_c] ^ t_b_of_c ^ t_q) == 0).sum()) == n
    xs = {
        "p": torch.cat([_pm(t_p), torch.randn(n, NOISE, generator=g)], 1),
        "a": torch.randn(num_a, NOISE, generator=g),
        "b": torch.cat([_pm(t_b), torch.randn(num_b, NOISE, generator=g)], 1),
        "c": torch.randn(num_c, NOISE, generator=g),
        "q": torch.cat([torch.nn.functional.one_hot(q_class, 2).float(), _pm(t_q),
                        torch.randn(num_c, NOISE, generator=g)], 1),
    }
    relations = {
        ("p", "r1", "a"): torch.stack([p_of_a, torch.arange(num_a)]),
        ("a", "r2", "b"): torch.stack([a_of_b, torch.arange(num_b)]),
        ("b", "r3", "c"): torch.stack([b_of_c, torch.arange(num_c)]),
        ("c", "r4", "q"): torch.stack([torch.arange(num_c), torch.arange(num_c)]),
    }
    info = {"relevant_cpaper": c_rel.clone(), "max_length": 4, "target": TARGET, "carrier": CARRIER,
            "n": n, "t_p": t_p, "t_b": t_b, "t_q": t_q, "a_rel": a_rel, "b_rel": b_rel,
            "c_rel": c_rel, "relevant_c": relevant_c, "y": y}
    return h5._hetero(TARGET, xs, relations, y), info
