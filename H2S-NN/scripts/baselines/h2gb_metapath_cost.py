"""Part M3b B7: SeHGNN's and PSHGCN's meta-path counts on the H-Pokec schema and the memory of what they store,
by arithmetic, before any run (decisions log 2026-11-13, M-path 4; search ranges unchanged).

    python scripts/baselines/h2gb_metapath_cost.py --counts ../docs/partM3b_evidence/h2gb_nc_counts.json --out <json>

Schema and node counts are read from the B2 count record (H2GB's loaded H-Pokec graph: user + 15 club types,
`user–has_friend–user` and `user–lists–<club>` with their reverses). The enumeration is each method's own code
(`src/baselines/metapath_counts.py`) on the schema graph; no feature is propagated, no data is loaded.
Memory (float32, 4 B):
* SeHGNN stored propagated features: sum over feature keys of N_user x dim(source type of the key), where a
  featured type keeps its width (user 65) and a featureless type is `torch.eye(N_type)` (SeHGNN
  `hgb/utils.py:246-247`), i.e. width N_type; label features: label keys x N_user x 2 classes.
* PSHGCN per forward (`hgb/model.py:45-100`): every term keeps one [N_all x hidden] tensor in each of the two
  passes (g and g^T), plus the input: (1 + sum_k |lis[k]|) x 2 x N_all x hidden; hidden from its NC range.
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.baselines.metapath_counts import pshgcn_terms, sehgnn_feature_keys, sehgnn_label_keys  # noqa: E402

GIB = 2 ** 30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    pokec = json.loads(Path(args.counts).read_text())["datasets"]["h-pokec"]["loaded"]
    nodes, dims = pokec["nodes"], pokec["feature_dims"]
    clubs = [t for t in nodes if t != "user"]
    letters = dict(zip(clubs, [c for c in string.ascii_uppercase if c != "U"]))
    letters["user"] = "U"
    relations = sorted({(letters[s], letters[d]) for s, _, d in (e.split("|") for e in pokec["edges"])})
    types = "".join(sorted(letters.values()))
    n_user, n_all = nodes["user"], sum(nodes.values())
    width = {letters[t]: (dims[t][0] if dims.get(t) else nodes[t]) for t in nodes}
    out: dict = {"letters": letters, "relations": ["".join(r) for r in relations], "n_user": n_user,
                 "n_all": n_all, "sehgnn": {}, "pshgcn": {}}
    for hops in (2, 3, 4):  # SeHGNN's num_hops and num_label_hops ranges {2, 3, 4}
        keys = sehgnn_feature_keys(types, relations, "U", hops)
        label = sehgnn_label_keys(types, relations, "U", hops)
        feat_bytes = sum(n_user * width[k[-1]] * 4 for k in keys)
        out["sehgnn"][hops] = {"feature_keys": len(keys), "label_keys": len(label),
                               "keys_ending_at_a_featured_type": sum(1 for k in keys if k[-1] == "U"),
                               "feature_gib": round(feat_bytes / GIB, 1),
                               "label_gib": round(len(label) * n_user * 2 * 4 / GIB, 2)}
        print("SeHGNN", hops, out["sehgnn"][hops], flush=True)
    keys = ["".join(r)[::-1] for r in relations]  # PSHGCN adjacency keys: rows first, as SeHGNN (`dst src`)
    for K in (1, 2, 3, 4, 5):  # its NC range {1, 3, 5}
        lis, _ = pshgcn_terms(keys, K)
        terms = 1 + sum(len(v) for v in lis.values())
        out["pshgcn"][K] = {"terms_per_pass": terms, "per_order": {k: len(v) for k, v in lis.items()},
                            **{f"activations_gib_hidden_{h}": round(terms * 2 * n_all * h * 4 / GIB, 1)
                               for h in (16, 64, 256)}}
        print("PSHGCN", K, out["pshgcn"][K], flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
