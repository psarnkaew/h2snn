"""J5.1 diagnosis of the DBLP miss: author features are a function of the author's paper set
(identical sets -> identical features, 150/150) but no pre-registered rule is exact. For the best
candidate (OR over the papers' terms, V2:r1/or) and each author column without an exact match,
take the term column with the largest agreement and count the two kinds of disagreement:
author 1 / rule 0 (the author has a keyword none of its papers here carries) and author 0 /
rule 1. Read only. Run from H2S-NN/: python ../docs/lp_evidence/j51_dblp_diagnosis.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("scripts/lp").resolve()))
import lp_feature_rules as fr  # noqa: E402

spec = fr.task_spec("dblp_ap", Path("data/lp_derived"))
types = fr.read_nodes(spec.release)
links = fr.read_links(spec.release / "link.dat")
rel_types = fr.relation_types(spec.release / "info.dat")
author, paper = types[0], types[1]
n = fr.neighbourhood(links[spec.target], rel_types[spec.target], author, paper)
g = fr.vocabulary_sources(types, links, rel_types, 0, 1, spec.target, None)["V2:r1"]
agg = fr.aggregate(n, g, "or").tocsc()
x = author.x.tocsc()
mapping, amb, matched, holds, _ = fr.evaluate_rule(author.x, agg.tocsr(), g, False)
deg = np.asarray(n.sum(axis=1)).ravel()
print(f"matched columns {matched}/334 (ambiguous {amb}); nodes holding {int(holds.sum())}/4057; "
      f"authors with no paper: {int((deg == 0).sum())}; all-zero author rows: "
      f"{int((np.asarray(author.x.sum(axis=1)).ravel() == 0).sum())}")
support = (agg != 0).astype(np.float64).tocsc()
one_zero = zero_one = cells = 0
best_agree = []
for j in np.flatnonzero(mapping == -1):
    col = x[:, j].toarray().ravel()
    overlap = (support.T @ (col != 0).astype(np.float64))
    v = int(np.argmax(overlap))
    rule = agg[:, v].toarray().ravel()
    one_zero += int(((col != 0) & (rule == 0)).sum())
    zero_one += int(((col == 0) & (rule != 0)).sum())
    cells += int((col != 0).sum())
    best_agree.append(float(((col != 0) == (rule != 0)).mean()))
print(f"unmatched columns {int((mapping == -1).sum())}: author-1 cells {cells}; with the best "
      f"term column: author 1 / rule 0 = {one_zero}, author 0 / rule 1 = {zero_one}; "
      f"per-column agreement median {np.median(best_agree):.4f}, min {min(best_agree):.4f}")
