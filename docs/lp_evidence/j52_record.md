# J5.2 Feature leak before the fix (features-only scorer on the existing HeaRT lists)
2026-11-10. Scorer, rank and control as pre-registered (`j5_preregistration.md`, J5.2/J5.3).

**Command (VERIFIED-RUN):** `python scripts/lp/lp_feature_leak.py --rules ../docs/lp_evidence/j52_rules.json --tasks dblp_ap imdb_ma --json-out ../docs/lp_evidence/j52_leak_before.json`, run from `H2S-NN/`, CPU.
- Output: `j52_leak_before_output.txt`.
- Lists: `data/lp_derived/<task>/heart/heart_{valid,test}_*.npy` (J4.4, 500 negatives per positive).

**Scorer.** s(x, y) = the share of y's contribution under the type's rule that the stored feature of x contains. **Ctrl-G** = the same scorer with x's feature replaced by the rule on x's training facts. Reading no stored feature of x, it is the information the training graph gives.

| Task (type, rule) | Split | Scorer MRR | Scorer Hits@10 | Ctrl-G MRR | Ctrl-G Hits@10 |
|---|---|---|---|---|---|
| acm_pa (author, OR) | valid | **1.000** | **1.000** | — | — |
| acm_pa | test | **1.000** | **1.000** | — | — |
| imdb_ma (actor, MEAN) | valid | **0.9105** | **0.9662** | 0.0566 | 0.0632 |
| imdb_ma | test | **0.9226** | **0.9777** | 0.0439 | 0.0562 |
| dblp_ap (author, best inexact: OR over terms, 133/334 columns) | valid | 0.0856 | 0.2596 | 0.0234 | 0.0402 |
| dblp_ap | test | 0.0855 | 0.2592 | 0.0244 | 0.0423 |

All-tie reference: MRR 0.00398, Hits@10 0.

**acm_pa is not recomputed (author's instruction).** It reuses the reviewer's binary containment, true pair strictly first in 895/895 and 995/995 lists (`j_independent_review.md`, `j_review_probes_output.txt`). Strictly first under binary containment means s = 1 for the positive and s < 1 for every negative, so MRR = Hits@10 = 1.000 (DERIVED).

## Against the pre-registration
- **imdb_ma ≥ 0.9: met** (0.91 / 0.92). It is below 1.000 because a negative movie whose few tokens all lie in the actor's support also scores 1 (ties).
- **dblp_ap: the pre-registered "wrong if" (< 0.1) is met** (0.086). Diagnosis:
  - The author rule is not exact (J5.1). The scorer sees only the 133 columns that match a term exactly.
  - The author's keyword set is a selection from its papers (J5.1 diagnosis), so a held-out paper's terms are often absent from it.
  - The leak is still there relative to the control: Hits@10 0.26 against 0.04 (×6), MRR +0.06.
  - The type is dropped (J5.3), which removes this route completely. No scorer was changed.
- Ctrl-G before the fix is reported for scale only; the gate applies after the fix (J5.3).

## Verification
`tests/test_lp_feature_leak.py` (4 tests, OK):
- all ties → rank 251, MRR 1/251;
- one tie → rank 1.5 (HeaRT's mean of optimistic and pessimistic rank);
- containment on a hand example; the column map of the contribution.
