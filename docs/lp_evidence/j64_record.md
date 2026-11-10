# J6.4 HeaRT negatives: acm_pa regenerated
2026-11-10. CPU only.

**Command (VERIFIED-RUN):** `python scripts/lp/build_heart_negatives.py --tasks acm_pa`, run from `H2S-NN/`. It uses seed 42 and 500 negatives per positive, with `features=False` (committed in J6.3). C2 (feature cosine) no longer applies to acm_pa because F17 removed the author block, so PubMed is the only C2 task.

Files: `j64_build_acm_pa_output.txt`, `j64_heart_md5_before.txt` (all six tasks before), `j64_acm_pa_md5_after.txt`.

| acm_pa | Before (J5.5, with feature cosine) | After |
|---|---|---|
| `heart_valid_pos.npy` / `heart_test_pos.npy` md5 | `32cc4b20…` / `36d3ea41…` | same (positives unchanged) |
| `heart_valid_samples.npy` md5 | `975918de…` | `eb4cbc1f…` |
| `heart_test_samples.npy` md5 | `829610417…` | `0a0e1ae7…` |
| Valid: lists with random fill; random-fill entries (share of 895 × 500) | 421; 105,250 (23.5 %) | 1,790 (both halves of every list); 427,089 (**95.4 %**) |
| Test: lists with random fill; random-fill entries (share of 995 × 500) | 461; 115,250 (23.2 %) | 1,990; 474,531 (**95.4 %**) |

**Recorded, not acted on (not a stop condition):**
- Without feature cosine, the L3 and PPR candidates fill under 5 % of each acm_pa list. The rest is uniform random fill (C4).
- acm_pa's HeaRT lists are therefore close to random negatives. This should be read with that in mind when acm_pa results are reported.
- It is for the author to decide.

**Not done (author decision, 2026-11-10: "only build the link-prediction edges; no further experiments or tests"):**
- the md5-identical rebuild of the other five tasks. It was stopped before it started and wrote no file.
- the J4.4 real-file checks on the new acm_pa lists.

The other five tasks' `heart/` files were not touched; their md5 are in `j64_heart_md5_before.txt`.
