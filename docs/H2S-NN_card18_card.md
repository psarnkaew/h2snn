# Card 18 — Baseline meta-paths on ACM paper–paper LP (MAGNN cited-by direction; H2G-former `metapath_pe`)

Written 2026-11-22 by the model design session after the card 15 report (`card15` `2e4881c`). Baseline adapters only; H2S-NN is not touched, so no H2S-NN number changes. Report: `docs/H2S-NN_card18_report.md`.

## Why
Card 15 report:
- **MAGNN.** In MAGNN link prediction on acm_pp, the cited-by meta-paths (PrPAP, PrPSP) are read in the citing direction (`src/baselines/adapters/magnn.py:205-209`, the "stored reverse dropped by the task" branch of `hop`). They silently duplicate PcPAP and PcPSP.
- **H2G-former.** With `metapath_pe` on, H2G-former would raise an error on acm_pp. This was found by reading the code; it has not been run.

Both affect cells we run on acm_pp.

## Change
1. **MAGNN.** When a hop names `rev_<rel>` or the reverse of a relation that the task removed, and the remaining relation joins the same two types, read the remaining relation's edges **reversed**. A cited-by hop must traverse the citation backwards.
   - Paper–paper on ACM is the only case where both relations join the same types. Fix it so that PrPAP and PrPSP differ from PcPAP and PcPSP.
   - All other tasks must keep the same edges. Check this on dblp_ap, imdb_ma and freebase_bb: the hop edge arrays are unchanged.
2. **H2G-former `metapath_pe`.** Make it run on acm_pp, using the same meta-path list and direction rule as MAGNN. If the error has another cause, fix that and state it.

## Base and git
- Branch `card18` from `card15` `2e4881c516442a5c31da63e2bf745b71b0773c0e`. Copy `vendor/`.
- Use `sbatch` only. Stage by explicit path and push `card18`. Never merge into `v3b`.

## Tests (only these; no full suite)
- (a) On acm_pp, the MAGNN PrPAP instance set differs from PcPAP. Each PrPAP instance's first hop is a reversed citation.
- (b) The MAGNN hop edges on dblp_ap, imdb_ma and freebase_bb are unchanged.
- (c) One smoke epoch on acm_pp for MAGNN and for H2G-former with `metapath_pe` on: finite loss, parameters move.

## Report
Include: files changed; tests; the hop arrays before and after on acm_pp.

Last line: `branch=card18 sha=<sha> card=docs/H2S-NN_card18_card.md base=2e4881c516442a5c31da63e2bf745b71b0773c0e`.

Estimate: 2–3 h.
