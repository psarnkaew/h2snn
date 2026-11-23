# Spec 1.31 — draft edits (model design session, 2026-11-23)

Status: **DRAFT, not applied.** Apply to `docs/H2S-NN_methodology_spec.md` after card 19 reports (`docs/H2S-NN_card19_card.md`) and the author approves the new Table 1 rows.

Decision: author 2026-11-23, option A — all three heterophily measures (ℋ², MLH, MDE) use H2GB's meta-path rule. The method is unchanged. No code that affects training, models or any H2S-NN or baseline run changes, so the freeze (`v3b` `0b9873d`) stays where it is (AE5).

## Changelog line (top of the spec)
**Changes from 1.30 (to 1.31; author decision of 2026-11-23 "heterophily meta-path set: option A", card 19; the method is unchanged):**
- (AE1) One meta-path set `M₂` for ℋ², MLH and MDE: H2GB's rule. It replaces the Hetero²Net lists / `get_metapaths` for MLH and MDE (1.14) and the two-set note of 1.15.
- (AE2) MLH is computed with H2GB's `calcHomophily` (`h_edge` per meta-path), the same call that gives ℋ². MDE is computed with Hetero²Net's formula and code (`attribute_homophily`) on each meta-path of H2GB's set.
- (AE3) Hetero²Net's lists are used only for the reproduction check of Hetero²Net's published MLH (DBLP 0.87, ACM 0.60, IMDB 0.24), reported in the appendix.
- (AE4) The perturbation study is unchanged: card 7's generator already measures h on H2GB's set. Its starting values are ACM 0.262 and DBLP 0.128.
- (AE5) Freeze unchanged. Card 19 adds an analysis-only script that reads data and card 6 records and does not touch `src/`.
- (AE6) LP MDE is recomputed on H2GB's set, per endpoint type that keeps features, on the task's training graph. It is reported in the appendix (author, 2026-11-23).

## Edits

**AE1 + AE2 — §1, bullet "Meta-path set and graph (1.14, replaces 1.8)"**: replace the whole bullet with:
> - **Meta-path set and graph (1.31, replaces 1.14/1.15).** `M₂` is H2GB's rule for all three measures (author, 2026-11-23): every length-2 sequence of stored relations from the target type back to it, same-type relations included (`calcHomophily.extract_metapath`, `H2GB/calcHomophily.py:91-105`, enumerated by `src/analysis/heterophily.py` `h2gb_metapaths`). A meta-path whose `G_P` is empty is skipped, as `calcHomophily` does. ℋ² and MLH come from one `calcHomophily` call per label problem and fill seed: ℋ² = 1 − mean `h_adj`, MLH = 1 − mean `h_edge`, over the same non-empty meta-paths. On equal sets, `h_edge` equals Hetero²Net's `label_homophily` (DBLP 0.1278 and IMDB 0.3247 in both codes, card 6). MDE applies Hetero²Net's `attribute_homophily` (`homophily.py:4-14`) to the `G_P` of each meta-path of the same set. Hetero²Net's predefined lists (Appendix B.1) are used only for the reproduction check of its published MLH (appendix). ogbn-mag and mag-year share a graph, so they share `M₂`. Their values differ only through the labels. The graph is symmetrised as the loaders build it (Freebase needs it). Self-pairs follow the reference code.

**AE2 — §1, bullet "MLH"**: replace "The aggregation, de-duplication and self-loop conventions are those of the reference code (recorded in the J0 report)." with:
> Computed as 1 − mean `h_edge` of H2GB's `calcHomophily` over `M₂` (1.31). The de-duplication and self-loop conventions are those of that code.

**AE2 — §1, bullet "Attribute heterophily: MDE (1.14)"**: replace "on raw features of the target type, same `M₂` as MLH" with:
> on raw features of the target type, on H2GB's `M₂` (1.31; one `G_P` per relation sequence, so e.g. ACM's paper–cite–paper and paper–ref–paper paths are separate terms).

In the same bullet, in the sentence "Link prediction: computed on the task's files after the F17 feature rule (author decision D1), per endpoint type that keeps features, on the training graph", append:
> ; `M₂` of an endpoint type is H2GB's rule with that type as the target, on the training graph (the task relation's training facts included, as every model sees them); reported in the appendix (1.31, AE6)

**AE4 — §6, perturbation sentence** "moves heterophily h = 1 − MLH ∈ [0, 1] (spec §1 `M₂`; ℋ² reported beside)": append:
> (1.31: `M₂` is H2GB's set, as the card 7 generator already uses; the original graphs have h = 0.262 on ACM and 0.128 on DBLP)

**`docs/ablation_analysis_preregistration_v3.md`**: no text change. "spec §1 `M₂`" and "ACM (h = 0.262 now)" become consistent under 1.31. Add one line under the Graphs paragraph:
> (Note 2026-11-23, spec 1.31: `M₂` is H2GB's set for every heterophily measure; the generator and the levels are unchanged.)

**Decisions log rows that 1.31 supersedes** (append "superseded by 1.31 (2026-11-23)" in the row, do not delete):
- 2026-11-11 "H2 uses the same meta-path set as MLH and MDE" (it is again true, by the other set).
- 2026-11-11 Datasets row "each gets its own meta-path set by the rule (ogbn-mag: Hetero2Net list PFP, PAP; mag-year: get_metapaths)".
- 2026-11-12 "ℋ² with H2GB's code and H2GB's own meta-path rule … MLH and MDE with Hetero2Net's code and sets".

## What changes in reported numbers (expected; card 19 confirms)
| Dataset | ℋ² | MLH 1.30 → 1.31 (from card 6 ℋ² records) | MDE |
|---|---|---|---|
| ACM | 0.399 (same) | 0.404 → **0.262** (7 paths, incl. cite/ref) | recompute |
| DBLP | 0.173 | 0.128 (same set) | same |
| IMDB | 0.737 | 0.325 (same set) | same |
| Freebase | 0.995 | 0.846 (8 paths, from 7; 0.8458 → 0.8464) | – |
| ogbn-mag | 0.877 | 0.873 → **0.844** (adds paper–cites–paper) | recompute |
| mag-year | 0.965 | 0.772 → **0.749** | recompute |
| oag-cs / eng / chem | 0.965 / 0.873 / 0.886 | 0.980 → **0.964** / 0.912 → **0.868** / 0.921 → **0.876** (16 non-empty of 46 paths) | recompute |
| IEEE-CIS-G | 0.985 | 0.135 (same 11 paths) | same |
| H-Pokec | 0.949 | 0.385 → **0.386** (adds has_friend) | recompute |
| PDNS | 0.787 | 0.485 → **0.392** (5 paths, from 1) | recompute |

Point for the paper: H2GB's own Table 2 prints MLH 0.8731 (ogbn-mag), 0.7718 (mag-year) and 0.3922 (H-Pokec). These equal our 1.30 values on the shorter sets, not the 1.31 values on H2GB's ℋ² set. Card 19 step 4 reads H2GB's code to find out why. The paper must not claim that our MLH reproduces H2GB's Table 2.

LP MDE (appendix), per endpoint type with features. 1.30 values are from the card 6 records. Card 19 step 3b computes the 1.31 values:
| Task | Endpoint type | 1.30 MDE (set) | 1.31 |
|---|---|---|---|
| acm_pp | paper | 0.737 (PAP, PSP, PTP) | recompute (adds paper–cite–paper) |
| dblp_ap | paper | 0.0386 (PTP, PVP, PAP) | gate: expected equal |
| imdb_ma | movie | 0.517 (MDM, MAM, MKM) | gate: expected equal |
| pubmed | disease | 0.0780 (3 paths) | recompute (adds disease–disease relations) |
| ogbn-mag_pp | paper | 0.00401 (PFP, PAP) | recompute (adds paper–cites–paper) |
| h-pokec_uu | user | 0.00576 (15 paths) | recompute (adds has_friend) |
| freebase_bb, lastfm | – | – (features removed, F17) | – |
