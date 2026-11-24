# Spec 1.31 — draft edits (model design session, 2026-11-23)

Status: **DRAFT, card 19 reported** (`card19` report `d7d89ab`, evidence `dfc9a8a`; reviewed by the model session 2026-11-24). Apply to `docs/H2S-NN_methodology_spec.md` when the author approves the new Table 1 rows. `card19` is not merged into `v3b`; the numbers carry its hash.

Decision: author 2026-11-23, option A — all three heterophily measures (ℋ², MLH, MDE) use H2GB's meta-path rule. The method is unchanged. No code that affects training, models or any H2S-NN or baseline run changes, so the freeze (`v3b` `0b9873d`) stays where it is (AE5).

## Changelog line (top of the spec)
**Changes from 1.30 (to 1.31; author decision of 2026-11-23 "heterophily meta-path set: option A", card 19; the method is unchanged):**
- (AE1) One meta-path set `M₂` for ℋ², MLH and MDE: H2GB's rule. It replaces the Hetero²Net lists / `get_metapaths` for MLH and MDE (1.14) and the two-set note of 1.15.
- (AE2) MLH is computed with H2GB's `calcHomophily` (`h_edge` per meta-path), the same call that gives ℋ². MDE is computed with Hetero²Net's formula and code (`attribute_homophily`) on each meta-path of H2GB's set.
- (AE3) Hetero²Net's lists are used only for the reproduction check of Hetero²Net's published MLH (DBLP 0.87, ACM 0.60, IMDB 0.24), reported in the appendix.
- (AE4) The perturbation study is unchanged: card 7's generator already measures h on H2GB's set. Its starting values are ACM 0.262 and DBLP 0.128.
- (AE5) Freeze unchanged. Card 19 adds an analysis-only script that reads data and card 6 records and does not touch `src/`.
- (AE7) Exact closed form for MDE on a meta-path whose `G_P = A Aᵀ` (step 2 is the transpose of step 1): inclusion–exclusion over shared mid sets (card 19, author 2026-11-24). It is used only where D-a cannot finish in the job limit (four OAG paper–field–paper paths, up to 1.6e11 edges). It equals D-a to ≤ 1.6e-8 with identical edge counts on five gate paths.
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

**AE7 — §1, MDE bullet**, after the AE2 text append:
> Large `G_P` use the routes of card 6 (H2b complete `G_P`, H2c-1 disjoint cliques, the reference, D-a chunks). A path with `G_P = A Aᵀ` that D-a cannot finish uses an exact closed form (1.31, card 19): with `F(p)` the mid set of target `p`, `n_S = #{q : S ⊆ F(q)}` and `y = x/√d`, `d_p = Σ_{∅≠S⊆F(p)} (−1)^{|S|+1} n_S` and `Σ_{(r,c)∈G_P} y_r·y_c = Σ_S (−1)^{|S|+1} ‖Σ_{q⊇S} y_q‖²`, in float64. Where D-a finished, its value is reported.

**AE4 — §6, perturbation sentence** "moves heterophily h = 1 − MLH ∈ [0, 1] (spec §1 `M₂`; ℋ² reported beside)": append:
> (1.31: `M₂` is H2GB's set, as the card 7 generator already uses; the original graphs have h = 0.262 on ACM and 0.128 on DBLP)

**`docs/ablation_analysis_preregistration_v3.md`**: no text change. "spec §1 `M₂`" and "ACM (h = 0.262 now)" become consistent under 1.31. Add one line under the Graphs paragraph:
> (Note 2026-11-23, spec 1.31: `M₂` is H2GB's set for every heterophily measure; the generator and the levels are unchanged.)

**Decisions log rows that 1.31 supersedes** (append "superseded by 1.31 (2026-11-23)" in the row, do not delete):
- 2026-11-11 "H2 uses the same meta-path set as MLH and MDE" (it is again true, by the other set).
- 2026-11-11 Datasets row "each gets its own meta-path set by the rule (ogbn-mag: Hetero2Net list PFP, PAP; mag-year: get_metapaths)".
- 2026-11-12 "ℋ² with H2GB's code and H2GB's own meta-path rule … MLH and MDE with Hetero2Net's code and sets".

## Reported numbers (card 19, final)
NC (Table 1; ℋ² unchanged):
| | ACM | DBLP | IMDB | Free. | mag | m-year | oag-cs | oag-eng | oag-chem | IEEE | Pokec | PDNS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLH 1.31 | 0.262 | 0.128 | 0.325 | 0.846 | 0.844 | 0.749 | 0.964 | 0.868 | 0.876 | 0.135 | 0.386 | 0.392 |
| MDE 1.31 | 2.187 | 0.800 | 0.593 | – | 0.003 | 0.003 | 63.084 | 93.951 | 88.555 | 0.000 | 0.006 | 0.034 |

LP MDE (appendix): acm_pp 2.586, dblp_ap 0.039, imdb_ma 0.517, pubmed 0.061, ogbn-mag_pp 0.004 (0.0035), h-pokec_uu 0.006 (0.0057); freebase_bb and lastfm "–".

**OAG MDE (×10–15 from 1.30).** Hetero²Net's MDE is a feature distance normalised by the `G_P` degrees (`homophily.py:4-14`). On a dense `G_P` it goes to 0; on a sparse one (degree ≈ 1) it approaches ¼ of the raw squared feature distance. H2GB's set adds the author-position paths of OAG (paper–author–paper through write_first/last/other). They are sparse (1.5e6–8e6 edges) and give 4.1–238 (cross-position 83–238, against ≤ 1.35 for the field paths). They dominate the mean over the 16 non-empty paths. ACM's cite/ref paths (9,402–44,587 edges, 1.2–4.7) raise ACM the same way. This is the definition, not an error. MDE stays a per-dataset descriptor, not comparable across datasets (§1 already says so).

**H2GB Table 2 MLH (card 19 step 4).** Our ℋ² matches H2GB Table 2 on every dataset. Our MLH (released `calcHomophily`, `extract_metapath` active condition `if src == cur:` at `:101`) does not match its printed MLH on MAG/OAG: Table 2 equals our set without the paper–paper citation path (≤ 1e-4), consistent with the commented-out `src != dst` condition at `:100`. H-Pokec is unexplained (0.3922 vs 0.3861). The paper states that MLH follows the released code and does not claim to reproduce Table 2's MLH.
