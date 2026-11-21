# Paper session handoff, 2026-11-20 (b): Sections 4–5 and Appendix B on pre-registration v3

Read this first. It replaces `docs/handoff_paper_2026-11-20.md` (721da5d) where they differ; that file still holds the older background. Then read:
- `docs/paper_session_brief_2026-11-18.md` §0 (the 2026-11-20 block);
- `docs/ablation_analysis_preregistration_v3.md` (binding with spec 1.29; replaces v2);
- the memory notes `paper-writing-style-approved`, `paper-self-contained`, `paper-text-output-in-chat` and `heterophily-metapath-sets-open-2026-11-20`.

The paper is `H2S-NN/paper/ijcai27/h2snn_ijcai27_v3b.tex` on `design-docs`. The last paper commits are `ce3abb5`, `9f364a9` and `3009239`, all pushed.

## Rules (author)
- Show proposed text in chat first. Edit the .tex only on a direct instruction ("แก้ลงไฟล์", "ใส่เลย", "แก้ … ด้วย"). Commit only when told ("commit ได้เลย"). Stage the .tex by name, then push.
- **Before you edit, check the file.** The author often pastes an approved draft into the file himself, sometimes with small changes, and before telling you. Compare word by word, and never overwrite his version.
- Do not edit anything from the Overview upward. Methodology 3.1–3.4 is approved.
- Never state unchecked facts. Check numbers, metrics and complexity against the source paper, the code on `v3b` or the evidence files. Mark what you derived yourself.
- Reply in Thai, short. Paper text is in English.
- Render with PyMuPDF from the scratchpad: `/opt/conda/bin/python -m pip install --target <scratchpad>/pylib pymupdf`. Build with TinyTeX: `~/.TinyTeX/bin/x86_64-linux`, `latexmk -pdf -outdir=<scratchpad>/build`.

## Writing shape for analysis paragraphs (author: "นี่คือวิธีการเขียน analysis ที่ดีและถูกต้อง จำไว้")
One paragraph in the shape of the author's 5.1:
1. "To test whether X (RQn), we compare/analyse [what] on [datasets] with 5 seeds, and report [measure] (Table/Figure)."
2. One or two sentences on what is measured or changed, and why.
3. `\res{}`.
4. "The results from Table~… show that …, suggesting that …" (the best-case result, to be checked against real results).

Other rules for these paragraphs:
- Plain titles of the form "… Analysis".
- No examples, rules or thresholds in the main text; they go to Appendix B.
- Use "we" only in the opening and method sentences.
- Use connectives so the text is concise but not abrupt.
- **Rejected:** narrative "should … / First, Second, Third" text and catchy titles. The author called this "เหมือนเด็กเล่น".

Captions are short, use the same terms as the text and say what each column or panel shows.

## Decisions made this session (author)
1. **Main-table models** (the author's table):
   - NC (12 datasets): the 4 groups as in the Baselines paragraph.
   - LP (8 tasks): GCN, GAT | NSD, CSNN | R-GCN, HAN, MAGNN, PSHGCN | H2G-former, Hetero²Net, H2SGNN, HetSheaf.
   - GC (4 datasets): GCN, GAT | NSD | R-GCN, HAN | HetSheaf.
   - LP and GC tables now have a Group column.
   - The name is spelled **HINormer** everywhere.
2. **Link prediction: random test negatives only.**
   - The 5 HGB 2-hop placeholders are back to `\res{}`.
   - Register rows are `removed` (6a559ae).
   - Appendix B LP text is fixed.
3. **H2GB published numbers are not filled** (author: "ช่างมัน"):
   - H2GB reports Accuracy on ogbn-mag, mag-year, oag-* and H-Pokec, and F1 on IEEE-CIS-G and PDNS. Our table is Macro-F1.
   - IEEE-CIS-G stays filled: its F1 is Macro-F1 (author decision of 2026-11-11).
   - PDNS does not say which F1 averaging is used.
   - H2GB has no LP or GC results.
4. **Section 5 is rewritten on pre-registration v3:**
   - 5.1 is the author's own text.
   - 5.2 Learned Meta-Paths Analysis.
   - 5.3 Heterophily Perturbation Analysis.
   - 5.4 Face Effect Analysis.
   - 5.5 Energy and Representation Analysis.
   - 5.6 Computational Cost Analysis.
   - The ablation table has 8 rows (with Uniform weights) and uses 5 seeds.
   - New tables and figures: `tab:schema`, `fig:schema`, `tab:perturb`, `tab:regime` (Transport, Energy, Probe G_low/G3 for Full and Id.), `fig:regime`, `tab:complexity`.
5. **Complexity table (`tab:complexity`, table\*):**
   - **H2S-NN rows** come from a second reviewer and were checked against code on `v3b`:
     - WISP step: O(|R| + ℓ_max log|E|).
     - Activations: O(|E|d) and O(Σℓ_f d).
     - Solver working memory: O((N+|E|+Σℓ_f)d).
     - Backward: one more solve, with memory independent of K.
   - **Baselines with stated complexity:**
     - GCN and NSD: Bodnar et al. 2022, Sec. 5 and App. E.1.
     - HetSheaf TE, non-diagonal: Braithwaite et al. 2024, App. C.4.5, Table 27.
   - **Baselines derived by us (marked †):** R-GCN, HAN (on meta-path graphs E_P), SeHGNN, Hetero²Net, H2SGNN.
   - **Space is derived by us for every baseline.**
   - **H2G-former is left out:** H2GB gives no complexity and no value of k for its k-hop attention.
   - **The 5.6 text states the cost:** every edge applies a d×d rotation at every iteration. That is a factor d over GCN, and more than NSD or HetSheaf, whose stalk dimension is d_s ≤ 5. The author asked for a fair, not one-sided, comparison.
6. **Appendix B is on v3:**
   - Controlled variants: 4-of-5 rule, 3-of-4 claims, no p-values.
   - Learned meta-paths.
   - Perturbation: drop, gap, E1 and E2.
   - Face effect: tertiles, G_low, net fix rate, normalised margin.
   - Energy and representation: R1–R3 rules.

## Open items
1. **Red TODO before Table 1 (do not remove).** The heterophily measures use different meta-path sets:
   - ℋ² uses H2GB's rule (spec 1.15).
   - MLH and MDE use Hetero²Net's list (spec 1.14 §1).
   - This goes against the 2026-11-11 decision of one set for all three.
   - ACM MLH is 0.262 under H2GB's rule and 0.404 under the Hetero²Net list. Table 1 shows 0.404.
   - The perturbation pre-registration quotes ACM h = 0.262 as "spec §1 M₂", but the spec §1 list gives 0.404, so the set used by card 7's generator must be checked.
   - The author has no compute to fix this now. The note points to "H2S-NN design session 04". The recommended fix is one rule (H2GB's), then recompute, then update Table 1 and 5.3.
2. **Length.** The main text ends on about p.9–10, and Tables 9 (complexity) and 10 (cost) float after the References. The author decides the cuts with the advisor and keeps every table and figure for now.
3. **5.1: grammar fixed** in 4fc06a2, at the author's request ("grammar only, nothing else"): `\ref{tab:ablation}`, `\method{}`, spelling, "sheaves", "performance". It still has no `\res{}` slot; add one only if the author asks.
4. **Best-case result sentences.** The closing sentences of 5.2–5.6 assume the best case. Check each against the real results. The 5.6 sentence ("stays within the time and memory of the meta-path-based models") is the most at risk.
5. **Complexity table.** Two things still need checking before submission:
   - the R_r build cost O(|R|d³) (FastHPP; assumed, not measured);
   - the derived (†) baseline rows.
6. **Carried over from the previous handoff and still not applied:**
   - the 3.4 lead-sentence fixes (Props 1, 3, 4);
   - the 3.2 "Sampling instances" tightening;
   - the author's Table 1 caption sentence "Some metrics are -- since it doesn't have node features or labels" (the author said to leave it).
7. **Conclusion rewritten, NOT committed** (the author asked for the edit but not yet for a commit). It has the question, the method in one sentence, the propositions, `\res{}`, best-case results, and the cost and limits: the d×d rotation per edge per iteration, and an ablation on NC only. Ask the author to approve it, then commit.
8. **Review artifact** https://claude.ai/artifact/GE4rDZgJDA4FdymyyWEBpv is not updated for this session's changes.
