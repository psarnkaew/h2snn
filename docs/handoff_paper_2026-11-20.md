# Paper session handoff, 2026-11-20 (Sections 4–5)

Read this first, then `docs/paper_session_brief_2026-11-18.md` §0 and `docs/ablation_analysis_preregistration_v2.md` (the latest ablation and analysis document).

## Role and rules (author)
- You are the PAPER session. Paper: `H2S-NN/paper/ijcai27/h2snn_ijcai27_v3b.tex` on `design-docs`.
- Do not edit anything from the Overview upward. Methodology 3.1–3.4 is approved; do not change it.
- Show proposed text in chat first. Edit the .tex only when the author says so ("แก้ลงไฟล์"), or gives an equally direct instruction ("ใส่เลย", "เพิ่มเข้าไปเลย").
- The author edits the .tex directly and leaves edits uncommitted. Keep them. Commit them together with yours (stage by name) and say so.
- Every figure change is a new version file.
- The paper must be self-contained: reviewers do not read the appendix.
- Style: concise, formal, narrative, no implementation detail. Keep the author's own style when he pastes a draft; fix only what he asks. Do not add `\paragraph` headings he did not ask for.
- Reply in Thai, short. Paper text in English.
- Every analysis paragraph must say concretely: what is measured, on which datasets and seeds, with which measures, and which table or figure shows it. Base it on the pre-registration v2, not on the old .tex text.
- Do not state as fact what you have not checked (the author caught an unsupported claim about ablation practice).
- Render with PyMuPDF (`/opt/conda/bin/python -m pip install --target <scratchpad>/pylib pymupdf`). Build: TinyTeX at `~/.TinyTeX/bin/x86_64-linux`, `latexmk -pdf -outdir=<scratchpad>/build`.
- Review artifact: https://claude.ai/artifact/GE4rDZgJDA4FdymyyWEBpv (v28, "Review of Sections 4–5"). Its source was in the old session scratchpad and may be gone. Rebuild from `action: read` if you update it.

## Decisions made this session (author)
1. **Ablation uses configuration (a) only:** every variant runs at the full model's tuned configuration, with no retuning. Configuration (b) and the capacity reading are dropped. The pre-registration v2 still says (b) decides C2, C2-cells, C14 and C15. The model session must amend it. The author passes this on.
2. **Link prediction uses random test negatives** as the main set. Commit 377f16e by the model session says "random test negatives only".
3. **Section 5 keeps every table and figure for now,** even over 7 pages. The author decides the cuts with the advisor.
4. **Heterophily rows of Table 1 are filled** from `table.md` of card 6, at heterophily worktree commit 59ab582 (ℋ² uses H2GB's rule; MLH and MDE use Hetero²Net's M₂). A caption sentence says that MDE is not bounded by 1 and is not comparable across datasets. The other caveats were left as they are, by the author's choice.

## State of the .tex (uncommitted; author and paper session edits mixed)
- **Section 4:** the author's own opening, Datasets, Baselines and Protocol paragraphs, in his style without `\paragraph` headings. Results paragraph with `\res{}` slots (claims of best on every task, generalisation, comparison with meta-path-based and sheaf baselines).
- **Section 5 (Discussion):**
  - 5.1 Ablation: the author's opening; Faces and Maps paragraphs; Table 4 `tab:ablation`, 7 rows, no markers.
  - 5.2 Learned Meta-Paths (ρ_σ, Kendall τ_b, all-ten-seeds rule, Spearman with schema homophily).
  - 5.3 Robustness to Heterophily, with Fig. 3 `fig:perturb` (placeholder box).
  - 5.4 How Faces Fix Predictions: Table 5 `tab:face` (dataset × G0–G3) and Fig. 4 `fig:case` (placeholder).
  - 5.5 One Model for Both Regimes (Prop. 4; R1–R3).
  - 5.6 Computational Cost: Table 6 `tab:cost` (16 methods; params, time, memory on IMDB and ogbn-mag) and Fig. 5 `fig:cost` (placeholder).
- **Appendix B, "Controlled variants":** rewritten for configuration (a) only.
- **Length:** the Conclusion is on p.8, and the References start on p.9 (about 1 page over).

## Open items
- **Proposed in chat and not applied:** merge 5.1 into one paragraph (text in the 2026-11-20 chat). The author has not yet said "แก้ลงไฟล์".
- **Seed count in 5.1:** the author's text says "5 seeds"; the Table 4 caption, the Controlled variants paragraph and the pre-registration say seeds 1–10. At n = 5, the smallest two-sided Wilcoxon p is 0.0625, so the pre-registered test cannot pass. The author was told and said not to fix it for now.
- **Link prediction text:** proposed and not applied:
  - Table 2 caption: random test negatives; MRR in the appendix.
  - Appendix B LP paragraph: it still describes two-hop test negatives.
  - The 5 HGB two-hop placeholders in Table 2 (GCN PubMed and LastFM, R-GCN PubMed and LastFM, MAGNN LastFM) must become `\res{}`, with the matching rows of `docs/published_number_register.md` updated in the same commit. Check commit 377f16e and card 8 amendment 3 first: the model session plans to re-run these cells.
- **Not applied:**
  - 3.4 lead-sentence fixes (Props 1, 3, 4);
  - the 3.2 "Sampling instances" tightening, which saves 2–3 lines.
- **The author's caption sentence on Table 1** ("Some metrics are -- since it doesn't have node features or labels") has no full stop and an agreement error. The author said to leave it.
