# Advisor progress report: cumulative

This is the revision's progress reports for the author's advisor in **one document**, with one chapter per report. Each chapter is opened by a full-page divider with its date. This is the author's decision of 2026-11-09. Like the earlier reports, it is written for a reader who does not work in the code. It is not a source of truth for the model.

## Files

| File | Role |
|---|---|
| `H2S-NN_progress_report_cumulative.html` | **Generated** by `build_cumulative.py`. Do not edit it by hand. |
| `H2S-NN_progress_report_cumulative.pdf` | Rendered copy for the advisor (A4), made by `render_pdf.js`. |
| `chapter3_2026_11_09.html` | Source of chapter 3, an HTML fragment. Edit this file for chapter 3. |
| `build_cumulative.py` | Builds the cover, the dividers and the chapters. |
| `render_pdf.js` | Renders the PDF with Playwright, A4, `printBackground: true`. |

**How chapters 1–2 are built.**
- They are the `<body>` content of `../advisor_report/H2S-NN_progress_report.html` and `../advisor_report_2026_11_08/H2S-NN_progress_report_2026_11_08.html`, **embedded unchanged**.
- `build_cumulative.py` reads those folders and never writes them. It asserts that both files have the same md5 before and after, and that each body appears verbatim in the output.
- Chapter 1's stylesheet is used for the whole document. Chapter 2's stylesheet is the same minus the `.compare` rules, which chapter 2 does not use.
- The SVG ids of the two reports (`a`, `g` and `ar`, `b`) do not collide.

**Divider dates** come from each report's README version history: 8 November 2026 for both earlier reports, and 9 November 2026 for chapter 3.
- `session_handoff.md` calls the first report "2026-11-07", but its README and its own header say 8 November 2026. The dividers follow the README, as the task card asks.

## Rules for agents
The rules of `../advisor_report/README.md` apply unchanged:
- edit only when the author asks in the current conversation;
- the reader is the advisor:
  - no internal codes, task or test IDs, commit hashes, branch names or file paths in the text;
  - no accuracy numbers;
- every number must be traceable.

To update a chapter, edit its source, then run `python build_cumulative.py` and `node render_pdf.js` from this folder.
- Chapters 1–2: their own folders, and only when the author asks for that report to change.
- Chapter 3: `chapter3_2026_11_09.html`.

## Number sources (chapter 3)

**Computed values** (DBLP, ACM, IMDB; MLH in heterophily form and the ℋ² index, rounded to three decimals): `../heterophily_evidence/results.json` and `results.md` (summary table), produced by `H2S-NN/scripts/heterophily/compute_heterophily.py`.
- They passed the validation gate in `../heterophily_evidence/j04_gate.md`: DBLP 0.872 against the published 0.87.
- **"Without these self-pairs the value would be 0.80"** is derived from the gate run's counts: (9,693 − 4,057) / (11,113 − 4,057) = 0.799 (`j04_gate.md`).
- **ACM 0.596 and IMDB 0.237** are the "MLH, Hetero^2Net set" column of `results.md`. The published 0.60 and 0.24 are from Hetero²Net (TPAMI 2025) Table II. Its Table VII swaps these two values (`../heterophily_evidence/preregistration.md`).

**Cited values** (ogbn-mag, mag-year, IEEE-CIS-G, H-Pokec): ℋ²GB (KDD 2025) Table 2, rows MLH and ℋ², rounded to three decimals. They are not recomputed.

**The small example** (P1–P4): computed by hand.
- 12 pairs, 8 agree, class shares 9/12 and 3/12, chance agreement 0.625, adjusted homophily 1/9, ℋ² 0.89.
- The same graph is the toy fixture of `H2S-NN/tests/test_heterophily_metrics.py`, whose tests check the reference code against these values.

**IMDB and Freebase (updated 2026-11-09, spec v1.8 §1 conventions; `../heterophily_evidence/results.json`, `results.md`, `j41_record.md`):**
- **IMDB:** per genre, averaged over the five genres; unlabelled movies get the reference code's random fill (fill seed 0).
  - MLH 0.3247 → 0.325; ℋ² 0.7373 → 0.737.
  - "359 of 4,932 movies": `num_labeled` 4,573 of `num_targets` 4,932.
  - "less than 0.003": the spread over 5 fill seeds is 0.0027 (ℋ²) and 0.0011 (MLH).
  - "0.763": the exact-match reproduction row on the same meta-paths, `exact_match_mlh_h2gb_heterophily_mean` 0.7630.
- **Freebase:** symmetrised graph, random fill (fill seed 0).
  - MLH 0.8464 → 0.846; ℋ² 0.9953 → 0.995.
  - Labelled-only diagnostic: ℋ² 0.8821 → 0.882, MLH 0.47755 → 0.478.
  - "about 0.001": the ℋ² spread over 5 fill seeds is 0.0012 (0.9944–0.9956).
  - "one in five": 7,954 of 40,402 books labelled.

**Not-measurable entries:** PubMed, `H2S-NN/data/PROVENANCE.md` (the label release returned a Drive 404).

## Rendering the PDF
- Run `node render_pdf.js`. It loads the `playwright` library from the global `@playwright/cli` install, or from `PLAYWRIGHT_MODULE` if set, and drives the installed Chrome (`channel: 'chrome'`), so no browser download is needed.
- Equivalent: print the HTML to A4 with background graphics enabled.
- The 2026-11-09 render has 21 pages: cover, 3 dividers, and the chapters.

## Version history

| Date | Change | Requested by |
|---|---|---|
| 2026-11-09 | First version. Chapters 1–2 are the reports of 8 November 2026, unchanged. Chapter 3 covers the heterophily of the evaluation datasets: what MLH and ℋ² measure with a small example, the check against a published value, one table of the node-classification datasets with sources, and the datasets without node labels. | Author (task card) |
| 2026-11-09 | Chapter 3: IMDB and Freebase values computed under the author's conventions (IMDB per genre; unlabelled nodes randomly labelled as the reference code does; Freebase made two-way as the benchmark's loaders do), with one sentence per convention and Freebase's labelled-only diagnostic. Chapters 1–2 unchanged; PDF re-rendered. | Author (task card J4.6) |
