# Advisor progress report

This folder holds the progress report written **for the author's advisor**. It explains the revision's changes to a reader who does not work in the code, so it is not a planning document, a task report, or a source of truth for the model.

## Files

| File | Role |
|---|---|
| `H2S-NN_progress_report.html` | Source. Edit this file, never the PDF. |
| `H2S-NN_progress_report.pdf` | Rendered copy sent to the advisor (A4). |

## Rules for agents

1. **Do not edit, regenerate or "refresh" this report unless the author explicitly asks for an update in the current conversation.** Code changes, new task reports, and plan changes are not a reason to touch it.
2. **Audience: the advisor, not a developer.**
   - No internal decision codes (D1, D7, …), task-card IDs, test IDs, commit hashes, branch names, or file paths in the text.
   - Name each change by what it does, for example "Lighter restriction maps" or "Each fact is stored once".
3. **Structure of each change:** Before → why it is a problem → after (with the equation) → a concrete example → measured result.
   - Merge changes that tell one story, as section 3 merges path sampling and path computation, instead of giving each code-level decision its own section.
4. **Every number must be traceable to a recorded run or an exact count** in `../H2S-NN_D*_report.md`, `../D*_evidence/`, `../H2S-NN_v2_plan.md`, or `../D11_evidence/screen/summary.md`.
   - A ratio may be derived from those numbers; check it by arithmetic.
   - A value derived from a verified formula rather than a run must say so in the source comment or the update note. Example: map storage with one shared scale per edge, `2·M·k·d + M + 2·|R|·d²`.
   - Write anything that is a proposal or still under test as such. The listen/broadcast gates are an example.
5. Keep claims no stronger than the evidence. Example: the submitted model's 2-cell term had zero effect in linear diffusion, but path information still reached predictions through closure edges and the readout's edge pooling. Say both.
6. The author chose to leave preliminary accuracy results out of this report. Do not add them back unless asked.

## Rendering the PDF

- **In Chrome or Edge:** open the HTML file, choose Print → Save as PDF, set paper size A4 and margins to Default, and enable "Background graphics".
- **With Playwright** (the cloud workspace has it):

  ```js
  const { chromium } = require('playwright');
  (async () => {
    const b = await chromium.launch(); const p = await b.newPage();
    await p.goto('file:///<absolute path>/H2S-NN_progress_report.html');
    await p.pdf({ path: 'H2S-NN_progress_report.pdf', format: 'A4', printBackground: true, preferCSSPageSize: true });
    await b.close();
  })();
  ```

- **Fonts used:** Bitstream Charter (body), Inter (headings), DejaVu Sans Mono (formulas). Missing fonts fall back to similar serif and sans fonts.

## Version history

| Date | Change | Requested by |
|---|---|---|
| 2026-11-08 | First version. Covers lighter restriction maps, direct path-to-node coupling, fact storage, path sampling and computation, the listen/broadcast gates (marked as under test), big-O and next steps. | Author |
| 2026-11-08 | Section 2 rewritten around the meaning of a meta-path face (coherence condition; closure-edge map = holonomy). Removed the claim that the closure edge is the HAN-style mechanism the paper argues against. Added the exact pairwise rewriting of the face term, a subsection on what the change gives up, and a corrected intermediate-node property. Figure 2 redrawn with the face, the closure edge, z_f and the maps T. Next steps name the two ablations that test this. No new run numbers. | Author |
