# Advisor progress report, 2026-11-08 (encoder and geometry stage)

The second progress report for the author's advisor. It follows `../advisor_report/` (scalability and path coupling) and covers the encoder and geometry stage: the data-pipeline check, the frequency-selective filter, the neighbourhood summary, the known-answer test of the path mechanism, and cost.

## Files

| File | Role |
|---|---|
| `H2S-NN_progress_report_2026_11_08.html` | Source. Edit this file, never the PDF. |
| `H2S-NN_progress_report_2026_11_08.pdf` | Rendered copy for the advisor (A4). |

## Rules for agents

The rules of `../advisor_report/README.md` apply unchanged. In short:

- Edit only when the author asks in the current conversation.
- The reader is the advisor:
  - no internal codes, task or test IDs, commit hashes, branch names or file paths;
  - no code-level detail.
- Every number must trace to `../H2S-NN_D12_report.md` or `../D12_evidence/`, or to the earlier reports for the cost table's first two rows.
- No benchmark accuracy. All runs are untuned, and cross-model gaps are not design signals (`AGENTS.md`).

## Number sources

- **Cost table:**
  - submitted model and the 2.5 GB row: D1 and D4 reports;
  - remaining rows: `../D12_evidence/g3_cost_output.txt`.
- **Path test:** `../D12_evidence/g3c_results.json`, `g3c_oracle_results.json`, `g3c_v1_leak_mechanism.json` and `g3c_v2_diagnosis.json`.
- **Stability bound reached 1.067:** the D12 report, section on the functional check.
- **Damping example (9% at λ = 1, 1% at λ = 1.5):** derived from (1 − ηλ)^K with η = 0.45 and K = 4.

## Rendering the PDF

As in `../advisor_report/README.md`: print to A4 with background graphics enabled, or use Playwright `page.pdf({format: 'A4', printBackground: true, preferCSSPageSize: true})`.

## Version history

| Date | Change | Requested by |
|---|---|---|
| 2026-11-08 | First version. | Author |
