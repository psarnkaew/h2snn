# Published-number register (spec 1.26 §6 "Published numbers as placeholders")

**What this file is.** It lists every result cell that holds a **published** number because we have not run that (dataset, task, method) cell ourselves. It is project memory, not the paper. The author decided this on 2026-11-19 (decisions log).

**Rules (spec 1.26).**
- A placeholder enters no statistical test, and no claim rests on it.
- Before submission, each row is either replaced by our run (fill `replaced by`) or kept as published by an explicit author decision (fill `kept: decision row`).
- Whoever puts a published number into a table (the paper session or any agent) adds the row here **in the same commit**.

**Columns.**
- `protocol differences`: how the source differs from spec §6, for example split, metric (Macro-F1 vs accuracy), checkpoint selection (B1a subset), number of seeds, test negatives (LP), or tuning budget.
- `status`: `placeholder`, `replaced`, or `kept`.

| dataset | task | method | metric | value (mean ± std) | source (paper, table, page) | protocol differences | status | replaced by / kept: decision row |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — |
