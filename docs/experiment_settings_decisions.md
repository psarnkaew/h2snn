# Experiment settings: decisions log (H2S-NN)

Started 2026-11-11 by the design session. Single place where experiment-setting decisions are written as they are taken; merged into the spec (next version) only after the author approves. The spec is not edited from this file directly.

## Pending work (read before every new topic)
- [x] Review of L8 (report section 8) done 2026-11-11: all pass; part-j = v3b = 3250d2b; T3/m4 decision (a); finding: Thm 4 wording (in spec draft E-6).
- [x] Draft spec edits (item 5) -> docs/spec_v1.12_draft_edits.md (awaiting author approval; adds Thm 4/L4' and Part L decision wording): stale lines in spec section 6 (filter-bank cost gate; ablation table C2-C4b, C8, C9), C13 wording without v2 time, T16 / "hub_A = v2" wording; then commit of v1.11 (author).
- [x] Draft pre-registrations S1'-B, S3, S2' (item 6) -> docs/known_answer_preregistration_draft.md (5 open decisions); author to classify the local known-answer runs (now vs cluster).
- [ ] Experiment-settings topics of 2026-11-11 (author list): 1 mini-batch, 2 heterophily metrics (MLH, H2, MDE), 3 datasets per task (+ HSB-GDL), 4 search spaces, 5 HGB protocol vs HetSheaf (node, link, seeds), 6 graph classification as HetSheaf. Report written 2026-11-11: docs/experiment_settings.html (author decisions D1-D8 open).

## Decisions (author)
| Date (JST) | Topic | Decision | Source |
|---|---|---|---|
| 2026-11-11 | C13 timing | No paired v2/v3-B timing; memory and PCG from L7 are the results; time vs v1/baselines later on the server | chat |
| 2026-11-11 | Heterophily | Report MLH, H2 and Meta-path Dirichlet Energy (attribute heterophily) for every dataset; "-" where not measurable | chat |
| 2026-11-11 | Datasets | Avoid datasets with a single node type, for every task | chat |
| 2026-11-11 | Spec | E-1–E-7 of docs/spec_v1.12_draft_edits.md approved; applied as spec v1.12 (G11–G16), uncommitted | chat |
| 2026-11-11 | Known-answer | S1'-B approved as drafted (docs/known_answer_preregistration_draft.md); S2' option (a) chosen (rerun K2b graph under v3-B, read-out criterion unchanged, C2b wording narrowed), local, <= 2 h; S3 pending | chat |
| 2026-11-11 | Known-answer | Author: defer S1'-B, S3 and S2' (KA1 prompt not run). Note: the idealised S1'-B check (d = 1, identity maps) is not a model run; capacity of the weight channel under v3-B is not shown until S1'-B's full+oracle arm runs | chat |
| 2026-11-11 | Graph classification | Follow HetSheaf's code protocol (details to be documented) | chat |
