# Spec v1.25 — draft edits (part Z: relational context `c` removed), for the author's approval

**Status:** DRAFT. Decision: `docs/experiment_settings_decisions.md`, row of 2026-11-18 "Relational context `c` removed" (author). The other pending 1.25 items (B1a, B2a, OOM range rule 1, §1 GC heterophily line) are drafted with card 5 and are added to this file then; part Z is independent of them.

**Scope:** spec v1.24 (`docs/H2S-NN_methodology_spec.md`) and the pre-registration that is frozen with it (`docs/ablation_analysis_preregistration_draft.md`). The positioning table (§8, row "Identity of intermediates in geometry … yes (pair-conditioned maps)") needs no edit: it now holds exactly. Change logs of earlier versions are not edited.

---

## Z-1 Header: change line for 1.25
**Insert** (above "Changes from 1.23 (to 1.24)"), merged with card 5's items when those are drafted:
> **Changes from 1.24 (to 1.25; author decision of 2026-11-18 in `docs/experiment_settings_decisions.md`):** (Z1) the relational context `c` is removed from the method: the geometry (maps `F_{v◁e}`, fact weights `a_e`) reads each endpoint's `h₀` only, as `g` already did (1.4); R10 restated; C9, its arm and the appendix pair are withdrawn. Reasons: the map claim (pair conditioning carries the identity of intermediate nodes) holds exactly only without `c`; graph structure reaches the representation only through the sheaf solve, which keeps C2's attribution clean; the evidence for `c` (D12 G0/G1) was measured on v1/v2. Accepted without a test (author).

## Z-2 §2 requirements, row R10
**Replace the row beginning** `| R10 | The geometry (restriction maps, fact weights, path weights) depends on each node's **typed neighbourhood**` **with:**
> | R10 | The geometry (restriction maps, fact weights, path weights) is a function of the **incidence alone** (its nodes' encodings `h₀`, relations and roles; no neighbourhood summary), and is computed **once** | Graph structure enters the representation only through the sheaf solve; one sheaf per forward pass keeps R5 and R7 and makes every theorem of §4 apply |

## Z-3 §3 pipeline block
**Delete the line:**
> `c   = RelCtx(h₀)                                                # per (relation, role): mean of neighbours' h₀`

**In the line beginning** `F_{v◁e}, a_e, β_f = w_f m_f` **replace** `computed once from [h₀, c] / h₀` **with** `computed once from h₀`.

## Z-4 §3 component table
**Delete the row beginning** `| Relational context `c` | R10 |`.

## Z-5 §3 "Heterogeneity" bullet
**Replace** `type encoder, relation-and-role frames, relational context `c`, path cells` **with** `type encoder, relation-and-role frames, path cells`.

## Z-6 §3.2 (c) Weights
**Delete** ` (1.4: `h₀`, not `geom = [h₀, c]`)` (no `geom` exists any more; the 1.4 change log keeps the history).

## Z-7 §5 claims table
**Delete the row beginning** `| C9 Relational context helps (R10) |`. Claim numbers are not reused.

## Z-8 §6 Ablation table bullet
**Replace** `exactly the comparisons of §5 (C2, C2b, C9–C17)` **with** `exactly the comparisons of §5 (C2, C2b, C10–C17)`.
**Delete the sentence** `The appendix pair (`g` reads `[h₀, c]` in `full` and `endpoints`) runs with configuration (a) on seeds 1–10 and is reported without a claim (1.23).`

## Z-9 Pre-registration (`docs/ablation_analysis_preregistration_draft.md`, frozen with the spec)
- Claim map: **delete the row** `| Relational context in the maps | C9 (`geometry_context: false`) | 3 |`.
- Arms table: **delete the rows** `| geometry context off | … |` and `| Appendix pair | … |`.
- Decision rules: **delete the row** `| C9 relational context | `full` vs geometry context off (a) | better on ≥ 3 of 4 | Drop `c` (spec) |`.
- Closed-items line of 2026-11-16: **append** ` Withdrawn 2026-11-18 (author): the appendix pair and C9, with `c` (spec 1.25 Z1).`
- Holm families are per claim (m = 4 datasets), so no other family changes; check any run-budget line that counts the `geometry_context_off` arm or the appendix pair, and reduce it.

---

## Not spec text: what the change touches (for the card)

**Code** (`v3b`, H2S-NN): remove `model.geometry_context` and `ablation.path_weight_context` (config keys, guards at `src/configs/config.py:67, 74, 235-238, 300-304, 331, 1346, 1408-1410, 1459, 1492, 1524-1526, 1616-1619`), `src/models/relational_context.py`, and the `[h₀, c]` path in `src/models/h2snn_model.py` (`_geometry_input` and the `factored_inputs` flags that follow `geometry_context`); arm `geometry_context_off` in `scripts/ablation/arms.py:45`; `scripts/known_answer/s3_audit.py` (imports `RelationalContext`), `scripts/v3b_cost_smoke.py`, `CLAUDE.md:888`. Tests: `tests/test_relational_context.py` (delete), `tests/test_v2_model.py` (T11-ctx, appendix-pair cases), `tests/test_wisp.py` (34 references). `endpoint_hidden_width` is unaffected (`g` reads `h₀` in both arms). Golden and mb0 fingerprints change by design: regenerate them and record the old/new hashes in the report. A config that still sets either key must be refused with a message that names spec 1.25 Z1.

**Results obtained with `c`** (open, author): card 4 known-answer (S1′-B, S3, S2′) and the card 2 central-point timings that fix B2a's trial count `N`. Either rerun on the `c`-free model before the freeze, or report them as obtained on the model with `c` (deviation) and the timings as upper bounds.

**Paper** (`h2snn_ijcai27_v3b.tex`; text proposed in chat, not edited here): line 149 sentence defining `c_v`; Eq. (maps) line 155 `u = MLP([h_v, h_{v'}, ζ_r, ρ])`; line 159 add that the geometry is a function of the incidence alone; line 222 "No relational context" sentence; line 355 ablation row.

**Other docs:** `docs/minibatch_design_v3B.md:34` (scope example mentions `c`) — a note that the difference no longer exists, at the next edit of that file.
