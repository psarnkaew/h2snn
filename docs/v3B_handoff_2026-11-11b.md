# v3-B handoff (2026-11-11 03:10 JST) — state after Part L; for a new session or a new account

Everything a new session needs is in files under `C:\Users\Acer\Desktop\h2snn\docs\` and in git branch `v3b` of the same folder. Nothing depends on the earlier chat or on the claude.ai Project of the earlier account. Replies to the author in Thai; specs, cards, prompts in English; no flattery; separate DERIVED / VERIFIED-RUN / UNVERIFIED.

## 0. Read in this order
1. `docs/working_preferences.md` (and `H2S-NN/AGENTS.md`, `H2S-NN/CLAUDE.md`)
2. This file
3. `docs/H2S-NN_v3B_report.md` (Part L report; §4 judgment calls, §5 not done)
4. `docs/H2S-NN_methodology_spec.md` (v1.11, design authority; §3 equations, §4 theorems and limits L1′–L15, §5 claims)
5. `docs/v3B_final_model.html`, `docs/v3B_decisions.html`, `docs/v3B_prior_art.html`, `docs/H2S-NN_v3B_card.md`
6. Reviews: `docs/Reviewer_review.pdf`, `docs/MetaReviewer_review.pdf`

## 1. State
- Model: v3-B (three-level sheaf energy node–fact–cell, one anchored solve, PCG + implicit gradient, no gates, no filter bank, τ = e^θ learned). Implemented and checked on branch `v3b` (base `c41bf28`, head `9bbf32f`, six commits L1–L6, not pushed, not merged). Tests T1–T3, T7, T9, T10, T14 pass; 2-epoch IMDB smoke: loss 0.712 → 0.677, PCG 7–10 iterations, peak 1,324 MB (v2 889 MB), 1.15 s/epoch (v2 1.33 s).
- The other sessions' uncommitted files on `part-j` (AGENTS.md, CLAUDE.md, v2_plan, possible_reviewer_qa, session_handoff, working_preferences, paper/*, references/*.pdf, `Claude outputs/`, `vendor/*`) are still dirty: never stage, commit, stash, reset or edit them. Stage by explicit path only.
- Known environment problem: `test_heterophily_metrics` (17) and `test_baselines` (23) fail with `WinError 4551` (Windows Application Control blocks vendor clones). Not caused by v3-B; must be fixed on the machine before heterophily metrics can be computed.

## 2. Author decisions on the report §4 (2026-11-11 03:02 JST): all (a)
1(a) exact reduced Jacobi · 2(a) θ_τ stays in the scalar group · 3(a) `instance_features` at both endpoints · 4(a) keep `E_shared` fact states (do NOT eliminate all facts; reasons: saving is ~16 MB of 1.3 GB, definition matches the three-level energy, `fact_states` recovery stays available) · 5(a) · 6(a) · 7(a) no checkpointing for now · 8(a) · 10(a) · 11(a).
Consequence for wording: the difference between `full` and `hub_A` is the fact–cell coupling, not the existence of fact states (fact states carry no input information; L14). Claim C11 must say "fact–cell coupling". A fact has three roles: it hosts the direction-aware maps, it forces the cell maps (functoriality, Thm 3), and it carries the fact–cell term. Its stored state s_e adds no freedom to x*, z* (exact elimination); never write "edge states carry new information".

## 3. Next steps, in order
1. **ACM cost smoke (Claude Code, one session, no tuning).** Prompt in §5. Gives the v1 comparison (ACM 9,260 MB / 3.0 s) that answers the reviewers' scalability concern, and PCG iterations at tol 1e-4. Cost gate (memory ≤ 3× v2, PCG ≤ 40, time ≤ 4× v2) is reported, not a stop. Claim C13 may be written as a result only after this.
2. Author reads `docs/H2S-NN_v3B_report.md`; then decide merge of `v3b` into the working branch (not before ACM numbers).
3. Fix Application Control on the machine (author).
4. Paper / slides / docs edits by the author: use spec v1.11 as the only source for equations; do not copy from older HTML reports without checking; claims C11, C13, C14 are hypotheses until measured. Goal statement (English, AAAI style) was drafted in the 2026-11-11 01:36 chat; if lost, redraft: heterophily on the original graph, meta-path instances as embedded two-dimensional cells without closing edge, one energy problem, goals (i) and (ii) stated as goals not results.
5. Presentation 2026-11-11 15:00 JST: present design + proofs + implementation checks + IMDB smoke; no accuracy numbers. `progress_2026-11-11.pptx` and the paper tex were written on the superseded brief: check against v1.11 first.
6. When a GPU server is available: arm comparison (C10–C14), multi-scope PCG test if mini-batch is used, kernel test (flat sheaf gives dim ker = d; `free_cell_maps` gives 0), unroll-vs-implicit check, T16 (`hub_A` = v2 term) before claiming equality.

## 4. Open items (not blocking)
Example for R2 comment 1 (IMDB M–A–M, data-only, CPU) · full read of arXiv 2606.00937 and DSHN (2510.04727) before citing beyond abstract · `|E_shared|` on ACM/DBLP · graph-classification readout (only on the author's order) · LP card · OPEN-5 · experiment-table audit before freeze · `docs/minibatch_design_decisions.md` M2 is superseded by `docs/minibatch_v3B_addendum.md` (no edit made).

## 5. Ready-to-paste prompt for Claude Code (ACM smoke)
```text
Continue Part L of H2S-NN in C:\Users\Acer\Desktop\h2snn on branch v3b (head 9bbf32f). Read docs/H2S-NN_v3B_report.md, docs/H2S-NN_v3B_card.md and docs/H2S-NN_methodology_spec.md (v1.11) first.

Author decisions on report §4: all (a). In particular 4(a): keep E_shared as the default kept set; do not eliminate all facts.

Task: add configs/experiments/acm_h2snn_v3b.yaml (same solver values as the IMDB v3b config, WISP r = 42, d = 32, l_max = 4, arm full). Run scripts/v3b_cost_smoke.py on ACM for 2 epochs on the GPU (if OOM, halve r and record it), then one extra forward + backward at tolerance 1e-4 on IMDB and on ACM to record PCG iterations and time. Report peak memory, time per epoch, PCG iterations (fwd/bwd), |E_P|, |E_shared|, eliminated share, tau, next to v2 H6 (ACM 1,661 MB / 2.82 s) and v1 (ACM 9,260 MB / 3.0 s). Report whether the cost gate (memory <= 3x v2, PCG <= 40, time <= 4x v2) is met; a miss is a report item, not a stop. No accuracy numbers, no tuning, no other runs. Stop and report if PCG reaches T_max = 100, tau > 50, or OOM at r = 4.

Also commit under docs/v3B_evidence/ any scratch script that produced a number in the report and is not yet in the repo.

Append the results to docs/H2S-NN_v3B_report.md as section 7 (status tags, commands, DEVIATIONs, decisions for the author). One commit "L7: ACM cost smoke". Stage by explicit path only; never touch the other sessions' uncommitted files; do not push, merge or rebase. Then stop.
```

## 6. Starting a new chat (new account or new session)
1. Open the desktop app on this computer, select the folder `C:\Users\Acer\Desktop\h2snn`, and link the chat to the computer so the assistant can read `docs/`.
2. Load the skill `/research-to-coding-agent` in that account (the skill is per account).
3. First message to the new assistant:
```text
Read docs/working_preferences.md, then docs/v3B_handoff_2026-11-11b.md, then docs/H2S-NN_v3B_report.md and docs/H2S-NN_methodology_spec.md (v1.11). Use /research-to-coding-agent. Tell me in Thai what you read and how deeply, what state the project is in, and what you propose as the next step. Do not start work until I answer.
```
4. If the new account has a claude.ai Project, upload `docs/` files listed in §0 there (the old Project does not follow the account). Not required: the local folder is enough.
