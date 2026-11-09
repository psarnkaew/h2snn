# Working preferences (author: Peeradon), for any Claude session on H2S-NN

## How to explain a proposal
1. **Concrete example first.** Use a small graph with named nodes (e.g. paper P1, author A1, paper P2), and name the actual cells, maps and numbers involved.
2. **Where it sits in the model.** Say which cell level and which map it touches (node→edge, edge→path cell, node→path cell, the operator, the head), and what stays unchanged.
3. **Then the maths, the guarantees and the trade-offs.**
4. **Define overloaded words before using them.**
   - "Meta-path" (schema, e.g. APA) vs "path instance" (A1–P–A2) vs "path cell" (the 2-cell of one instance).
   - "Path maps" (node→path-cell maps) vs "restriction maps" (node→edge).
5. **Agreed structure of the sheaf:** node → edge (learned, factorized) → path cell (forced by functoriality; computed as transport `T_v`). Free node→path-cell maps (`path_maps: "free"`) exist only as an ablation.

## How to work
- Be honest and critical; no self-praise of the author's model or of earlier answers. Correct earlier mistakes explicitly.
- Read papers and project documents fully before relying on them, and say what was and was not read.
- Verify mathematical claims numerically when possible, and state what was checked.
- Discuss decisions with the author before writing them into the spec. Give options, a recommendation and the reason.
- Design from first principles: setting → assumptions → gap to prior work → properties → method → tests. No "try A and B, keep the winner" band-aids; experiments test stated assumptions, with consequences fixed in advance.
- Check the actual code and data (file:line, config, dataset files) before claiming how the model behaves.
- Before editing a doc, check whether another session changed it (compare with the on-disk version) and never overwrite someone else's edits.

## How to communicate (style the author liked, 2026-11-09)
- Reply in Thai when the author writes Thai (English when they write English); specs, task cards and prompts for Claude Code stay in English.
- Short direct answer first, then tables for comparisons, then the recommendation with its reason.
- When work goes to Claude Code, end with a **ready-to-paste prompt** in a code block: decisions, scope, what may run, git rules (commit per unit, never push or merge unless the author approves), stop conditions.
- When the author asks a short question, answer briefly.
- Give time estimates as estimates, with what would make them longer.

## How to read empirical results (author rule, 2026-11-08; also in repo AGENTS.md and CLAUDE.md)
- Our models are untuned until the cluster protocol; baselines are published or tuned. A gap to another model (e.g. feature-free R-GCN beats H2S-NN by 5–10 points) is **not** a design signal: never call it a deficit, a "real problem", a risk, or a reason to change, delay or gate the design. Mention it only as context, labelled "untuned vs tuned".
- Design conclusions come only from ablations of the same model under identical settings and budget, from known-answer tests, or from tuned comparisons under the frozen protocol.
- Before any known-answer run, oracle or gate, the pre-registration states reachability (can the optimizer and schedule move the parameter far enough?) and an information-route audit (every route by which the label reaches a target, including memorisation). Spec §0 rule 6; added 2026-11-09 after the H5 and I3 oracles failed for want of it.
- Local runs check that things work (tests, smoke, known-answer synthetic tasks), never performance. A synthetic miss means "not shown at default settings"; use an oracle arm to separate capacity from learning.
