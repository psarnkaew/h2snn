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

## How to think (author, 2026-11-10; added after the design review that found the path cell equals MAGNN's instance encoder in a special case)
- **The goal is methodological novelty.** Theory (stability, spectra, kernels) supports a method; it is never the contribution and never the reason for a design choice. If the only answer to "what is new?" is a guarantee, the design is not novel yet.
- **Say the research question and the novelty in plain words first.** One sentence a non-specialist understands plus one concrete example (e.g. "P1 and P9 share author A7: does it matter who A7 is?"). If it cannot be said simply, it is not ready. No jargon in the claim; define every technical word.
- **Check the closest prior methods at the formula level before proposing a design.** Write their update on the same toy example as ours, compute both, and state the special case where ours reduces to theirs. "X does not have property P" in a gap table needs the formula, not a summary (MAGNN's rotation encoder equals our cell state when maps depend only on the relation type: missed from D4 until 2026-11-10).
- **For every component ask: what can it do on a concrete example that MAGNN / HAN / SeHGNN / a hypergraph cannot?** Answer with the example before any maths.
- **A warning signal stops the work for a discussion with the author.** An exploratory ablation in which the core component does not beat its own control (e.g. D11 screen: path cells vs no cells / endpoint edges) is not just logged under rule 5; it triggers a rethink of the thesis before more engineering.
- **Test the core hypothesis before building around it.** Guarantees, scaling and tooling come after the central claim has a design that can win on a known-answer task.
- **Generate alternatives from the data's structure** (what the data contains that current models ignore), not variations of the current model.
- **Give the conclusion that matters first, in one place.** No scattering a key finding across several answers, no flattery of the author's model or of earlier answers, no comfort language. Correct errors explicitly and at once (e.g. the withdrawn "order" difference from MAGNN, 2026-11-10).
- **Read sources deeply** (method, equations, what is updated and what is not) before relying on them, and state the depth of reading per source.


## How to explain to the author (author, 2026-11-11; "I like this style a lot, remember it")
- No jargon. Use everyday words and show the reasoning step by step; add a concrete example (small table, everyday analogy) when needed. Example of the liked style: the S3/S2′ explanation of 2026-11-11 — (1) the two parts of the model that can "choose", in plain words; (2) what the test wants; (3) the problem, with a concrete example; (4) the proposal; (5) how to read the result; (6) cost; (7) recommendation and reason.
