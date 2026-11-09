# J5.6 freebase_bb task-level H_edge without self-loops (spec v1.9 F15)
2026-11-10. **No new computation** (author's instruction): the stored review evidence holds the value.

**Source (VERIFIED-RUN in J4.7):** `../lp_evidence/j_review_task_h_edge_without_loops.txt`, output of `j_review_task_h_edge_without_loops.py`.
- self-loops kept (reported in J4.3): edges 270,106; fill seed 0 **0.7794** (seeds 0.7794–0.7829); labelled-only 0.3229 on 15,707 edges.
- self-loops dropped: edges 255,070; fill seed 0 **0.8254** (seeds 0.8254–0.8291); labelled-only **0.3529** on 14,372 edges.

**Reported value now:** 0.8254 (fill convention of spec §1), with the labelled-only diagnostic 0.3529 beside it.
- It matches spec v1.9 F15 ("0.825") and §6 ("labeled-only diagnostic 0.353"). The values are taken at the probe's printed precision (4 decimals).
- The stored evidence gives only the seed range, not per-seed values, for the loop-free convention. That is enough for the reported value, so nothing was recomputed.

## Updated
- **`results.json`:** new top-level key `task_level_h_edge.freebase_bb`, with the reported loop-free value and the loops-kept values marked "not reported". Only additions; existing keys are unchanged.
- **`results.md`:** a new final section with the same two rows. It states that `compute_heterophily.py` does not write it, so a rerun of that script would drop it.
- **`freebase_bb_task_h_edge.json` (J4.3, loops kept):** left as the J4.3 record. `results.json` names it as the source of the loops-kept row.

## Not changed
- **Chapter 3 of the advisor report** does not show the task-level value. A grep of `docs/advisor_report_cumulative/` for "0.779", "0.825" and "task-level" finds nothing, and its table is dataset-level heterophily. So chapter 3 is not edited and nothing is re-rendered, as the card says ("if it shows the value").
- The Part J report §10 (J4) keeps the J4.3 value as history. §11 (J5) states the change.

**Numbers traced:** 0.8254, 0.8291, 0.3529, 255,070 and 14,372 appear in the source file above, as printed.
