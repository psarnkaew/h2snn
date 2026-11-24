# Handoff to the paper session — 2026-11-24 (spec 1.31, heterophily rows)

From the model design session. Source: spec 1.31 (applied 2026-11-24), card 19 (`card19` report `d7d89ab`, table `docs/heterophily_evidence/h2gb_set/table.md` on `card19`).

## Author instruction
**Edit the table only.** Change the two rows below in Table 1 (`tab:node-cls`) of `H2S-NN/paper/ijcai27/h2snn_ijcai27_v3b.tex`. Nothing else changes:
- not the caption;
- not the red TODO line before the table (it stays);
- not §2.2, §5.3 or the appendix.

## Rows
The ℋ² row is unchanged. Replace the MLH and MDE rows with:

```latex
    \multicolumn{2}{l}{Heterophily MLH} & 0.262 & 0.128 & 0.325 & 0.846 & 0.844 & 0.749 & 0.964 & 0.868 & 0.876 & 0.135 & 0.386 & 0.392 \\
    \multicolumn{2}{l}{Heterophily MDE} & 2.187 & 0.800 & 0.593 & -- & 0.003 & 0.003 & 63.08 & 93.95 & 88.56 & 0.000 & 0.006 & 0.034 \\
```

The column order is the table's own: ACM, DBLP, IMDB, Free., mag, m-year, oag-cs, oag-eng, oag-chem, IEEE, Pokec, PDNS.

What changed (for the session's information; do not write it into the paper):
- **MLH:** ACM 0.404 → 0.262, mag 0.873 → 0.844, m-year 0.772 → 0.749, oag-cs/eng/chem 0.980/0.912/0.921 → 0.964/0.868/0.876, Pokec 0.385 → 0.386, PDNS 0.485 → 0.392.
- **MDE:** ACM 0.737 → 2.187, mag and m-year 0.004 → 0.003, oag-cs/eng/chem 5.379/9.202/6.150 → 63.08/93.95/88.56, PDNS 0.002 → 0.034.

## Not in this edit (kept for later, only on author request)
- LP MDE for an appendix table: acm_pp 2.586, dblp_ap 0.039, imdb_ma 0.517, pubmed 0.061, ogbn-mag_pp 0.004, h-pokec_uu 0.006; freebase_bb and lastfm "–".
- The caption/appendix sentences on MDE scale and on H2GB Table 2's MLH are in `docs/spec_v1.31_draft_edits.md` ("Reported numbers").
