# B3 — OAG identity check (decision of 2026-11-12)

**Result: settled from the processed files. The folder `NC/OAG/engineering` holds the chemistry field and the
folder `NC/OAG/chemistry` holds the engineering field; `NC/OAG/cs` holds computer science.** The loaders and the
registry names follow the verified identity: `oag-eng` reads `OAG/chemistry`, `oag-chem` reads
`OAG/engineering`, `oag-cs` reads `OAG/cs` (`src/data/data_layout.py`, `H2GB_NC_DIRS`).

**Rule applied:** a folder's identity is the Table 1 row whose node count, node-type count, edge count and
edge-type count it equals exactly (loaded graph, after H2GB's `ToUndirected(merge=True)`); the split ratios and
the file-size order are checked as corroboration. The folder name is not used.

## Why the folders are swapped (VERIFIED-CODE)
`H2GB/datasets/oag_dataset.py:55-57` maps the name `engineering` to the hub file `OAG_Chemistry.pt` and
`chemistry` to `OAG_Engineering.pt`; the processed folder is named after the name (`processed_dir` =
`<root>/<name>/processed`, `:75-76`). The author's folders were produced by H2GB's own code
(`examples/use_dataset.py`, configs `oag-eng`→`engineering`, `oag-chem`→`chemistry`), so each folder holds the
file of the other field. `OAG/raw` is empty, so the hub file names cannot be checked on disk; the counts settle it.

## Evidence (counts by folder: `b3_counts_by_folder/`, jobs 16759 and 16760 on g3an01, loaded under H2GB's names; oag-cs: 16758)

| Folder | file (bytes) | nodes (types), loaded | edges (types), loaded | classes | labelled split train/val/test | Table 1 row it equals |
|---|---|---|---|---|---|---|
| `cs` | 3,649,001,164 | 1,112,691 (4) | 27,537,448 (22) | 3,514 | 79.6 / 8.5 / 11.9 % | **oag-cs**: 1,112,691 (4), 27,537,448 (22), 3,514, Time 80/9/11 |
| `engineering` | 6,213,821,636 | 1,918,881 (4) | 38,098,014 (22) | 2,986 | 90.6 / 8.0 / 1.4 % | **oag-chem**: 1,918,881 (4), 38,098,014 (22), 2,985, Time 90/8/2 |
| `chemistry` | 2,960,710,156 | 929,315 (4) | 12,346,854 (22) | 3,957 | 88.2 / 9.7 / 2.1 % | **oag-eng**: 929,315 (4), 12,346,854 (22), 3,956, Time 88/10/2 |

Table 1 = H2GB, arXiv 2407.10916v2, Table 1 "Statistics of H2GB datasets" (read from the arXiv HTML page,
2026-11-15; the paper has no per-type counts, its Figure 6 is an image that was not read). File sizes as the card
states (cs 3.65 GB, chemistry 2.96 GB, engineering 6.21 GB): the largest file holds the largest graph, Table 1's
oag-chem. Classes differ from Table 1 by one for both swapped fields (2,986 vs 2,985; 3,957 vs 3,956); the number
is `max(y) + 1` of H2GB's processed labels with every class present (`classes_present` equal), i.e. H2GB's own
count from its file; reported, not a stop (card: a count that differs from the paper while equal to H2GB's files).

Per type, stored graph (before `ToUndirected`):

| Folder | paper | author | field | affiliation | PP_cite | AP_write_first/last/other | PF_in_L0..L5 | FF_in | author–in–affiliation |
|---|---|---|---|---|---|---|---|---|---|
| `cs` | 546,704 | 511,122 | 45,775 | 9,090 | 5,872,552 | 456,903 / 431,027 / 664,266 | 546,831 / 1,202,487 / 2,347,239 / 869,729 / 304,912 / 203,338 | 262,839 | 614,161 |
| `engineering` (= chemistry) | 747,290 | 1,097,433 | 65,290 | 8,868 | 7,228,234 | 630,999 / 664,930 / 1,634,514 | 747,417 / 1,332,023 / 3,108,157 / 1,710,391 / 398,183 / 87,674 | 410,909 | 1,097,433 |
| `chemistry` (= engineering) | 370,624 | 504,888 | 44,714 | 9,089 | 1,054,651 | 288,440 / 283,131 / 402,095 | 370,755 / 686,302 / 1,569,093 / 577,152 / 140,354 / 58,205 | 240,326 | 504,888 |

No `venue` node type and no paper–venue relation in any of the three (B4 audits this).
sha256 of the processed files: `h2gb_nc_counts.md` (oag-cs `4ead7163…`, `OAG/engineering` `f69877c0…`,
`OAG/chemistry` `c9c64ab7…`).
