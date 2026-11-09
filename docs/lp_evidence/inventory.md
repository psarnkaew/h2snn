# J1.1 Link-prediction inventory (2026-11-09)
**Command:** `python ../docs/lp_evidence/j11_inventory.py`, run from `H2S-NN/` → `inventory.json` and `j11_inventory_output.txt`. The run's facts are VERIFIED-RUN.
- Every file is read through the link-prediction loader's own readers.
- ρ and the facts after tying (D7) come from `src/data/fact_layer.py` (`inverse_relations_typed`, `build_fact_layer`, `fact_mask`). This is the same ρ the loader applies at `link_prediction_dataset.py:351-352`.

## Raw files read (HGB node-classification releases; md5 in `inventory.json`)
- `H2S-NN/data/acm/acm/raw/ACM/`
- `H2S-NN/data/dblp/dblp/raw/DBLP/`
- `H2S-NN/data/imdb/imdb/raw/IMDB/`
- `H2S-NN/data/freebase/freebase/raw/Freebase/`

Files used: `info.dat`, `node.dat`, `link.dat`, `label.dat`, `label.dat.test`.

## Format the loader reads (VERIFIED-CODE `src/data/link_prediction_dataset.py`); the NC releases conform
**`node.dat`, read by `_load_local_hgb_nodes` (`:187-242`).**
- Each row is split on tabs into 3 or 4 fields (`:198-200`): id = field 0 (`:201`), type = field 2 (`:202`), comma-separated float features = field 3 (`:204-207`, `:180-184`). The name (field 1) is ignored.
- Local ids are the rank of the raw id within its type (`:217-221`).
- A type with features on only some nodes raises an error (`:224-231`).
- **All four releases conform** (VERIFIED-RUN, `awk` over `node.dat`): every row has 3 or 4 fields, and each type has features on all of its nodes or on none. Feature widths:

| Dataset | Feature widths |
|---|---|
| ACM | paper, author and subject 1,902; term none |
| DBLP | author 334, paper 4,231, term 50; venue none |
| IMDB | movie 3,489, director and actor 3,341; keyword none |
| Freebase | no features |

**`link.dat` / `link.dat.test`, read by `_read_local_hgb_link_file` (`:245-282`).**
- Each row has 4 tab fields (`:254-256`): src, dst, relation id (`:257-259`). **The 4th column is ignored.**
- The relation is named `rel_<id>` (`:176-177`), its endpoint types come from `node.dat` (`:260-264`), and they must be the same for every row of that id (`:265-270`).
- Rows are de-duplicated per relation (`:281`).
- Every `link.dat` row of the four releases has 4 fields (`every_link_row_has_4_columns: true`).

**Split as the loader does it.**
- `link.dat.test` is the test set (`:342`).
- Validation is carved from `link.dat` per relation by `_split_edge_index_hgb_by_head` (`:366-370`), with ratio `link_val_ratio` and seed `acm_split_seed + relation_id · 9973` (`:369`).
- A relation tied to a lower-id partner takes its partner's split, flipped (`:357-378`). (J1.3 checks the seed.)

## Relations (stored distinct edges → facts after tying; ρ = exact inverse)
| Dataset | id | Relation (info.dat) | Endpoints | Stored | ρ | Facts after tying |
|---|---|---|---|---|---|---|
| ACM | 0 | paper-cite-paper | paper→paper | 5,343 (8 loops) | 1 | 5,343 |
| | 1 | paper-ref-paper | paper→paper | 5,343 | 0 | 0 (tied to 0) |
| | **2** | **paper-author** | paper→author | **9,949** | 3 | **9,949** |
| | 3 | author-paper | author→paper | 9,949 | 2 | 0 |
| | 4 | paper-subject | paper→subject | 3,025 | 5 | 3,025 |
| | 5 | subject-paper | subject→paper | 3,025 | 4 | 0 |
| | 6 | paper-term | paper→term | 255,619 | 7 | 255,619 |
| | 7 | term-paper | term→paper | 255,619 | 6 | 0 |
| DBLP | **0** | **author-paper** | author→paper | **19,645** | 3 | **19,645** |
| | 1 | paper-term | paper→term | 85,810 | 4 | 85,810 |
| | 2 | paper-venue | paper→venue | 14,328 | 5 | 14,328 |
| | 3, 4, 5 | the reverses | | 19,645 / 85,810 / 14,328 | 0 / 1 / 2 | 0 |
| IMDB | 0 | movie->director | movie→director | 4,932 | 1 | 4,932 |
| | **2** | **movie->actorh** (typo in HGB's `info.dat`) | movie→actor | **14,779** | 3 | **14,779** |
| | 4 | movie->keyword | movie→keyword | 23,610 | 5 | 23,610 |
| | 1, 3, 5 | the reverses | | 4,932 / 14,779 / 23,610 | 0 / 2 / 4 | 0 |

**Totals.** Facts before → after: ACM 547,872 → 273,936; DBLP 239,566 → 119,783; IMDB 86,642 → 43,321; Freebase 1,057,688 → 869,706.

**Target relations for J1.2 (bold):**
- ACM paper–author: relation 2, 9,949 facts.
- DBLP author–paper: relation 0, 19,645 facts.
- IMDB movie–actor: relation 2, 14,779 facts.

In each, the reverse relation stores the same facts, so a held-out fact must be removed in both orientations: from relations 2 and 3 (ACM), 0 and 3 (DBLP), 2 and 3 (IMDB).

## Freebase: relations only (the author chooses the target; no task is built in this card)
**36 relations; none has a separately stored reverse relation.**
- 29 have no inverse (ρ = none), so every stored edge is a fact.
- The same-type relations FILM, MUSIC, SPORTS, PEOPLE, LOCATION, ORGANIZATION and BUSINESS "-and-" are **self-inverse** (ρ = itself): one fact per unordered pair.
- **BOOK-and-BOOK is not:** it has 202,674 edges, 15,036 self-loops and no inverse.

| id | Relation | Edges → facts | | id | Relation | Edges → facts |
|---|---|---|---|---|---|---|
| 0 | BOOK-and-BOOK | 202,674 → 202,674 (15,036 loops) | | 18 | PEOPLE-and-PEOPLE | 22,813 → 11,502 (self-inverse) |
| 1 | BOOK-to-FILM | 38,299 | | 19 | PEOPLE-on-LOCATION | 15,134 |
| 2 | BOOK-on-SPORTS | 6,615 | | 20 | PEOPLE-in-ORGANIZATION | 2,215 |
| 3 | BOOK-on-LOCATION | 26,921 | | 21 | PEOPLE-in-BUSINESS | 5,378 |
| 4 | BOOK-about-ORGANIZATION | 21,900 | | 22 | LOCATION-in-FILM | 21,299 |
| 5 | FILM-and-FILM | 87,838 → 44,113 (self-inverse) | | 23 | LOCATION-and-LOCATION | 47,817 → 25,146 (self-inverse) |
| 6 | MUSIC-in-BOOK | 31,486 | | 24 | ORGANIZATION-in-FILM | 13,128 |
| 7 | MUSIC-in-FILM | 11,291 | | 25 | ORGANIZATION-to-MUSIC | 10,702 |
| 8 | MUSIC-and-MUSIC | 283,670 → 176,398 (self-inverse) | | 26 | ORGANIZATION-to-SPORTS | 559 |
| 9 | MUSIC-for-SPORTS | 8,975 | | 27 | ORGANIZATION-on-LOCATION | 2,696 |
| 10 | MUSIC-on-LOCATION | 42,915 | | 28 | ORGANIZATION-and-ORGANIZATION | 1,101 → 766 (self-inverse) |
| 11 | SPORTS-in-FILM | 6,763 | | 29 | ORGANIZATION-for-BUSINESS | 1,073 |
| 12 | SPORTS-and-SPORTS | 1,290 → 653 (self-inverse) | | 30 | BUSINESS-about-BOOK | 18,625 |
| 13 | SPORTS-on-LOCATION | 656 | | 31 | BUSINESS-about-FILM | 8,397 |
| 14 | PEOPLE-to-BOOK | 35,587 | | 32 | BUSINESS-about-MUSIC | 24,764 |
| 15 | PEOPLE-to-FILM | 17,604 | | 33 | BUSINESS-about-SPORTS | 610 |
| 16 | PEOPLE-to-MUSIC | 10,948 | | 34 | BUSINESS-on-LOCATION | 6,647 |
| 17 | PEOPLE-to-SPORTS | 14,850 | | 35 | BUSINESS-and-BUSINESS | 4,448 → 2,417 (self-inverse) |

**For the author's choice** (facts only, no recommendation is binding):
- **Only books carry labels** (7,954 of 40,402), so task-level `H_edge` is measurable only for BOOK-and-BOOK (book–book).
- **BOOK-and-BOOK is not self-inverse** and holds 15,036 self-loops. A split by fact would treat (u, v) and (v, u) as different facts, which is the D7 rule.
- **Book–X relations with both endpoint types featureless** (Freebase has no features): BOOK-to-FILM 38,299; PEOPLE-to-BOOK 35,587; MUSIC-in-BOOK 31,486; BOOK-on-LOCATION 26,921.
