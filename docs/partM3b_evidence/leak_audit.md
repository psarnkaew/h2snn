# B4 — venue-leak audit (F18 style; decision of 2026-11-12)

**Result: pass on oag-cs, oag-eng and oag-chem.** No `venue` node type and no paper–venue relation is in the graph H2GB trains on (`dataset[0]`), in the relation list H2S-NN's `build_dataset` flattens from it, or in the baseline bundles (they are built by `build_dataset`, `src/baselines/bundles.py:395-405`). No relation fingerprints the venue label. The other five datasets (informational, one row each) have no edge type that carries the label.

## Method (rule fixed in `src/data/leak_audit.py` before the audit ran on real data)
Spec F18 (v1.10) is written for link prediction: an attribute inventory plus a fingerprint probe against a training-graph control. For a node label it reads:
1. **Inventory:** every node type, relation and node attribute; a name that marks the label source (`venue`, `PV_`; OAG labels are `PV_Journal` paper–venue edges, `H2GB/datasets/oag_dataset.py:138-150`) fails.
2. **Fingerprint probe:** for each relation with a `paper` end, every node at the other end (for paper–paper: the citing paper) with >= 2 labelled paper neighbours; *majority share* = Σ largest same-label count / Σ labelled-neighbour count over those nodes. A venue node has share 1.0 (one venue per paper). **Control:** the same statistic after a seeded permutation of the labels among labelled papers (chance level). A relation **carries the label** at share >= 0.95. The share is a homophily-type statistic, so values well above the control are expected (co-authors and citations share venues); only a near-deterministic relation is a leak.
3. Tests (`tests/test_leak_audit.py`): a venue relation left in the message graph is reported by name and by the probe (share 1.0); without it the audit passes (negative control); a label carrier under an innocent name is caught by the probe; the share on a hand-counted example.

Job 16771 on g3an01 (`b4_audit_job.sh`, log `b4_audit_16771.log`, output `b4_audit.json`), 37 s, peak RSS 10.3 GiB (sacct), 10.6 GiB (getrusage).

## oag-cs (target `paper`): PASS
- node types: ['paper', 'field', 'author', 'affiliation']; label-source names present: none
- attributes: {'paper': ['test_mask', 'train_mask', 'val_mask', 'x', 'y', 'year'], 'field': ['x'], 'author': ['x'], 'affiliation': ['x']} (`year` defines H2GB's split, `oag_dataset.py:151-153`; H2S-NN reads only `x`, `y` and the masks)
- relations H2S-NN flattens (22): `paper__PP_cite__paper`, `author__AP_write_last__paper`, `author__AP_write_other__paper`, `author__AP_write_first__paper`, `field__FF_in__field`, `paper__PF_in_L0__field`, `paper__PF_in_L3__field`, `paper__PF_in_L1__field`, `paper__PF_in_L2__field`, `paper__PF_in_L5__field`, `paper__PF_in_L4__field`, `author__in__affiliation`, `paper__rev_AP_write_last__author`, `paper__rev_AP_write_other__author`, `paper__rev_AP_write_first__author`, `field__rev_PF_in_L0__paper`, `field__rev_PF_in_L3__paper`, `field__rev_PF_in_L1__paper`, `field__rev_PF_in_L2__paper`, `field__rev_PF_in_L5__paper`, `field__rev_PF_in_L4__paper`, `affiliation__rev_in__author`

| relation | nodes with >= 2 labelled papers | majority share | control | carries label |
|---|---|---|---|---|
| `paper|PP_cite|paper` | 480,448 | 0.2921 | 0.1232 | no |
| `author|AP_write_last|paper` | 25,010 | 0.4234 | 0.2418 | no |
| `author|AP_write_other|paper` | 40,861 | 0.4520 | 0.2869 | no |
| `author|AP_write_first|paper` | 31,994 | 0.5082 | 0.3411 | no |
| `paper|PF_in_L0|field` | 10 | 0.0217 | 0.0216 | no |
| `paper|PF_in_L3|field` | 8,161 | 0.1693 | 0.0461 | no |
| `paper|PF_in_L1|field` | 257 | 0.0918 | 0.0223 | no |
| `paper|PF_in_L2|field` | 10,754 | 0.1070 | 0.0338 | no |
| `paper|PF_in_L5|field` | 3,405 | 0.1904 | 0.0679 | no |
| `paper|PF_in_L4|field` | 3,944 | 0.1849 | 0.0547 | no |
| `paper|rev_AP_write_last|author` | 25,010 | 0.4234 | 0.2418 | no |
| `paper|rev_AP_write_other|author` | 40,861 | 0.4520 | 0.2869 | no |
| `paper|rev_AP_write_first|author` | 31,994 | 0.5082 | 0.3411 | no |
| `field|rev_PF_in_L0|paper` | 10 | 0.0217 | 0.0216 | no |
| `field|rev_PF_in_L3|paper` | 8,161 | 0.1693 | 0.0461 | no |
| `field|rev_PF_in_L1|paper` | 257 | 0.0918 | 0.0223 | no |
| `field|rev_PF_in_L2|paper` | 10,754 | 0.1070 | 0.0338 | no |
| `field|rev_PF_in_L5|paper` | 3,405 | 0.1904 | 0.0679 | no |
| `field|rev_PF_in_L4|paper` | 3,944 | 0.1849 | 0.0547 | no |

Largest share: 0.5082 (< 0.95).

## oag-eng (target `paper`): PASS
- node types: ['paper', 'field', 'author', 'affiliation']; label-source names present: none
- attributes: {'paper': ['test_mask', 'train_mask', 'val_mask', 'x', 'y', 'year'], 'field': ['x'], 'author': ['x'], 'affiliation': ['x']} (`year` defines H2GB's split, `oag_dataset.py:151-153`; H2S-NN reads only `x`, `y` and the masks)
- relations H2S-NN flattens (22): `paper__PP_cite__paper`, `author__AP_write_first__paper`, `author__AP_write_other__paper`, `author__AP_write_last__paper`, `field__FF_in__field`, `paper__PF_in_L1__field`, `paper__PF_in_L5__field`, `paper__PF_in_L0__field`, `paper__PF_in_L2__field`, `paper__PF_in_L3__field`, `paper__PF_in_L4__field`, `author__in__affiliation`, `paper__rev_AP_write_first__author`, `paper__rev_AP_write_other__author`, `paper__rev_AP_write_last__author`, `field__rev_PF_in_L1__paper`, `field__rev_PF_in_L5__paper`, `field__rev_PF_in_L0__paper`, `field__rev_PF_in_L2__paper`, `field__rev_PF_in_L3__paper`, `field__rev_PF_in_L4__paper`, `affiliation__rev_in__author`

| relation | nodes with >= 2 labelled papers | majority share | control | carries label |
|---|---|---|---|---|
| `paper|PP_cite|paper` | 244,930 | 0.4250 | 0.1571 | no |
| `author|AP_write_first|paper` | 34,873 | 0.6038 | 0.3515 | no |
| `author|AP_write_other|paper` | 44,060 | 0.5523 | 0.3158 | no |
| `author|AP_write_last|paper` | 30,761 | 0.5249 | 0.2795 | no |
| `paper|PF_in_L1|field` | 271 | 0.1139 | 0.0211 | no |
| `paper|PF_in_L5|field` | 1,795 | 0.2527 | 0.0845 | no |
| `paper|PF_in_L0|field` | 10 | 0.0195 | 0.0194 | no |
| `paper|PF_in_L2|field` | 13,169 | 0.1383 | 0.0351 | no |
| `paper|PF_in_L3|field` | 10,490 | 0.2072 | 0.0521 | no |
| `paper|PF_in_L4|field` | 3,784 | 0.2447 | 0.0692 | no |
| `paper|rev_AP_write_first|author` | 34,873 | 0.6038 | 0.3515 | no |
| `paper|rev_AP_write_other|author` | 44,060 | 0.5523 | 0.3158 | no |
| `paper|rev_AP_write_last|author` | 30,761 | 0.5249 | 0.2795 | no |
| `field|rev_PF_in_L1|paper` | 271 | 0.1139 | 0.0211 | no |
| `field|rev_PF_in_L5|paper` | 1,795 | 0.2527 | 0.0845 | no |
| `field|rev_PF_in_L0|paper` | 10 | 0.0195 | 0.0194 | no |
| `field|rev_PF_in_L2|paper` | 13,169 | 0.1383 | 0.0351 | no |
| `field|rev_PF_in_L3|paper` | 10,490 | 0.2072 | 0.0521 | no |
| `field|rev_PF_in_L4|paper` | 3,784 | 0.2447 | 0.0692 | no |

Largest share: 0.6038 (< 0.95).

## oag-chem (target `paper`): PASS
- node types: ['paper', 'field', 'author', 'affiliation']; label-source names present: none
- attributes: {'paper': ['test_mask', 'train_mask', 'val_mask', 'x', 'y', 'year'], 'field': ['x'], 'author': ['x'], 'affiliation': ['x']} (`year` defines H2GB's split, `oag_dataset.py:151-153`; H2S-NN reads only `x`, `y` and the masks)
- relations H2S-NN flattens (22): `paper__PP_cite__paper`, `author__AP_write_last__paper`, `author__AP_write_other__paper`, `author__AP_write_first__paper`, `field__FF_in__field`, `paper__PF_in_L3__field`, `paper__PF_in_L2__field`, `paper__PF_in_L1__field`, `paper__PF_in_L0__field`, `paper__PF_in_L4__field`, `paper__PF_in_L5__field`, `author__in__affiliation`, `paper__rev_AP_write_last__author`, `paper__rev_AP_write_other__author`, `paper__rev_AP_write_first__author`, `field__rev_PF_in_L3__paper`, `field__rev_PF_in_L2__paper`, `field__rev_PF_in_L1__paper`, `field__rev_PF_in_L0__paper`, `field__rev_PF_in_L4__paper`, `field__rev_PF_in_L5__paper`, `affiliation__rev_in__author`

| relation | nodes with >= 2 labelled papers | majority share | control | carries label |
|---|---|---|---|---|
| `paper|PP_cite|paper` | 733,913 | 0.2692 | 0.0884 | no |
| `author|AP_write_last|paper` | 83,128 | 0.4041 | 0.1873 | no |
| `author|AP_write_other|paper` | 261,856 | 0.4306 | 0.2529 | no |
| `author|AP_write_first|paper` | 114,574 | 0.5324 | 0.3391 | no |
| `paper|PF_in_L3|field` | 15,515 | 0.1521 | 0.0471 | no |
| `paper|PF_in_L2|field` | 12,211 | 0.1113 | 0.0424 | no |
| `paper|PF_in_L1|field` | 242 | 0.1016 | 0.0392 | no |
| `paper|PF_in_L0|field` | 10 | 0.0390 | 0.0390 | no |
| `paper|PF_in_L4|field` | 10,617 | 0.2066 | 0.0642 | no |
| `paper|PF_in_L5|field` | 5,698 | 0.2719 | 0.1019 | no |
| `paper|rev_AP_write_last|author` | 83,128 | 0.4041 | 0.1873 | no |
| `paper|rev_AP_write_other|author` | 261,856 | 0.4306 | 0.2529 | no |
| `paper|rev_AP_write_first|author` | 114,574 | 0.5324 | 0.3391 | no |
| `field|rev_PF_in_L3|paper` | 15,515 | 0.1521 | 0.0471 | no |
| `field|rev_PF_in_L2|paper` | 12,211 | 0.1113 | 0.0424 | no |
| `field|rev_PF_in_L1|paper` | 242 | 0.1016 | 0.0392 | no |
| `field|rev_PF_in_L0|paper` | 10 | 0.0390 | 0.0390 | no |
| `field|rev_PF_in_L4|paper` | 10,617 | 0.2066 | 0.0642 | no |
| `field|rev_PF_in_L5|paper` | 5,698 | 0.2719 | 0.1019 | no |

Largest share: 0.5324 (< 0.95).

## The other datasets (informational; card B4)

| Dataset | node types | relations (loaded graph) | label source | relation carrying the label |
|---|---|---|---|---|
| ogbn-mag | paper, author, field_of_study, institution | author|affiliated_with|institution, institution|rev_affiliated_with|author, author|writes|paper, paper|rev_writes|author, paper|cites|paper, paper|has_topic|field_of_study, field_of_study|rev_has_topic|paper | paper venue (OGB `y_dict['paper']`, mag_dataset.py:128); no venue node type | none |
| mag-year | paper, author, field_of_study, institution | author|affiliated_with|institution, institution|rev_affiliated_with|author, author|writes|paper, paper|rev_writes|author, paper|cites|paper, paper|has_topic|field_of_study, field_of_study|rev_has_topic|paper | publication year in 5 quantile classes (mag_dataset.py:163-165); year is not a node attribute or node type | none |
| ieee-cis-g | transaction, card1, card2, card3, card4, card5, card6, ProductCD, addr1, addr2, P_emaildomain, R_emaildomain | transaction|to|card1, transaction|to|card2, transaction|to|card3, transaction|to|card4, transaction|to|card5, transaction|to|card6, transaction|to|ProductCD, transaction|to|addr1, transaction|to|addr2, transaction|to|P_emaildomain, transaction|to|R_emaildomain, card1|rev_to|transaction, card2|rev_to|transaction, card3|rev_to|transaction, card4|rev_to|transaction, card5|rev_to|transaction, card6|rev_to|transaction, ProductCD|rev_to|transaction, addr1|rev_to|transaction, addr2|rev_to|transaction, P_emaildomain|rev_to|transaction, R_emaildomain|rev_to|transaction | `isFraud` column of the transaction table (ieee_cis_dataset.py:227); edges are built from the identity columns card1–6, ProductCD, addr1–2, P/R_emaildomain (`DEFAULT_NON_TARGET_NODE_TYPES`), not from `isFraud` (an excluded column, `exclude_cols`, :146) | none |
| h-pokec | user, life_style, music, cars, politics, relationships, art_culture, hobbies_interests, science_technologies, computers_internet, education, sport, movies, travelling, health, companies_brands | user|has_friend|user, user|lists|life_style, user|lists|music, user|lists|cars, user|lists|politics, user|lists|relationships, user|lists|art_culture, user|lists|hobbies_interests, user|lists|science_technologies, user|lists|computers_internet, user|lists|education, user|lists|sport, user|lists|movies, user|lists|travelling, user|lists|health, user|lists|companies_brands, life_style|rev_lists|user, music|rev_lists|user, cars|rev_lists|user, politics|rev_lists|user, relationships|rev_lists|user, art_culture|rev_lists|user, hobbies_interests|rev_lists|user, science_technologies|rev_lists|user, computers_internet|rev_lists|user, education|rev_lists|user, sport|rev_lists|user, movies|rev_lists|user, travelling|rev_lists|user, health|rev_lists|user, companies_brands|rev_lists|user | `gender` profile column (pokec_dataset.py:280); edges are friendships and club memberships parsed from the profile's text columns (`process`, :210-307), gender is not one of them | none |
| pdns | domain_node, ip_node | apex, similar, resolves (B5) | malicious/benign tag of `domains2.csv` (B5) | none expected; checked in B5 |
