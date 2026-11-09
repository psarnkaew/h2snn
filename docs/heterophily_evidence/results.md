# Heterophily of the node-classification graphs (J0.5, conventions of J4.1)

Heterophily form H = 1 - h unless marked *homophily*. Conventions: spec v1.8 §1 (J4.1, `j4_preregistration.md`) on the reference code recorded in `j02_reference_code.md`. Computed values: `scripts/heterophily/compute_heterophily.py` (validated by the J0.4 gate). H_node follows PyG / Hetero^2Net: a target without in-edges in G_P counts as homophily 0, so H_node can exceed H_edge on sparse G_P.

## Summary (primary convention)

| Dataset | Graph | Convention | MLH, H2GB M_2, mean | H^2 index | Labeled-only diagnostic MLH / H^2 | Reproduction: MLH, Hetero^2Net set, *homophily* (ours / published) | Source |
|---|---|---|---|---|---|---|---|
| acm | as loaded by PyG HGBDataset | single label, fully labeled; H2GB calcHomophily on its own M_2, mean | 0.2621 | 0.3994 | — (fully labeled) | 0.5962 / 0.60 (Hetero^2Net Table II) | computed |
| dblp | as loaded by PyG HGBDataset | single label, fully labeled; H2GB calcHomophily on its own M_2, mean | 0.1278 | 0.1732 | — (fully labeled) | 0.8722 / 0.87 (Hetero^2Net Table II) | computed |
| imdb | as loaded by PyG HGBDataset | multilabel per genre (5 binary labels), H2GB calcHomophily per genre with its fill (seed 100*s+g, s = 0), mean over genres and over M_2 | 0.3247 (fill seeds: 0.3243–0.3255) | 0.7373 (fill seeds: 0.7366–0.7393) | 0.3120 / 0.7181 | 0.2370 / 0.24 (Hetero^2Net Table II) (exact label-vector match) | computed |
| freebase | symmetrised as H2GB's loaders build it (primary) | single label; H2GB calcHomophily on its own M_2, mean; unlabeled targets random-filled by calcHomophily (seed 0) | 0.8464 (fill seeds: 0.8453–0.8466) | 0.9953 (fill seeds: 0.9944–0.9956) | 0.4775 / 0.8821 | 0.1542 | computed |
| freebase | as loaded by PyG HGBDataset (flagged comparison) | single label; H2GB calcHomophily on its own M_2, mean; unlabeled targets random-filled by calcHomophily (seed 0) | 0.8528 (fill seeds: 0.8527–0.8533) | 0.9992 (fill seeds: 0.9987–0.9993) | 0.5500 / 0.8923 | — | computed |
| ogbn-mag | H2GB loader | H2GB's | 0.8731 | 0.8773 | — | — | H2GB Table 2 (cited) |
| mag-year | H2GB loader | H2GB's | 0.7718 | 0.9654 | — | — | H2GB Table 2 (cited) |
| IEEE-CIS-G | H2GB loader | H2GB's | 0.1352 | 0.9846 | — | — | H2GB Table 2 (cited) |
| H-Pokec | H2GB loader | H2GB's | 0.3922 | 0.9488 | — | — | H2GB Table 2 (cited) |

## acm (as loaded by PyG HGBDataset); target `paper`, 3025/3025 labeled

- Convention: single label, fully labeled; H2GB calcHomophily on its own M_2, mean
- MLH, H2GB M_2, **mean (H2GB's aggregation)**: 0.2621; max: 0.6662
- H^2 index (H2GB M_2, mean of H_adj): 0.3994
- Reproduction row, MLH, Hetero^2Net set, homophily form: 0.5962
- Class counts (labeled): {'class_0': 994, 'class_1': 970, 'class_2': 1061}

| Meta-path | pairs in G_P (loops incl.) | H_edge | H_node | H_adj | labeled-only H_edge / H_adj (diagnostic) |
|---|---|---|---|---|---|
| paper -cite-> paper -cite-> paper | 9,402 | 0.1566 | 0.7243 | 0.2474 | — |
| paper -cite-> paper -ref-> paper | 44,587 | 0.1867 | 0.4237 | 0.2848 | — |
| paper -ref-> paper -cite-> paper | 14,540 | 0.1234 | 0.5697 | 0.1920 | — |
| paper -ref-> paper -ref-> paper | 9,402 | 0.1566 | 0.5687 | 0.2465 | — |
| paper -to-> author -to-> paper | 29,767 | 0.1855 | 0.1005 | 0.2819 | — |
| paper -to-> subject -to-> paper | 2,217,089 | 0.3597 | 0.3373 | 0.5432 | — |
| paper -to-> term -to-> paper | 9,150,595 | 0.6662 | 0.6662 | 1.0000 | — |


## dblp (as loaded by PyG HGBDataset); target `author`, 4057/4057 labeled

- Convention: single label, fully labeled; H2GB calcHomophily on its own M_2, mean
- MLH, H2GB M_2, **mean (H2GB's aggregation)**: 0.1278; max: 0.1278
- H^2 index (H2GB M_2, mean of H_adj): 0.1732
- Reproduction row, MLH, Hetero^2Net set, homophily form: 0.8722
- Class counts (labeled): {'class_0': 1197, 'class_1': 745, 'class_2': 1109, 'class_3': 1006}

| Meta-path | pairs in G_P (loops incl.) | H_edge | H_node | H_adj | labeled-only H_edge / H_adj (diagnostic) |
|---|---|---|---|---|---|
| author -to-> paper -to-> author | 11,113 | 0.1278 | 0.0705 | 0.1732 | — |


## imdb (as loaded by PyG HGBDataset); target `movie`, 4573/4932 labeled

- Convention: multilabel per genre (5 binary labels), H2GB calcHomophily per genre with its fill (seed 100*s+g, s = 0), mean over genres and over M_2
- MLH, H2GB M_2, **mean (H2GB's aggregation)**: 0.3247; max: 0.3753
- H^2 index (H2GB M_2, mean of H_adj): 0.7373
- Reproduction row, MLH, Hetero^2Net set, homophily form: 0.2370
- Class counts (labeled): {'label_0': 1090, 'label_1': 1384, 'label_2': 1837, 'label_3': 1135, 'label_4': 2517, 'distinct_label_sets': 30}
- Labeled-only diagnostic (pairs with both endpoints labeled; never primary): MLH 0.3120, H^2 0.7181
- Over fill seeds [0, 1, 2, 3, 4]: H^2 0.7373, 0.7379, 0.7393, 0.7368, 0.7366; MLH 0.3247, 0.3249, 0.3255, 0.3244, 0.3243
- Reproduction row, exact label-vector match on H2GB M_2 (heterophily): 0.7630

| Meta-path | pairs in G_P (loops incl.) | H_edge | H_node | H_adj | labeled-only H_edge / H_adj (diagnostic) |
|---|---|---|---|---|---|
| movie -to-> director -to-> movie | 21,282 | 0.2315 | 0.1469 | 0.5308 | 0.2141 / 0.5008 |
| movie ->actorh-> actor -to-> movie | 99,782 | 0.3672 | 0.3062 | 0.8398 | 0.3584 / 0.8295 |
| movie -to-> keyword -to-> movie | 403,848 | 0.3753 | 0.3838 | 0.8413 | 0.3635 / 0.8239 |

### Per genre (heterophily form; mean over M_2)

| Genre | positives (labeled) | fill seeds | MLH | H^2 | H^2 over fill seeds | labeled-only MLH / H^2 |
|---|---|---|---|---|---|---|
| 0 | 1090 | [0, 100, 200, 300, 400] | 0.3054 | 0.7871 | 0.7871, 0.7894, 0.7919, 0.7917, 0.7895 | 0.2910 / 0.7712 |
| 1 | 1384 | [1, 101, 201, 301, 401] | 0.3290 | 0.7379 | 0.7379, 0.7401, 0.7392, 0.7350, 0.7351 | 0.3156 / 0.7187 |
| 2 | 1837 | [2, 102, 202, 302, 402] | 0.3269 | 0.6814 | 0.6814, 0.6813, 0.6841, 0.6818, 0.6825 | 0.3154 / 0.6589 |
| 3 | 1135 | [3, 103, 203, 303, 403] | 0.2838 | 0.7151 | 0.7151, 0.7149, 0.7184, 0.7142, 0.7146 | 0.2690 / 0.6933 |
| 4 | 2517 | [4, 104, 204, 304, 404] | 0.3781 | 0.7650 | 0.7650, 0.7636, 0.7629, 0.7613, 0.7613 | 0.3690 / 0.7482 |

### Appendix: Jaccard label homophily (Zhao et al., arXiv 2304.10398, Definition 1; *homophily* form; labeled-only pairs)

Mean over M_2: **0.4703**

| Meta-path | labeled pairs | Jaccard | exact match | per-genre agreement |
|---|---|---|---|---|
| movie -to-> director -to-> movie | 19,327 | 0.6191 | 0.4158 | 0.7793, 0.7698, 0.8274, 0.8005, 0.7526 |
| movie ->actorh-> actor -to-> movie | 93,039 | 0.4001 | 0.1780 | 0.6843, 0.6256, 0.6511, 0.6662, 0.5809 |
| movie -to-> keyword -to-> movie | 366,437 | 0.3917 | 0.1424 | 0.6636, 0.6578, 0.5753, 0.7262, 0.5593 |

- Note: Per genre (spec v1.8 §1): calcHomophily on 1(g in Y_v) for each of 5 genres; 359 unlabeled movies filled by calcHomophily with seed 100*s+g; primary s = 0; MLH and H^2 = means over genres and over M_2.

## freebase (symmetrised as H2GB's loaders build it (primary)); target `book`, 7954/40402 labeled

- Convention: single label; H2GB calcHomophily on its own M_2, mean; unlabeled targets random-filled by calcHomophily (seed 0)
- MLH, H2GB M_2, **mean (H2GB's aggregation)**: 0.8464; max: 0.8554
- H^2 index (H2GB M_2, mean of H_adj): 0.9953
- Reproduction row, MLH, Hetero^2Net set, homophily form: 0.1542
- Class counts (labeled): {'class_0': 429, 'class_1': 3886, 'class_2': 1748, 'class_3': 185, 'class_4': 962, 'class_5': 359, 'class_6': 385}
- Labeled-only diagnostic (pairs with both endpoints labeled; never primary): MLH 0.4775, H^2 0.8821
- Over fill seeds [0, 1, 2, 3, 4]: H^2 0.9953, 0.9948, 0.9948, 0.9944, 0.9956; MLH 0.8464, 0.8453, 0.8458, 0.8459, 0.8466

| Meta-path | pairs in G_P (loops incl.) | H_edge | H_node | H_adj | labeled-only H_edge / H_adj (diagnostic) |
|---|---|---|---|---|---|
| book -and-> book -and-> book | 77,209,334 | 0.8512 | 0.7895 | 0.9985 | 0.5133 / 0.9061 |
| book -to-> film -rev_to-> book | 4,193,476 | 0.8517 | 0.9512 | 0.9961 | 0.5264 / 0.8384 |
| book -on-> sports -rev_on-> book | 14,405,128 | 0.8335 | 0.9777 | 0.9997 | 0.0469 / 0.9927 |
| book -on-> location -rev_on-> book | 37,884,185 | 0.8527 | 0.9496 | 0.9996 | 0.5888 / 0.9706 |
| book -about-> organization -rev_about-> book | 54,932,925 | 0.8546 | 0.9458 | 0.9995 | 0.7640 / 0.9896 |
| book -rev_in-> music -in-> book | 84,404 | 0.8200 | 0.9888 | 0.9728 | 0.4741 / 0.7852 |
| book -rev_to-> people -to-> book | 3,145,804 | 0.8524 | 0.9408 | 0.9967 | 0.2610 / 0.6050 |
| book -rev_about-> business -about-> book | 35,046,698 | 0.8554 | 0.9496 | 0.9998 | 0.6458 / 0.9694 |

- Note: 32448 unlabeled targets: H2GB fills them with random classes (calcHomophily.py:110-113); fill seeds [0, 1, 2, 3, 4]; reported values use seed 0; H_node uses the same filled labels.

## freebase (as loaded by PyG HGBDataset (flagged comparison)); target `book`, 7954/40402 labeled

- Convention: single label; H2GB calcHomophily on its own M_2, mean; unlabeled targets random-filled by calcHomophily (seed 0)
- MLH, H2GB M_2, **mean (H2GB's aggregation)**: 0.8528; max: 0.8528
- H^2 index (H2GB M_2, mean of H_adj): 0.9992
- Reproduction row, MLH, Hetero^2Net set, homophily form: none (empty set)
- Class counts (labeled): {'class_0': 429, 'class_1': 3886, 'class_2': 1748, 'class_3': 185, 'class_4': 962, 'class_5': 359, 'class_6': 385}
- Labeled-only diagnostic (pairs with both endpoints labeled; never primary): MLH 0.5500, H^2 0.8923
- Over fill seeds [0, 1, 2, 3, 4]: H^2 0.9992, 0.9989, 0.9993, 0.9987, 0.9988; MLH 0.8528, 0.8527, 0.8532, 0.8533, 0.8528

| Meta-path | pairs in G_P (loops incl.) | H_edge | H_node | H_adj | labeled-only H_edge / H_adj (diagnostic) |
|---|---|---|---|---|---|
| book -and-> book -and-> book | 49,275,062 | 0.8528 | 0.7730 | 0.9992 | 0.5500 / 0.8923 |

- Note: 32448 unlabeled targets: H2GB fills them with random classes (calcHomophily.py:110-113); fill seeds [0, 1, 2, 3, 4]; reported values use seed 0; H_node uses the same filled labels.
