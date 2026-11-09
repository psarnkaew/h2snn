# Heterophily of the node-classification graphs (J0.5)

Heterophily form H = 1 - h unless marked *homophily*. Conventions and their code sources: `j02_reference_code.md`. Computed values: `scripts/heterophily/compute_heterophily.py` (validated by the J0.4 gate). H_node follows PyG / Hetero^2Net: a target without in-edges in G_P counts as homophily 0, so H_node can exceed H_edge on sparse G_P.

## Summary

| Dataset | Graph | MLH, H2GB set, mean (H2GB) | MLH, max | H^2 index | MLH, Hetero^2Net set, *homophily* (ours / published) | Source |
|---|---|---|---|---|---|---|
| acm | as loaded by PyG HGBDataset | 0.2621 | 0.6662 | 0.3994 | 0.5962 / 0.60 (Hetero^2Net Table II) | computed |
| dblp | as loaded by PyG HGBDataset | 0.1278 | 0.1278 | 0.1732 | 0.8722 / 0.87 (Hetero^2Net Table II) | computed |
| imdb | as loaded by PyG HGBDataset | 0.7630 | 0.8661 | not computed (multilabel) | 0.2370 / 0.24 (Hetero^2Net Table II) | computed |
| freebase | as loaded by PyG HGBDataset | 0.8528 | 0.8528 | 0.9992 (fill seeds: 0.9987–0.9993) | — | computed |
| freebase | symmetrised as H2GB's loaders do (flagged) | 0.8464 | 0.8554 | 0.9953 (fill seeds: 0.9944–0.9956) | 0.1542 | computed |
| ogbn-mag | H2GB loader | 0.8731 | — | 0.8773 | — | H2GB Table 2 (cited) |
| mag-year | H2GB loader | 0.7718 | — | 0.9654 | — | H2GB Table 2 (cited) |
| IEEE-CIS-G | H2GB loader | 0.1352 | — | 0.9846 | — | H2GB Table 2 (cited) |
| H-Pokec | H2GB loader | 0.3922 | — | 0.9488 | — | H2GB Table 2 (cited) |

## acm (as loaded by PyG HGBDataset); target `paper`, 3025/3025 labeled

- MLH, H2GB set, **mean (H2GB's aggregation)**: 0.2621; max: 0.6662
- H^2 index (H2GB set, mean of H_adj): 0.3994
- MLH, Hetero^2Net set, homophily form: 0.5962
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

- MLH, H2GB set, **mean (H2GB's aggregation)**: 0.1278; max: 0.1278
- H^2 index (H2GB set, mean of H_adj): 0.1732
- MLH, Hetero^2Net set, homophily form: 0.8722
- Class counts (labeled): {'class_0': 1197, 'class_1': 745, 'class_2': 1109, 'class_3': 1006}


| Meta-path | pairs in G_P (loops incl.) | H_edge | H_node | H_adj | labeled-only H_edge / H_adj (diagnostic) |
|---|---|---|---|---|---|
| author -to-> paper -to-> author | 11,113 | 0.1278 | 0.0705 | 0.1732 | — |


## imdb (as loaded by PyG HGBDataset); target `movie`, 4573/4932 labeled

- MLH, H2GB set, **mean (H2GB's aggregation)**: 0.7630; max: 0.8661
- H^2 index (H2GB set, mean of H_adj): not computed (multilabel)
- MLH, Hetero^2Net set, homophily form: 0.2370
- Class counts (labeled): {'label_0': 1090, 'label_1': 1384, 'label_2': 1837, 'label_3': 1135, 'label_4': 2517, 'distinct_label_sets': 30}


| Meta-path | pairs in G_P (loops incl.) | H_edge | H_node | H_adj | labeled-only H_edge / H_adj (diagnostic) |
|---|---|---|---|---|---|
| movie -to-> director -to-> movie | 21,282 | 0.5956 | 0.3851 | — | 0.5842 / — |
| movie ->actorh-> actor -to-> movie | 99,782 | 0.8274 | 0.6952 | — | 0.8220 / — |
| movie -to-> keyword -to-> movie | 403,848 | 0.8661 | 0.8414 | — | 0.8576 / — |

- Note: Multilabel: calcHomophily cannot run; h_edge and h_node from Hetero^2Net (exact label-vector match) on G_P built as calcHomophily builds it, with PyG's all-zero rows for unlabeled movies; H_adj and H^2 not computed (undefined in both codebases).

## freebase (as loaded by PyG HGBDataset); target `book`, 7954/40402 labeled

- MLH, H2GB set, **mean (H2GB's aggregation)**: 0.8528; max: 0.8528
- H^2 index (H2GB set, mean of H_adj): 0.9992
- MLH, Hetero^2Net set, homophily form: none (empty set)
- Class counts (labeled): {'class_0': 429, 'class_1': 3886, 'class_2': 1748, 'class_3': 185, 'class_4': 962, 'class_5': 359, 'class_6': 385}

- Over fill seeds [0, 1, 2, 3, 4]: H^2 0.9992, 0.9989, 0.9993, 0.9987, 0.9988; MLH 0.8528, 0.8527, 0.8532, 0.8533, 0.8528

| Meta-path | pairs in G_P (loops incl.) | H_edge | H_node | H_adj | labeled-only H_edge / H_adj (diagnostic) |
|---|---|---|---|---|---|
| book -and-> book -and-> book | 49,275,062 | 0.8528 | 0.7730 | 0.9992 | 0.5500 / 0.8923 |

- Note: 32448 unlabeled targets: H2GB fills them with random classes (calcHomophily.py:110-113); fill seeds [0, 1, 2, 3, 4]; reported values use seed 0; H_node uses the same filled labels.

## freebase (symmetrised as H2GB's loaders do (flagged)); target `book`, 7954/40402 labeled

- MLH, H2GB set, **mean (H2GB's aggregation)**: 0.8464; max: 0.8554
- H^2 index (H2GB set, mean of H_adj): 0.9953
- MLH, Hetero^2Net set, homophily form: 0.1542
- Class counts (labeled): {'class_0': 429, 'class_1': 3886, 'class_2': 1748, 'class_3': 185, 'class_4': 962, 'class_5': 359, 'class_6': 385}

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
