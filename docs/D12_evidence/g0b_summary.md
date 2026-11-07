## G0b ladder (test macro-F1 at best val macro-F1; mean ± sample std over seeds)
| rung | dataset | n | test macro-F1 | val macro-F1 | train macro-F1 (eval mode) | best epoch |
|---|---|---|---|---|---|---|
| L0 | acm | 5 | 89.72 ± 0.52 | 90.16 | 99.84 | 37 |
| L0 | dblp | 5 | 91.56 ± 1.37 | 91.41 | 100.00 | 75 |
| L1 | acm | 3 | 92.23 ± 0.16 | 95.93 | 95.69 | 158 |
| L1 | dblp | 3 | 80.22 ± 0.18 | 79.58 | 93.78 | 177 |
| L2raw | acm | 3 | 83.63 ± 0.17 | 86.78 | 100.00 | 67 |
| L2raw | dblp | 3 | 77.76 ± 0.86 | 78.53 | 94.33 | 42 |
| L2row | acm | 3 | 83.46 ± 1.24 | 88.32 | 98.53 | 32 |
| L2row | dblp | 3 | 77.52 ± 0.56 | 78.65 | 96.84 | 60 |
| L3 | acm | 3 | 84.46 ± 0.54 | 87.55 | 99.95 | 20 |
| L3 | dblp | 3 | 81.73 ± 0.50 | 83.15 | 97.47 | 54 |
| L4 | acm | 3 | 85.09 ± 1.67 | 88.94 | 100.00 | 39 |
| L4 | dblp | 3 | 83.05 ± 0.78 | 84.15 | 99.47 | 19 |

## Reading rules (g0b_preregistration.md)
acm: (i) L2row > L2raw + 2: False (83.46 vs 83.63)
acm: (ii) |L3 - L0| <= 2 and L4 < L3 - 2: False (L0 89.72, L3 84.46, L4 85.09)
acm: (iii) L3 < L0 - 2: True overfitting (train >= 98, val more than 2 below L0's) (L3 train 99.95, val 87.55; L0 val 90.16)
acm: (iv) L1 < band lower edge 90.61: False (L1 92.23)
dblp: (i) L2row > L2raw + 2: False (77.52 vs 77.76)
dblp: (ii) |L3 - L0| <= 2 and L4 < L3 - 2: False (L0 91.56, L3 81.73, L4 83.05)
dblp: (iii) L3 < L0 - 2: True unlabelled (train in [90, 98) or val within 2 of L0) (L3 train 97.47, val 83.15; L0 val 91.41)
dblp: (iv) L1 < band lower edge 87.33: True (L1 80.22)