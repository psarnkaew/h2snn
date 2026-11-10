## Test macro-F1 (mean ± sample std over seeds)
| dataset | row | n | test macro-F1 | test micro-F1 | val macro-F1 | best epoch |
|---|---|---|---|---|---|---|
| acm | full | 3 | 85.09 ± 1.67 | 85.08 ± 1.51 | 88.94 | 39 |
| acm | gamma0 | 3 | 86.26 ± 1.16 | 86.21 ± 1.16 | 88.78 | 32 |
| acm | endpoint_edges | 3 | 85.88 ± 0.89 | 85.82 ± 0.96 | 90.49 | 35 |
| acm | h0_only | 3 | 83.63 ± 0.17 | 83.70 ± 0.17 | 86.78 | 67 |
| acm | pair_attention | 3 | 87.34 ± 1.16 | 87.32 ± 1.13 | 91.55 | 17 |
| dblp | full | 1 | 82.18 ± nan | 83.24 ± nan | 82.63 | 22 |
| dblp | gamma0 | 1 | 81.93 ± nan | 82.82 ± nan | 81.98 | 17 |
| dblp | endpoint_edges | 1 | 81.02 ± nan | 81.80 ± nan | 78.95 | 21 |
| dblp | h0_only | 1 | 76.98 ± nan | 77.71 ± nan | 79.22 | 28 |

## Pairwise relations (a beats b iff mean diff > max std)
| dataset | full vs gamma0 | full vs endpoint_edges | full vs h0_only | pair_attention vs full |
|---|---|---|---|---|
| acm | ~ | ~ | ~ | beats |
| dblp | n/a | n/a | n/a | n/a |
| imdb | n/a | n/a | n/a | n/a |

## Triggers (plan 12.6, rules fixed in pre_registration.md)
T1 path cells do not beat gamma0 and endpoint_edges (on >= 2 of 3): HIT -> STOP
T2 full ~ h0_only on >= 2 datasets: not hit
T3 path cells win and full beats h0_only on >= 2 datasets: not hit
T4 gates do not move (mean max |gate mean - 0.6| = 0.246): not hit
T5 pair_attention beats full on >= 2 datasets: not hit
T6 otherwise A4 retained: yes