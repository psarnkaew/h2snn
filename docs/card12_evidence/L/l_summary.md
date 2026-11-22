| task | head / tail | features head / tail | other rel. head / tail | val pos deg0 tail / head | val neg deg0 tail / head | AUROC all: train / val / test-2hop / test-rand | AUROC target pair: train / val | (i) | (ii) | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| acm_pa | paper / author | yes / no (Sequential) | 3 / 0 | 0.470 / 0.000 | 0.124 / 0.000 | 0.6844 / 0.4482 / 0.3203 / 0.4578 | 0.6851 / 0.4696 | yes | yes | **FLAGGED** |
| dblp_ap | author / paper | no (Sequential) / yes | 0 / 2 | 0.000 / 0.090 | 0.000 / 0.090 | 0.5148 / 0.4855 / nan / 0.4808 | 0.5519 / 0.4200 | no | yes | **no** |
| imdb_ma | movie / actor | yes / no (Sequential) | 2 / 0 | 0.289 / 0.000 | 0.123 / 0.000 | 0.7554 / 0.5986 / nan / 0.5981 | 0.7591 / 0.5931 | yes | no | **no** |
| lastfm | user / artist | no (NoneType) / no (NoneType) | 1 / 1 | 0.116 / 0.000 | 0.146 / 0.000 | 0.8615 / 0.8024 / 0.5679 / 0.8188 | 0.8775 / 0.7944 | no | no | **no** |
| pubmed | disease / disease | yes / yes | 3 / 3 | 0.065 / 0.065 | 0.005 / 0.065 | 0.7263 / 0.7575 / 0.7459 / 0.7952 | 0.7426 / 0.7431 | no | no | **no** |
| freebase_bb | book / book | no (NoneType) / no (NoneType) | 7 / 7 | 0.027 / 0.026 | 0.024 / 0.026 | 0.6904 / 0.6675 / 0.5992 / 0.6647 | 0.6951 / 0.6660 | no | no | **no** |
| h-pokec_uu | user / user | yes / yes | 15 / 15 | 0.009 / 0.000 | 0.021 / 0.000 | 0.7000 / 0.6905 / 0.4405 / 0.6905 | 0.7031 / 0.6929 | no | no | **no** |
| ogbn-mag_pp | paper / paper | yes / yes | 2 / 2 | 0.000 / 0.000 | 0.000 / 0.000 | 0.6257 / 0.6153 / 0.6150 / 0.6151 | 0.6708 / 0.6548 | no | no | **no** |
