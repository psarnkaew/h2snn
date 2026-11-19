| node_f task | lines (host peak GiB) | known sum GiB (GB) | unknown |
|---|---|---|---|
| 1 | h2snn_node_classification_acm 1.68; h2snn_node_classification_dblp 2.02; h2snn_node_classification_imdb 1.88; h2snn_node_classification_freebase >= 13.91 | 19.49 (20.93) | 0 |
| 2 | h2snn_node_classification_ogbn-mag 9.75; h2snn_node_classification_mag-year 9.89; h2snn_node_classification_oag-cs 10.05; h2snn_node_classification_oag-eng 6.95 | 36.64 (39.34) | 0 |
| 3 | h2snn_node_classification_oag-chem 14.47; h2snn_node_classification_ieee-cis-g 6.38; h2snn_node_classification_h-pokec 39.79; h2snn_node_classification_pdns 14.32 | 74.96 (80.49) | 0 |
| 4 | h2snn_link_prediction_acm_pa 1.87; h2snn_link_prediction_dblp_ap >= 4.34; h2snn_link_prediction_imdb_ma 2.52; h2snn_link_prediction_freebase_bb >= 14.11 | 22.84 (24.52) | 0 |
| 5 | h2snn_link_prediction_lastfm >= 2.16; h2snn_link_prediction_pubmed >= 3.6; h2snn_link_prediction_ogbn-mag_pp unknown; h2snn_link_prediction_h-pokec_uu unknown | 5.76 (6.18) | 2 |
| 6 | h2snn_graph_classification_mutag 1.54; h2snn_graph_classification_enzymes 1.53; h2snn_graph_classification_nci1 1.51; h2snn_graph_classification_proteins 1.6 | 6.18 (6.64) | 0 |

Largest known sum: 80.49 GB; threshold 600 GB -> no reorder (card T7.5)
