# H2GB NC datasets as H2GB loads them (Part M3b B2)

Loaded through `src/data/h2gb_nc.py` (H2GB's own `preformat_*`, `dataset[0]`), by `scripts/data/h2gb_nc_counts.py` in SLURM jobs; merged by `scripts/data/h2gb_nc_counts_table.py`. Table 1 = H2GB, arXiv 2407.10916v2, Table 1 (read from the arXiv HTML page, 2026-11-15). `stored` = the processed file; `loaded` = after H2GB's transform.

| Dataset | nodes (types) loaded | Table 1 | edges (types) stored | edges (types) loaded | Table 1 | target feat | Table 1 | classes | Table 1 | train / val / test | load s | peak RSS GiB (sacct) | job, node |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ogbn-mag | 1,939,743 (4) | 1,939,743 (4) | 42,182,144 (7) | 42,182,144 (7) | 42,182,144 (7) | 128 | 128 | 349 | 349 | 629571 / 64879 / 41939 | 32.4 | 5.3 | 16756, g3an01 |
| mag-year | 1,939,743 (4) | 1,939,743 (4) | 42,182,144 (7) | 42,182,144 (7) | 42,182,144 (7) | 128 | 128 | 5 | 5 | 368194 / 184097 / 184098 | 32.3 | 5.3 | 16757, g3an01 |
| oag-cs | 1,112,691 (4) | 1,112,691 (4) | 13,776,284 (12) | 27,537,448 (22) | 27,537,448 (22) | 768 | 768 | 3514 | 3514 | 182310 / 19551 / 27243 | 42.1 | 8.2 | 16758, g3an01 |
| oag-eng | 929,315 (4) | 929,315 (4) | 6,175,392 (12) | 12,346,854 (22) | 12,346,854 (22) | 768 | 768 | 3957 | 3956 | 259444 / 28524 / 6082 | 10.7 | 3.4 | 16764, g3an01 |
| oag-chem | 1,918,881 (4) | 1,918,881 (4) | 19,050,864 (12) | 38,098,014 (22) | 38,098,014 (22) | 768 | 768 | 2986 | 2985 | 676738 / 59664 / 10651 | 11.6 | 7.4 | 16765, g3an01 |
| ieee-cis-g | 153,880 (12) | 153,880 (12) | 1,436,736 (11) | 2,873,472 (22) | 2,873,472 (22) | 4785 | 4823 | 2 | 2 | 115386 / 14423 / 14424 | 34.7 | 8.3 | 16761, g3an01 |
| h-pokec | 1,731,977 (16) | 1,731,977 (16) | 34,208,018 (16) | 51,774,836 (31) | 51,774,836 (31) | 65 | 66 | 2 | 2 | 816320 / 408160 / 408160 | 11.1 | 6.1 | 16741 step 32, g2n01 |

Equal to Table 1 (nodes, node types, edges and edge types of the loaded graph): ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem, ieee-cis-g, h-pokec.
Reported differences from the paper (equal to H2GB's processed files, not a stop): oag-eng: classes; oag-chem: classes; ieee-cis-g: target_feature_dim; h-pokec: target_feature_dim.

## Per dataset

### ogbn-mag
- H2GB: `MAGDataset` via `preformat_MAG` (name 'ogbn-mag'); transform `None`; target `paper`, label multiclass; split: OGB's official split, in the processed file (mag_dataset.py:130-142).
- processed `data/NC/mag/ogbn-mag/processed/data.pt`: 2,602,203,256 bytes, sha256 `d32e050f386dcd176fe4dc97efe7c3dc3d8ea4b5f06c6b7bb4c09410c1656fbc`; folder: {'data.pt': 2602203256, 'pre_filter.pt': 864, 'pre_transform.pt': 864}.
- nodes: {'paper': 736389, 'author': 1134649, 'field_of_study': 59965, 'institution': 8740}
- feature dims: {'paper': [128], 'author': [128], 'field_of_study': [128], 'institution': [128]}
- edges stored: {'author|affiliated_with|institution': 1043998, 'institution|rev_affiliated_with|author': 1043998, 'author|writes|paper': 7145660, 'paper|rev_writes|author': 7145660, 'paper|cites|paper': 10792672, 'paper|has_topic|field_of_study': 7505078, 'field_of_study|rev_has_topic|paper': 7505078}
- edges loaded: {'author|affiliated_with|institution': 1043998, 'institution|rev_affiliated_with|author': 1043998, 'author|writes|paper': 7145660, 'paper|rev_writes|author': 7145660, 'paper|cites|paper': 10792672, 'paper|has_topic|field_of_study': 7505078, 'field_of_study|rev_has_topic|paper': 7505078}
- labels: y [736389] torch.int64, labelled 736,389, unlabelled (y = -1) 0, classes (max y + 1) 349, present 349; split overlaps {'train&val': 0, 'train&test': 0, 'val&test': 0}.
  - train: 629,571 nodes, unlabelled in mask 0, classes present 349, class counts {'min': 70, 'max': 28600, 'top5': [28600, 27449, 22771, 22689, 21319]}
  - val: 64,879 nodes, unlabelled in mask 0, classes present 331, class counts {'min': 0, 'max': 4085, 'top5': [4085, 3447, 3149, 2535, 2088]}
  - test: 41,939 nodes, unlabelled in mask 0, classes present 311, class counts {'min': 0, 'max': 2776, 'top5': [2776, 2437, 2427, 1803, 1653]}
- Table 1 comparison: {'nodes': True, 'node_types': True, 'edges': True, 'edge_types': True, 'classes': True, 'target_feature_dim': True}

### mag-year
- H2GB: `MAGDataset` via `preformat_MAG` (name 'mag-year'); transform `None`; target `paper`, label multiclass; split: random 50/25/25 drawn at processing time, in the processed file (mag_dataset.py:163-172).
- processed `data/NC/mag/mag-year/processed/data.pt`: 2,602,203,256 bytes, sha256 `6ba59a64782f2d6a147d23877b2df89b05958bf3188ce6f9dd8955fc077732e4`; folder: {'data.pt': 2602203256, 'pre_filter.pt': 864, 'pre_transform.pt': 864}.
- nodes: {'paper': 736389, 'author': 1134649, 'field_of_study': 59965, 'institution': 8740}
- feature dims: {'paper': [128], 'author': [128], 'field_of_study': [128], 'institution': [128]}
- edges stored: {'author|affiliated_with|institution': 1043998, 'institution|rev_affiliated_with|author': 1043998, 'author|writes|paper': 7145660, 'paper|rev_writes|author': 7145660, 'paper|cites|paper': 10792672, 'paper|has_topic|field_of_study': 7505078, 'field_of_study|rev_has_topic|paper': 7505078}
- edges loaded: {'author|affiliated_with|institution': 1043998, 'institution|rev_affiliated_with|author': 1043998, 'author|writes|paper': 7145660, 'paper|rev_writes|author': 7145660, 'paper|cites|paper': 10792672, 'paper|has_topic|field_of_study': 7505078, 'field_of_study|rev_has_topic|paper': 7505078}
- labels: y [736389] torch.int64, labelled 736,389, unlabelled (y = -1) 0, classes (max y + 1) 5, present 5; split overlaps {'train&val': 0, 'train&test': 0, 'val&test': 0}.
  - train: 368,194 nodes, unlabelled in mask 0, classes present 5, class counts [36786, 80456, 84159, 78386, 88407]
  - val: 184,097 nodes, unlabelled in mask 0, classes present 5, class counts [18338, 39990, 41955, 39373, 44441]
  - test: 184,098 nodes, unlabelled in mask 0, classes present 5, class counts [18429, 40540, 41945, 39016, 44168]
- Table 1 comparison: {'nodes': True, 'node_types': True, 'edges': True, 'edge_types': True, 'classes': True, 'target_feature_dim': True}

### oag-cs
- H2GB: `OAGDataset` via `preformat_OAG` (name 'cs'); transform `ToUndirected()`; target `paper`, label multiclass; split: by year: <= 2016 / 2017 / >= 2018, in the processed file (oag_dataset.py:151-153).
- processed `data/NC/OAG/cs/processed/data.pt`: 3,649,001,164 bytes, sha256 `4ead71634b34f8125e35256ac742f2d7efbba1bb65cae8c65739db23f0b727c6`; folder: {'data.pt': 3649001164, 'pre_filter.pt': 864, 'pre_transform.pt': 864}.
- nodes: {'paper': 546704, 'field': 45775, 'author': 511122, 'affiliation': 9090}
- feature dims: {'paper': [768], 'field': [768], 'author': [768], 'affiliation': [768]}
- edges stored: {'paper|PP_cite|paper': 5872552, 'author|AP_write_last|paper': 431027, 'author|AP_write_other|paper': 664266, 'author|AP_write_first|paper': 456903, 'field|FF_in|field': 262839, 'paper|PF_in_L0|field': 546831, 'paper|PF_in_L3|field': 869729, 'paper|PF_in_L1|field': 1202487, 'paper|PF_in_L2|field': 2347239, 'paper|PF_in_L5|field': 203338, 'paper|PF_in_L4|field': 304912, 'author|in|affiliation': 614161}
- edges loaded: {'paper|PP_cite|paper': 11729984, 'author|AP_write_last|paper': 431027, 'author|AP_write_other|paper': 664266, 'author|AP_write_first|paper': 456903, 'field|FF_in|field': 525678, 'paper|PF_in_L0|field': 546831, 'paper|PF_in_L3|field': 869729, 'paper|PF_in_L1|field': 1202487, 'paper|PF_in_L2|field': 2347239, 'paper|PF_in_L5|field': 203338, 'paper|PF_in_L4|field': 304912, 'author|in|affiliation': 614161, 'paper|rev_AP_write_last|author': 431027, 'paper|rev_AP_write_other|author': 664266, 'paper|rev_AP_write_first|author': 456903, 'field|rev_PF_in_L0|paper': 546831, 'field|rev_PF_in_L3|paper': 869729, 'field|rev_PF_in_L1|paper': 1202487, 'field|rev_PF_in_L2|paper': 2347239, 'field|rev_PF_in_L5|paper': 203338, 'field|rev_PF_in_L4|paper': 304912, 'affiliation|rev_in|author': 614161}
- labels: y [546704] torch.int64, labelled 229,104, unlabelled (y = -1) 317,600, classes (max y + 1) 3514, present 3514; split overlaps {'train&val': 0, 'train&test': 0, 'val&test': 0}.
  - train: 182,310 nodes, unlabelled in mask 0, classes present 3208, class counts {'min': 0, 'max': 3925, 'top5': [3925, 2769, 2713, 2482, 2362]}
  - val: 19,551 nodes, unlabelled in mask 0, classes present 1314, class counts {'min': 0, 'max': 929, 'top5': [929, 568, 508, 371, 341]}
  - test: 27,243 nodes, unlabelled in mask 0, classes present 1216, class counts {'min': 0, 'max': 1442, 'top5': [1442, 985, 782, 672, 567]}
- Table 1 comparison: {'nodes': True, 'node_types': True, 'edges': True, 'edge_types': True, 'classes': True, 'target_feature_dim': True}

### oag-eng
- H2GB: `OAGDataset` via `preformat_OAG` (name 'chemistry'); transform `ToUndirected()`; target `paper`, label multiclass; split: by year: <= 2016 / 2017 / >= 2018, in the processed file (oag_dataset.py:151-153).
- processed `data/NC/OAG/chemistry/processed/data.pt`: 2,960,710,156 bytes, sha256 `c9c64ab75ba34b233b77032d5a09b81c2551e173aec67671daba7f70cd908bcf`; folder: {'data.pt': 2960710156, 'pre_filter.pt': 864, 'pre_transform.pt': 864}.
- nodes: {'paper': 370624, 'field': 44714, 'author': 504888, 'affiliation': 9089}
- feature dims: {'paper': [768], 'field': [768], 'author': [768], 'affiliation': [768]}
- edges stored: {'paper|PP_cite|paper': 1054651, 'author|AP_write_first|paper': 288440, 'author|AP_write_other|paper': 402095, 'author|AP_write_last|paper': 283131, 'field|FF_in|field': 240326, 'paper|PF_in_L1|field': 686302, 'paper|PF_in_L5|field': 58205, 'paper|PF_in_L0|field': 370755, 'paper|PF_in_L2|field': 1569093, 'paper|PF_in_L3|field': 577152, 'paper|PF_in_L4|field': 140354, 'author|in|affiliation': 504888}
- edges loaded: {'paper|PP_cite|paper': 2105372, 'author|AP_write_first|paper': 288440, 'author|AP_write_other|paper': 402095, 'author|AP_write_last|paper': 283131, 'field|FF_in|field': 480652, 'paper|PF_in_L1|field': 686302, 'paper|PF_in_L5|field': 58205, 'paper|PF_in_L0|field': 370755, 'paper|PF_in_L2|field': 1569093, 'paper|PF_in_L3|field': 577152, 'paper|PF_in_L4|field': 140354, 'author|in|affiliation': 504888, 'paper|rev_AP_write_first|author': 288440, 'paper|rev_AP_write_other|author': 402095, 'paper|rev_AP_write_last|author': 283131, 'field|rev_PF_in_L1|paper': 686302, 'field|rev_PF_in_L5|paper': 58205, 'field|rev_PF_in_L0|paper': 370755, 'field|rev_PF_in_L2|paper': 1569093, 'field|rev_PF_in_L3|paper': 577152, 'field|rev_PF_in_L4|paper': 140354, 'affiliation|rev_in|author': 504888}
- labels: y [370624] torch.int64, labelled 294,050, unlabelled (y = -1) 76,574, classes (max y + 1) 3957, present 3957; split overlaps {'train&val': 0, 'train&test': 0, 'val&test': 0}.
  - train: 259,444 nodes, unlabelled in mask 0, classes present 3888, class counts {'min': 0, 'max': 5282, 'top5': [5282, 4318, 4215, 4154, 4075]}
  - val: 28,524 nodes, unlabelled in mask 0, classes present 1209, class counts {'min': 0, 'max': 1302, 'top5': [1302, 1071, 1056, 1050, 1034]}
  - test: 6,082 nodes, unlabelled in mask 0, classes present 439, class counts {'min': 0, 'max': 339, 'top5': [339, 318, 285, 275, 256]}
- Table 1 comparison: {'nodes': True, 'node_types': True, 'edges': True, 'edge_types': True, 'classes': False, 'target_feature_dim': True}

### oag-chem
- H2GB: `OAGDataset` via `preformat_OAG` (name 'engineering'); transform `ToUndirected()`; target `paper`, label multiclass; split: by year: <= 2016 / 2017 / >= 2018, in the processed file (oag_dataset.py:151-153).
- processed `data/NC/OAG/engineering/processed/data.pt`: 6,213,821,636 bytes, sha256 `f69877c0f4134b5c5e64b74811546fa54a1fb466e41f4a6126e864b2b7d0c3bd`; folder: {'data.pt': 6213821636, 'pre_filter.pt': 864, 'pre_transform.pt': 864}.
- nodes: {'paper': 747290, 'field': 65290, 'author': 1097433, 'affiliation': 8868}
- feature dims: {'paper': [768], 'field': [768], 'author': [768], 'affiliation': [768]}
- edges stored: {'paper|PP_cite|paper': 7228234, 'author|AP_write_last|paper': 664930, 'author|AP_write_other|paper': 1634514, 'author|AP_write_first|paper': 630999, 'field|FF_in|field': 410909, 'paper|PF_in_L3|field': 1710391, 'paper|PF_in_L2|field': 3108157, 'paper|PF_in_L1|field': 1332023, 'paper|PF_in_L0|field': 747417, 'paper|PF_in_L4|field': 398183, 'paper|PF_in_L5|field': 87674, 'author|in|affiliation': 1097433}
- edges loaded: {'paper|PP_cite|paper': 14452754, 'author|AP_write_last|paper': 664930, 'author|AP_write_other|paper': 1634514, 'author|AP_write_first|paper': 630999, 'field|FF_in|field': 821818, 'paper|PF_in_L3|field': 1710391, 'paper|PF_in_L2|field': 3108157, 'paper|PF_in_L1|field': 1332023, 'paper|PF_in_L0|field': 747417, 'paper|PF_in_L4|field': 398183, 'paper|PF_in_L5|field': 87674, 'author|in|affiliation': 1097433, 'paper|rev_AP_write_last|author': 664930, 'paper|rev_AP_write_other|author': 1634514, 'paper|rev_AP_write_first|author': 630999, 'field|rev_PF_in_L3|paper': 1710391, 'field|rev_PF_in_L2|paper': 3108157, 'field|rev_PF_in_L1|paper': 1332023, 'field|rev_PF_in_L0|paper': 747417, 'field|rev_PF_in_L4|paper': 398183, 'field|rev_PF_in_L5|paper': 87674, 'affiliation|rev_in|author': 1097433}
- labels: y [747290] torch.int64, labelled 747,053, unlabelled (y = -1) 237, classes (max y + 1) 2986, present 2986; split overlaps {'train&val': 0, 'train&test': 0, 'val&test': 0}.
  - train: 676,738 nodes, unlabelled in mask 0, classes present 2928, class counts {'min': 0, 'max': 27273, 'top5': [27273, 18714, 11986, 10478, 10259]}
  - val: 59,664 nodes, unlabelled in mask 0, classes present 1246, class counts {'min': 0, 'max': 1633, 'top5': [1633, 1552, 1213, 1093, 1065]}
  - test: 10,651 nodes, unlabelled in mask 0, classes present 663, class counts {'min': 0, 'max': 348, 'top5': [348, 307, 304, 269, 268]}
- Table 1 comparison: {'nodes': True, 'node_types': True, 'edges': True, 'edge_types': True, 'classes': False, 'target_feature_dim': True}

### ieee-cis-g
- H2GB: `IeeeCisDataset` via `preformat_IEEE_CIS` (name None); transform `Compose([
  ToUndirected(),
  AddSelfLoops(),
  NormalizeFeatures()
])`; target `transaction`, label binary; split: node order: first 80 % train, next 10 % val, rest test, set at load (master_loader.py:766-775).
- processed `data/NC/IEEE-CIS/processed/data.pt`: 2,784,766,460 bytes, sha256 `931fa351be9fcfe13d82d0cecf30581e0d92aac7457ea8ce5d4fc03159c2b664`; folder: {'data.pt': 2784766460, 'pre_filter.pt': 864, 'pre_transform.pt': 864}.
- nodes: {'transaction': 144233, 'card1': 8499, 'card2': 482, 'card3': 103, 'card4': 4, 'card5': 106, 'card6': 3, 'ProductCD': 4, 'addr1': 255, 'addr2': 72, 'P_emaildomain': 59, 'R_emaildomain': 60}
- feature dims: {'transaction': [4785], 'card1': None, 'card2': None, 'card3': None, 'card4': None, 'card5': None, 'card6': None, 'ProductCD': None, 'addr1': None, 'addr2': None, 'P_emaildomain': None, 'R_emaildomain': None}
- edges stored: {'transaction|to|card1': 144233, 'transaction|to|card2': 143331, 'transaction|to|card3': 144061, 'transaction|to|card4': 144049, 'transaction|to|card5': 143277, 'transaction|to|card6': 144055, 'transaction|to|ProductCD': 144233, 'transaction|to|addr1': 83786, 'transaction|to|addr2': 83786, 'transaction|to|P_emaildomain': 130842, 'transaction|to|R_emaildomain': 131083}
- edges loaded: {'transaction|to|card1': 144233, 'transaction|to|card2': 143331, 'transaction|to|card3': 144061, 'transaction|to|card4': 144049, 'transaction|to|card5': 143277, 'transaction|to|card6': 144055, 'transaction|to|ProductCD': 144233, 'transaction|to|addr1': 83786, 'transaction|to|addr2': 83786, 'transaction|to|P_emaildomain': 130842, 'transaction|to|R_emaildomain': 131083, 'card1|rev_to|transaction': 144233, 'card2|rev_to|transaction': 143331, 'card3|rev_to|transaction': 144061, 'card4|rev_to|transaction': 144049, 'card5|rev_to|transaction': 143277, 'card6|rev_to|transaction': 144055, 'ProductCD|rev_to|transaction': 144233, 'addr1|rev_to|transaction': 83786, 'addr2|rev_to|transaction': 83786, 'P_emaildomain|rev_to|transaction': 130842, 'R_emaildomain|rev_to|transaction': 131083}
- labels: y [144233] torch.int64, labelled 144,233, unlabelled (y = -1) 0, classes (max y + 1) 2, present 2; split overlaps {'train&val': 0, 'train&test': 0, 'val&test': 0}.
  - train: 115,386 nodes, unlabelled in mask 0, classes present 2, class counts [106740, 8646]
  - val: 14,423 nodes, unlabelled in mask 0, classes present 2, class counts [13106, 1317]
  - test: 14,424 nodes, unlabelled in mask 0, classes present 2, class counts [13069, 1355]
- Table 1 comparison: {'nodes': True, 'node_types': True, 'edges': True, 'edge_types': True, 'classes': True, 'target_feature_dim': False}

### h-pokec
- H2GB: `PokecDataset` via `preformat_Pokec` (name None); transform `ToUndirected()`; target `user`, label binary; split: random 50/25/25 of labelled users drawn at processing time, in the processed file (pokec_dataset.py:291-301).
- processed `data/NC/Pokec/processed/data.pt`: 989,825,292 bytes, sha256 `fac97b78e42f16b1a5a27b143f345a19af338e446cbade3fa13e05dd001ca3fd`; folder: {'data.pt': 989825292, 'pre_filter.pt': 864, 'pre_transform.pt': 864}.
- nodes: {'user': 1632803, 'life_style': 21579, 'music': 20080, 'cars': 2587, 'politics': 1423, 'relationships': 22067, 'art_culture': 1619, 'hobbies_interests': 6826, 'science_technologies': 756, 'computers_internet': 2236, 'education': 2347, 'sport': 7430, 'movies': 5870, 'travelling': 1251, 'health': 2390, 'companies_brands': 713}
- feature dims: {'user': [65], 'life_style': None, 'music': None, 'cars': None, 'politics': None, 'relationships': None, 'art_culture': None, 'hobbies_interests': None, 'science_technologies': None, 'computers_internet': None, 'education': None, 'sport': None, 'movies': None, 'travelling': None, 'health': None, 'companies_brands': None}
- edges stored: {'user|has_friend|user': 30622564, 'user|lists|life_style': 827974, 'user|lists|music': 580378, 'user|lists|cars': 92972, 'user|lists|politics': 40188, 'user|lists|relationships': 1010495, 'user|lists|art_culture': 51209, 'user|lists|hobbies_interests': 275117, 'user|lists|science_technologies': 41177, 'user|lists|computers_internet': 76867, 'user|lists|education': 82974, 'user|lists|sport': 165616, 'user|lists|movies': 180064, 'user|lists|travelling': 49697, 'user|lists|health': 74056, 'user|lists|companies_brands': 36670}
- edges loaded: {'user|has_friend|user': 44603928, 'user|lists|life_style': 827974, 'user|lists|music': 580378, 'user|lists|cars': 92972, 'user|lists|politics': 40188, 'user|lists|relationships': 1010495, 'user|lists|art_culture': 51209, 'user|lists|hobbies_interests': 275117, 'user|lists|science_technologies': 41177, 'user|lists|computers_internet': 76867, 'user|lists|education': 82974, 'user|lists|sport': 165616, 'user|lists|movies': 180064, 'user|lists|travelling': 49697, 'user|lists|health': 74056, 'user|lists|companies_brands': 36670, 'life_style|rev_lists|user': 827974, 'music|rev_lists|user': 580378, 'cars|rev_lists|user': 92972, 'politics|rev_lists|user': 40188, 'relationships|rev_lists|user': 1010495, 'art_culture|rev_lists|user': 51209, 'hobbies_interests|rev_lists|user': 275117, 'science_technologies|rev_lists|user': 41177, 'computers_internet|rev_lists|user': 76867, 'education|rev_lists|user': 82974, 'sport|rev_lists|user': 165616, 'movies|rev_lists|user': 180064, 'travelling|rev_lists|user': 49697, 'health|rev_lists|user': 74056, 'companies_brands|rev_lists|user': 36670}
- labels: y [1632803] torch.int64, labelled 1,632,640, unlabelled (y = -1) 163, classes (max y + 1) 2, present 2; split overlaps {'train&val': 0, 'train&test': 0, 'val&test': 0}.
  - train: 816,320 nodes, unlabelled in mask 0, classes present 2, class counts [414045, 402275]
  - val: 408,160 nodes, unlabelled in mask 0, classes present 2, class counts [207361, 200799]
  - test: 408,160 nodes, unlabelled in mask 0, classes present 2, class counts [206898, 201262]
- Table 1 comparison: {'nodes': True, 'node_types': True, 'edges': True, 'edge_types': True, 'classes': True, 'target_feature_dim': False}

## ogbn-mag vs mag-year (`--compare-mag`, stored graphs)

```
{
 "node_types_equal": true,
 "edge_types_equal": true,
 "edge_index_equal": {
  "author|affiliated_with|institution": true,
  "institution|rev_affiliated_with|author": true,
  "author|writes|paper": true,
  "paper|rev_writes|author": true,
  "paper|cites|paper": true,
  "paper|has_topic|field_of_study": true,
  "field_of_study|rev_has_topic|paper": true
 },
 "x_equal": {
  "paper": true,
  "author": true,
  "field_of_study": true,
  "institution": true
 },
 "y_equal": false,
 "masks_equal": {
  "train_mask": false,
  "val_mask": false,
  "test_mask": false
 },
 "paper_attributes": {
  "ogbn-mag": [
   "test_mask",
   "train_mask",
   "val_mask",
   "x",
   "y"
  ],
  "mag-year": [
   "test_mask",
   "train_mask",
   "val_mask",
   "x",
   "y"
  ]
 },
 "y_classes": {
  "ogbn-mag": 349,
  "mag-year": 5
 }
}
```

