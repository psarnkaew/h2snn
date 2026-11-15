# B6 — feature rule F17 and leak audit F18 (attribute inventory) for the two large LP tasks

**Result: no feature violates F17 on either task (no stop).** Both tasks are same-type (paper–paper, user–user), so
"an attribute of the other endpoint" cannot occur in a node's own row; what remains to check is whether a feature
is computed from the target relation (citations, friendships). F18 here is the attribute inventory; the fingerprint
probe was dropped for every LP task by the author on 2026-11-10 for cost (`docs/H2S-NN_partJ_report.md` §12, J6.2:
"J6.2 is the attribute inventory only"), and is not run here either (stated).

## ogbn-mag P–P (`paper | cites | paper` of H2GB's MAG graph)
| Attribute of `paper` | Source | Computed from citations? | Class |
|---|---|---|---|
| `x` (128) | OGB's paper feature, described by OGB as a word2vec average over the paper's title and abstract (UNVERIFIED here: OGB's documentation was not opened in this session; recalled; `H2GB/datasets/mag_dataset.py:127`, `graph.x_dict['paper']` as OGB ships it) | no | own |
| `y` (venue), `train/val/test_mask` | the NC task's label and split (`mag_dataset.py:128-142`) | no | not a feature (H2S-NN's encoder reads `x` only; an LP run does not read `y`) |

The other node types' `x` (author, field of study, institution) are H2GB's averages of paper features over
`writes`/`has_topic` and `affiliated_with` (`mag_dataset.py:144-160`), not over citations; they are not endpoint
types. Graph-level: H2GB stores `cites` symmetrised (`to_undirected`, `mag_dataset.py:103-106`), so every citation
is a pair stored in both orientations; the split is by unordered pair and removes both (B6 report).

## H-Pokec U–U (`user | has_friend | user` of H2GB's Pokec graph)
| Attribute of `user` | Source (`H2GB/datasets/pokec_dataset.py`) | Computed from friendships? | Class |
|---|---|---|---|
| `public`, `completion_percentage`, `AGE` (normalised) | profile columns (`:267-270`, `node_fields` `:68-127`) | no | own |
| `main_area` one-hot | first part of the profile's `region` (`:220-221, :268-272`) | no | own |
| 51 presence flags of the text columns `node_fields[7:-1]` (`body` … `more`, e.g. `spoken_languages`, `relationships`, `my_partner_should_be`) | 1 if the user filled the field (`:274-275`) | no (the user's own profile; the flags say whether a field is filled, not its content) | own |
| `y` (gender) and the masks | the NC task's label and split (`:280, :291-301`) | no | not a feature |

Feature width 65 (`h2gb_nc_counts.md`) = 3 numeric + 51 flags + 11 `main_area` columns (11 = 65 − 54, DERIVED; the one-hot width was not counted separately). The club relations `user | lists | <club>` come from the
profile's hobby columns (`:224-252`), not from friendships.
