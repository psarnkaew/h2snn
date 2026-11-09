# J6.2 Attribute inventory of both endpoint types, per LP task (after J6.1)

**Scope (author decision 2026-11-10).** J6.2 is this inventory only. The fingerprint probe and the Ctrl-G comparison are dropped for every task; see `j62_record.md`.

Every named attribute (or unnamed block) is classified as **own**, or **about the other endpoint / computed from the target relation** ("other / target"), with its source. "Removed" means removed by F17 in J6.1.

## acm_pa (paper → author)
| Type | Attribute | Class | Source |
|---|---|---|---|
| paper | 1,902 binary columns = indicators of the paper's term nodes | own (the paper's text) | J5.1: exactly the paper→term links, 3,025/3,025 (`j51_feature_rules_output.txt`, V2:r6) |
| author | 1,902 columns | other / target: OR of its papers → **removed** | J5.1, 5,959/5,959 |

## dblp_ap (author → paper)
| Type | Attribute | Class | Source |
|---|---|---|---|
| author | 334 binary keywords | other / target: a function of its paper set → **removed** | J5.1 (150/150 identical sets) |
| paper | 4,231 count columns (0–5), unnamed | own (bag of words). Not a function of the paper's authors: 0/14,328 rules; identical author sets give identical features only 20/8,627 | `info.dat` gives only the dimension; J5.1. The column meaning is **UNVERIFIED**: no on-disk documentation names it |

## imdb_ma (movie → actor)
| Type | Attribute (column) | Class | Source |
|---|---|---|---|
| movie | 0 num_critic_for_reviews, 1 duration, 5 gross, 6 num_voted_users, 8 facenumber_in_poster, 9 num_user_for_reviews, 10 budget, 11 title_year, 13 imdb_score, 14 aspect_ration, 15 movie_facebook_likes | own (attributes of the movie) | `info.dat` "Attributes" |
| movie | 2 director_facebook_likes | about the director, who is **not** on the target relation → kept (F17) | `info.dat`; spec v1.10 F17 |
| movie | 3 actor_1_, 4 actor_3_, 12 actor_2_facebook_likes, 7 cast_total_facebook_likes | other endpoint (the movie's actors) → **removed** | `info.dat`; J5 review #1 |
| movie | 16:19 color, 19:67 language, 67:132 country, 132:148 content_rating (one-hot) | own | `info.dat` |
| movie | 148:3489 keyword-token counts | own (the movie's plot keywords; actors were derived from these, not the reverse) | `info.dat` keyword names; J5.1 (actor = MEAN of these; movie = f(actors) holds on 0/4,932) |
| actor | 3,341 columns | other / target: MEAN of its movies → **removed** | J5.1, 6,124/6,124 |

## freebase_bb (book – book)
| Type | Attribute | Class | Source |
|---|---|---|---|
| book | none (no node features in the release) | — | `node.dat` (empty fourth column on every row) |

## pubmed (disease – disease)
| Type | Attribute | Class | Source |
|---|---|---|---|
| disease (both endpoints; all four node types carry the same kind of vector) | 200 unnamed real-valued columns | own: the node's text representation, an aggregate of word2vec word embeddings trained on the PubMed corpus. It is not computed from the disease–disease relation and not an attribute of the other endpoint, so nothing is removed under F17 | see below |

**Documented source (checked against the PDFs on 2026-11-10).**

- **HNE (Yang et al., TKDE; arXiv 2004.00216), §IV-A "PubMed".** Verbatim:
  - "We compute word2vec [99] on all PubMed papers and aggregate the word embeddings to get 200-dim features for all types of nodes."
  - On the links: "The links are constructed through open relation pattern mining [132] and manual selection."
- **HGB (Lv et al., KDD 2021; arXiv 2112.14936), §4.2.2.** Verbatim: "We use the subset constructed by HNE".
- **HNE `Data/README.md`.** Gives only "#attributes 200" and "#attributed nodes ALL".

**Residual, not a removal.** The sources do not state:
- the aggregation operator;
- which text of a paper is used;
- whether the corpus used for the embeddings is the one the links were mined from.

The features are corpus-level word embeddings of the node's own name or phrase. F17 does not cover them, since it removes only features computed from the target relation or attributes of the other endpoint. They could still carry topical similarity between diseases, but that is a property of the node, not a leak of the held-out link. Recorded for the author; it is not acted on.

## lastfm (user → artist)
| Type | Attribute | Class | Source |
|---|---|---|---|
| user, artist | none (no node features) | — | `node.dat` |

## Result
- **acm_pa, dblp_ap, imdb_ma, freebase_bb, lastfm:** nothing classified "other / target" remains after J6.1.
- **PubMed:** classified **own** from HNE §IV-A, with no removal.
- No column needs removal under the rule beyond those already removed in J6.1, so this is not a stop.
