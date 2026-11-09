# J6.2 Attribute inventory (F18, reduced by author decision)
2026-11-10. Documents only; nothing in the data or the code changes.

## Design change (author decision, 2026-11-10)
J6.0 pre-registered J6.2 as (i) an attribute inventory plus (ii) a fingerprint probe compared against the training-graph control Ctrl-G (`j6_preregistration.md` §J6.2). The author dropped (ii) and the Ctrl-G comparison for every task, PubMed included. J6.2 is now (i) only.

- **Reason: cost.**
  - The probe and Ctrl-G build two-hop sparse products. On this machine these ran out of memory repeatedly, on PubMed at the last attempt.
  - Another project's processes held most of the RAM, and those were not touched.
  - The rule itself (F17) is settled by the inventory: it removes only features positively classified as computed from the target relation or as attributes of the other endpoint.
- **The PubMed run was stopped.** The partial probe outputs computed before the decision are not evidence for J6 and are not committed. They covered:
  - the real positive control on the release IMDB movie columns 3/4/7/12, which failed as required;
  - dblp_ap and imdb_ma, which passed;
  - acm_pa, freebase_bb and lastfm, which had no non-binary family.

  The script, its test and the outputs were moved out of the repository to `Claude outputs/j62_dropped_probe/` (untracked).
- **The pre-registration (J6.0 commit `9789d25`) is left as filed.** This record supersedes its §J6.2 (ii), its tolerance and its power check.

## Result (`j62_inventory.md`)
- Both endpoint types of all six tasks are inventoried, with sources.
- No attribute is classified "computed from the target relation / about the other endpoint" beyond those already removed in J6.1:
  - ACM author, DBLP author and IMDB actor blocks;
  - IMDB movie columns 3, 4, 7, 12.
- `director_facebook_likes` stays (F17).
- **PubMed:** the documented source of its 200-dim features is HNE §IV-A: "We compute word2vec [99] on all PubMed papers and aggregate the word embeddings to get 200-dim features for all types of nodes." HGB §4.2.2 uses HNE's subset. The features are classified **own**, with no removal. The quotes were checked against the arXiv PDFs.
- **Open item for the author (recorded, not acted on):**
  - The aggregation operator, the text unit and whether the embedding corpus is the one the links were mined from are not stated.
  - The dblp_ap paper columns remain unnamed: own by J5.1's test, but their meaning is UNVERIFIED.
- **Not a stop:** the inventory found no column the rule should remove that is not already removed.
