# J4.4 HeaRT negatives (spec v1.8 F8): record (2026-11-09)
Pre-registration: `j4_preregistration.md` §J4.4 (HeaRT parameters with `file:line`; L3 quoted from Kovács et al. 2019, Eq. (1)). Script: `scripts/lp/build_heart_negatives.py`. Tests: `tests/test_heart_negatives.py`.

## Finding: HeaRT's "RA" is CN(a, v)/deg(v) in its code (VERIFIED-RUN, VERIFIED-SOURCE)
**What the code computes.** HeaRT's `calc_CN_metric` (`create_heart_negatives.py:69-75`) computes `cn = A @ (A * (1 / A.sum(dim=0)))`.
- `torch_sparse`'s `mul` with a `(1, N)` tensor scales **columns** (`mul.py:13-14`, identical at HeaRT's pinned 0.6.15 and 0.6.16, and in our 0.6.18).
- So the score is **CN(a, v) / deg(v)**, the common-neighbour count divided by the candidate's degree, not resource allocation Σ_w 1/deg(w).
- **Evidence:** running the three lines verbatim on a toy graph, `HeaRT == RA: False | HeaRT == CN/deg(v): True` (`j44_heart_ra_semantics.py` → `j44_heart_ra_semantics.txt`).

**What I did.** The author's C1 says "RA as HeaRT does", and the J4 card says "RA exactly as HeaRT"; here the two words conflict.
- The primary files follow **HeaRT's code**, which is what produced HeaRT's published negatives and so keeps comparability.
- The textbook-RA variant is built with the same seed (`--cn-metric ra`, under `heart_variant_ra/`) and compared below.
- This is not a card stop condition: the parameter is verifiable from the source; it disagrees with its name. **Decision for the author (§9).** It affects only the same-type tasks (PubMed, Freebase).

## Generation (VERIFIED-RUN, `j44_build_output.txt`; `meta.json` in each `heart/`)
| Task | Pair heuristic (C1) | Feature cosine (C2) | Positives val / test | Lists with random fill (of 2 per positive) | Random entries / all entries | Smallest eligible candidate pool | Smallest fill pool |
|---|---|---|---|---|---|---|---|
| acm_pa | L3 | yes | 895 / 995 | 0 / 0 | 0 % | 2,975 | — |
| dblp_ap | L3 | no | 1,768 / 1,964 | 3,064 / 3,386 | 54.1 % / 53.5 % | 4,051 | 3,810 |
| imdb_ma | L3 | no | 1,330 / 1,478 | 1,890 / 2,103 | 32.9 % / 32.7 % | 4,878 | 4,439 |
| lastfm | L3 | no | 7,301 / 18,567 | 1,097 / 2,597 | 7.2 % / 6.6 % | 1,281 | 1,459 |
| pubmed | HeaRT CN (as coded) | yes | 1,942 / 8,528 | 0 / 0 | 0 % | 15,206 | — |
| freebase_bb | HeaRT CN (as coded) | no | 16,894 / 18,723 | 3,489 / 4,118 | 7.1 % / 7.5 % | 35,275 | 39,911 |

- **No pool was too small (C4)**: every eligible pool is ≥ 1,281 against 250 needed, and every fill pool ≥ 1,459. Run time is 8 min 45 s on CPU.
- **Validation positives are the fixed `link.dat.valid` (C3)** through the loader; test positives are `link.dat.test`.

**Diagnosis (DBLP, IMDB).** About half of DBLP's and a third of IMDB's negatives are uniform random draws of the right type. This is HeaRT's rule: when fewer than 250 candidates score above zero, the rest is random.
- On these bipartite targets, an author or actor has few training facts, so L3 reaches few candidates.
- PPR at ε = 5e-5 rarely crosses the high-degree venue, term and keyword hubs.
- This measures how "hard" these negatives are; it is not a bug. Reported for the author; no parameter was changed.

## Checks (VERIFIED-RUN)
- **Real files** (`j44_real_checks.py` → `j44_real_checks_output.txt`), all 12 split files, **ALL PASS**:
  - shape n × 500 × 2;
  - positives = the loader's split;
  - first 250 keep the source, last 250 keep the destination;
  - corrupted endpoints of the required type;
  - no self-pair;
  - **none is a target-relation positive of any split, in either orientation**;
  - no duplicate within a list.
- **Same seed → identical files:** an acm_pa rebuild has all four md5s equal (`j44_rebuild_check.txt`).
- **Unit tests:** `tests/test_heart_negatives.py`, 22 OK. They cover:
  - hand values for HeaRT CN as coded, textbook RA, L3 (Eq. 1, both sides), `rank_score_matrix`, selection with fill, and the without-replacement fill;
  - Andersen push = exact PPR to 1e-6 at ε = 1e-9, and within 1e-3 at HeaRT's ε;
  - the card properties on three synthetic tasks (cross-type with features, same-type self-inverse with features, by-pair with self-loops);
  - a too-small pool stops.
- **Mutations** (`j44_mutation_check.py` → `j44_mutation_output.txt`), **4/4 caught**:
  - heuristics on the full graph (the card's mutation);
  - L3 on every target fact;
  - the mask limited to training positives (HeaRT's own filter);
  - the fill with replacement (HeaRT's default).

## Adaptation choices as implemented (all fixed by spec v1.8 C1–C4 or pre-registered)
- **Candidates** are only the target relation's type on the corrupted side.
- **The mask** covers positives of every split in either orientation, plus both endpoints, so self-pairs are excluded. This is stricter than HeaRT, which masks training (+ validation) neighbours.
- **The heuristic graph** is the loader's training message graph, homogeneous, symmetric, binary, without self-loops. L3 uses the target relation's training facts.
- **The fill** draws without replacement from the zero-score candidates not already selected.
  - HeaRT's top-k part can already contain zero-score candidates on rank ties (combined rank = min over heuristics), so excluding them is part of "no duplicate".
- **Hits@K** is a metric choice for the LP trainer card. HeaRT reports Hits@10 on its small datasets; the files do not depend on it.

## RA variant (textbook RA instead of HeaRT's CN as coded; same seed)
See `j44_ra_variant_overlap.txt`.
