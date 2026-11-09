# J1.4 Hard negatives (HeaRT): stopped. The adaptation needs further choices (2026-11-09)
Card rule: "If HeaRT's procedure cannot be adapted without a further choice, stop J1.4 and list the choices." I stopped before writing any negative file or generation script. The J1 negative tests are therefore not written.

## What I read
**Paper:** Li et al., "Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking", NeurIPS 2023 Datasets and Benchmarks. arXiv 2306.10453v3, downloaded to the session scratchpad (not the repo). Read: §§1–4, Appendices C, G, I.

**Code:** `https://github.com/Juanhui28/HeaRT` at commit `c447cbff4c493b60d14b6544c3c39d3b9c5ddff0`, shallow-cloned to the scratchpad. Read: `heart_negatives/create_heart_negatives.py` (283 lines), plus `hard_utils.py` and `calc_ppr.py` (only the functions this procedure uses).

## HeaRT's procedure (VERIFIED-SOURCE paper, VERIFIED-CODE clone)
**Negatives per positive: K = 500**, half per endpoint.
- Paper §4.3: "For all datasets we use K = 500 negative samples per positive sample during evaluation." Code: `--num-samples` default 500 (`create_heart_negatives.py:270`).
- For a positive (a, b): K/2 = 250 of the form (a, \*) and 250 of the form (\*, b) (Fig. 2; `:174`, `:220-226`).

**Heuristics:**
- **RA**, `--cn-metric` default `RA` (`:268`), computed as `A · (A ⊙ 1/deg)` (`:69-75`).
- **PPR** by Andersen push, α = 0.15, ε = 5e-5 (`:273-274`; `calc_ppr.py:19`).
- **Cosine feature similarity on every non-OGB dataset**, between the two nodes of the pair (`:82-91`, `:241-246`). Paper App. G.1 says "for the Cora, Citeseer, and Pubmed datasets".

**Graph:** the undirected training graph, `adj_t` (`hard_utils.py:91-140`). Collab alone may add validation edges (`--use-val-in-test`).

**Ranking (`rank_score_matrix`, `:94-116`):**
- Each heuristic ranks candidates from largest to smallest (`scipy rankdata`, method `min`). A zero score gets rank max + 1, and a filtered candidate max + 2.
- **Combined rank = min over heuristics** (Borda with g = min; `--agg` default `min`, `:269`; App. G.2).
- Take the top K/2 by `torch.topk` (`:150-152`).
- If fewer than K/2 candidates have a non-zero score, the rest are drawn at random from the zero-score candidates **with replacement** (`np.random.choice` default, `:154-162`).
- Seeds are fixed at 42 (`hard_utils.py:14-18`).

**Filtering (`:176-210`):**
- Masked: the candidate's training-graph neighbours (for test negatives, train + validation), self-loops, and the pair itself.
- `--keep-train-val` disables the neighbour mask. App. I: the collab results keep train/validation positives as possible negatives.

**Metrics:** MRR and Hits@K.
- Under HeaRT the main tables report Hits@10 for Cora, Citeseer and Pubmed, and Hits@20 for OGB (Tables 4–5).
- App. H adds Hits@1/3/100 and Hits@50/100.
- The existing setting uses K ∈ {1, 3, 10, 100} for Planetoid and {20, 50, 100} for OGB (§2.3).

## Spec v1.7 adaptations already fixed (not choices)
- The corrupted endpoint is drawn only from the target relation's type on that side.
- Heuristics are computed on the training message graph only.
- Card test: no negative is a positive of **any** split, in either orientation.
  - This is **stricter than HeaRT**, which masks only training-graph neighbours for validation negatives (so another validation or test positive may appear) and train + validation for test negatives.
  - Recorded as a stated adaptation.

## Choices the adaptation still needs (the author's; I chose none)
**C1. RA on cross-type pairs is degenerate (DERIVED from the J1.1 relation table).** HeaRT scores the kept endpoint against each candidate of the other type. In the homogeneous view of the training graph, RA(a, v) > 0 needs a common neighbour of a and v.
- **DBLP author–paper:** authors neighbour only papers; papers neighbour authors, terms and venues. No common neighbour type, so **RA ≡ 0** for every candidate.
- **IMDB movie–actor:** movies neighbour directors, actors and keywords; actors neighbour only movies. **RA ≡ 0.**
- **ACM paper–author:** RA > 0 only through paper–paper citations (5,343 facts).
- **Consequence:** under a literal application, the negatives come from PPR (plus C2) alone. The options:
  - **(a)** Apply RA literally and report the degeneracy.
  - **(b)** Use a typed analogue, e.g. length-3 path counts a–b′–a′–v through the target relation (the bipartite form of common neighbours).
  - **(c)** Score the candidate against the replaced endpoint, which has the same type. This changes HeaRT's semantics.

**C2. Feature similarity.** HeaRT uses cosine similarity between the pair's two nodes on featured datasets. The two endpoint types have:
- different feature spaces in DBLP (author 334 vs paper 4,231) and IMDB (movie 3,489 vs actor 3,341);
- equal widths in ACM (1,902 and 1,902; whether it is the same vocabulary is UNVERIFIED);
- no features in Freebase and LastFM;
- one shared 200-dimensional space in PubMed (disease–disease, same type).

The options:
- **(a)** Never use it, as HeaRT does on OGB.
- **(b)** Use it only where both endpoint types share a feature space (PubMed; ACM if verified).
- **(c)** Use the cosine between the candidate and the replaced endpoint (same type, defined wherever that type has features). This changes the semantics.

**C3. Which validation set the validation negatives belong to** (from the J1.3 finding). The loader does not read `link.dat.valid`: it carves its own validation from `link.dat` by head (`link_prediction_dataset.py:366-370`). The negatives must be built for the validation set the trainer will use.
- Either the J1.2 file (needs the J1.3 proposal applied), or the loader's carved split (then J1.2's 81/9/10 is not the evaluated split).
- PubMed and LastFM have no fixed validation file at all; the loader carves it the same way.

**C4. Duplicates.** HeaRT's random fill draws with replacement, so a positive's list can repeat a negative.
- With the type restriction, the candidate pools are smaller (e.g. DBLP authors 4,057, IMDB movies 4,932), so the fill runs more often than on HeaRT's datasets.
- Keep the code's behaviour, or draw without replacement.

**PROPOSAL (for the author):**
- **C1:** option (a), which applies HeaRT's definition to the stated graph without inventing a heuristic. Then report that RA is identically zero on DBLP and IMDB and that hardness comes from PPR. Option (b) is the strongest alternative if PPR alone gives easy negatives; that is measurable with a heuristic baseline on the cluster.
- **C2:** option (b).
- **C3:** the J1.2 file, so that the split stays fixed for every seed and model.
- **C4:** keep HeaRT's behaviour, and report the number of duplicates.

## What is unaffected
The J1.2 splits and the J1.3 registration do not depend on these choices. PubMed alone (same-type target, shared feature space, RA well defined) could be built with HeaRT literally once C3 and C4 are settled.
