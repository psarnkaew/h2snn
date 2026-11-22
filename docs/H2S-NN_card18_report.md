# Card 18 report — baseline meta-paths on ACM paper–paper LP (MAGNN cited-by direction; H2G-former `metapath_pe`)

Card `docs/H2S-NN_card18_card.md` (design-docs fb974d3), base `card15` `2e4881c`, branch `card18`. Only baseline
adapters are changed; H2S-NN is not touched.

## Result, and a correction to the card 15 report

**Neither problem occurs in the runs on the real acm_pp bundle.** Card 15's MAGNN and H2G-former notes were wrong. I
read MAGNN's graph as H2S-NN's tied relation set (4 kept relations). The baseline bundles, however, pin
`fact_layer: untied` (`src/baselines/bundles.py:61-62`, `BASELINE_DATA_PINS`, plan §9.5 option A). So MAGNN's graph
holds all 8 ACM relations, including `paper-rel_1-paper` (ref).

- **MAGNN on the bundle.** The PrPAP / PrPSP hop finds `paper-rel_1-paper` directly (first branch of `hop`). Its edges are the citations reversed (md5 f027946f vs PcP's 52e205dc). So the cited-by paths were already correct, before and after this card.
- **H2G-former with `metapath_pe` on.** It already ran on the unfixed code. A smoke on card15 `2e4881c` gave no error, and its training loss is bit-identical to the run after the change. Every schema hop is an edge type of the untied walk graph.
- **No baseline number changes** from this card.

**What the change fixes.** The rule is still made correct for a graph where the task dropped a relation, which is what the card asks for:
- **Before:** on a graph without `paper-rel_1-paper` (the tied view), MAGNN's fallback read the cited-by hop forward, so PrP duplicated PcP. H2G-former raised `KeyError` there.
- **After:** both read the kept citation backwards.
- On today's untied bundles this branch is not reached. It would matter if a baseline were ever run on a tied graph.

## Change

- **Shared rule: `src/baselines/adapters/_h2gformer_encoders.py`, `dropped_hop`.** It handles a listed hop whose relation is not in the graph, where exactly one remaining relation joins the hop's two types.
  - Cross-type: the direction follows the types. This is the old MAGNN logic, so those edges are unchanged.
  - Same-type (ACM cite/ref): the named relation is the dropped exact reverse, so the kept relation is walked backwards. A `rev_<dropped>` hop walks it forwards.
  - With no unique remaining relation it raises `KeyError`.
- **H2G-former: `walk_hop`.** It maps each schema hop onto the walk graph: the hop itself if present, otherwise the kept relation or its `rev_` copy by `dropped_hop`. `metapath2vec_embeddings` applies it before its missing-hop check.
- **MAGNN: `src/baselines/adapters/magnn.py`, `TypedGraph.hop`.** The fallback branch (formerly `:205-209`) now calls `dropped_hop`. The first two branches (stored hop; `rev_` of a stored relation) are unchanged.

## Files changed (vs the merge commit acf3f8f)

- `H2S-NN/src/baselines/adapters/_h2gformer_encoders.py` (+`dropped_hop`, `walk_hop`, one line in `metapath2vec_embeddings`, `__all__`)
- `H2S-NN/src/baselines/adapters/magnn.py` (`TypedGraph.hop` fallback)
- `H2S-NN/tests/test_card18_metapath_direction.py` (new: tests (a) and (b))
- `H2S-NN/scripts/diagnostics/card18/{smoke_epoch.py, hop_arrays.py, c18_tests.sbatch}`
- `docs/card18_evidence/` (job logs, hop arrays, smoke records)

## Hop arrays on acm_pp, before and after (`docs/card18_evidence/hop_arrays_acm_pp.json`)

**Bundle graph (untied, 8 relations) — the graph runs use.** Every listed hop is identical before and after:

| hop (listed) | resolves to | forward | edges | md5 (src, dst) | first pairs |
|---|---|---|---|---|---|
| PcP* hop 1 `paper-rel_0-paper` | `paper-rel_0-paper` | yes | 4,330 | 52e205dc | (0, 179), (0, 2697) |
| PrP* hop 1 `paper-rel_1-paper` | `paper-rel_1-paper` | yes | 4,330 | f027946f | (179, 0), (2697, 0) |
| PAP hop 1 `paper-rel_2-author` | `paper-rel_2-author` | yes | 9,949 | 32eb6b06 | (0, 3025), (0, 3026) |

H2G-former maps no hop: every schema hop is in the walk graph.

**Tied view (the same graph without `paper-rel_1-paper`) — where the fallback runs:**

| hop | before: resolves to, forward, md5 | after: resolves to, forward, md5 |
|---|---|---|
| PrP* hop 1 `paper-rel_1-paper` | `paper-rel_0-paper`, yes, **52e205dc** (= PcP: duplicate) | `paper-rel_0-paper`, **no**, **f027946f** (= stored reverse) |
| every other listed hop | unchanged | unchanged |

H2G-former, after the change: `paper-rel_1-paper` → `paper-rev_rel_0-paper`. Before the change, this hop was missing from the walk graph, which would raise `KeyError`.

## Tests (only (a)–(c); no full suite)

| test | job 18295 (2080 Ti, g2n01, 7 min, peak 3.7 GB; 18294 = first run, 3.8 GB) | result |
|---|---|---|
| (a) acm_pp, real bundle: PrPAP instance set ≠ PcPAP; every PrPAP first hop (v0, v1) has (v1, v0) a citation; on the tied view the PrPAP instance set equals the bundle's and the hop resolves to rel 0 backwards (PcPAP forwards) | `test_card18_metapath_direction.CitedByDirectionTests` | OK |
| (b) dblp_ap, imdb_ma, freebase_bb: every listed hop MAGNN reads has the same edge arrays, relation and direction as the pre-card-18 `hop` (copied in the test) | `OtherTasksUnchangedTests` | OK |
| (c) one epoch on acm_pp through the adapter's fit and the shared loop: MAGNN | `smoke_epoch.py --model magnn` | losses finite; 31/31 parameters moved |
| (c) H2G-former, `metapath_pe` on | `smoke_epoch.py --model h2gformer` | losses finite; 181/505 parameters moved |
| (c′) the same H2G-former smoke on the pre-card-18 code (card15 2e4881c, clean) | `C18_PROJECT_ROOT=…card15…` | no error; losses bit-identical to (c) |

**Notes on the tests.**
- **Test (a) changed in job 18295.** Its first version (job 18294) asserted that the bundle's PrP hop falls back to rel 0. That assertion failed, which is how the untied bundle was found. It now checks the bundle and the tied view separately.
- **Was the MetaPath2Vec channel on in (c)?** Shown by the code path only. The adapter's record does not carry `metapath_pe`. With `metapath_pe: True` and `has_walk_schema(LP, "acm_pp")` true (`DERIVED_SOURCE`), the stack includes `Hetero_Metapath` and `attach_metapath_embeddings` runs (`h2gformer.py:353-358`).
- **324 of H2G-former's 505 parameters did not move in one epoch,** both before and after this card (e.g. per-edge-type feed-forward layers of the vendor convs). This is not caused by this card and was not investigated; the card's check is that parameters move.

## Other

- MAGNN NC is unchanged: ACM/DBLP/IMDB NC are not wired (published numbers), and Freebase NC uses MAGNN's own lists through `from_hetero` (all stored relations).
- H2S-NN is unaffected: its literature hops resolve through the fact layer (`src/data/schemas.py:89`).

branch=card18 sha=078fc9b78470b7fe698690260fbbfcd6eff47504 card=docs/H2S-NN_card18_card.md base=2e4881c516442a5c31da63e2bf745b71b0773c0e
