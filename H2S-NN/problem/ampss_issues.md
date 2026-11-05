# AMPSS: open problems

Written 2026-07-27. Scope: `src/data/schema_discovery.py` and the face groups it materialises.
Supersedes handoff item #7, which recorded a bug that **does not exist as described** — read §0
before acting on anything here.

---

## 0. First: the "duplicate schemas" alarm was false

`docs/roadA_memory_handoff.md` item #7 records:

> **AMPSS duplicate schemas** — separate suspected bug, logged not fixed: ACM selects
> `paper-paper-term-paper` 2× and `paper-paper-paper-term-paper` 4×, each with identical instance
> counts; IMDB selects three reversal pairs. `lambda_redundancy` exists to prevent exactly this.

**Checked, and the schemas are distinct.** ACM's six selected groups carry six different
relation-id signatures:

| group | template name (as logged) | relation ids per position |
|---|---|---|
| 0 | `paper__paper__term__paper` | `[1, 6, 7]` |
| 1 | `paper__paper__term__paper` | `[0, 6, 7]` |
| 2 | `paper__paper__paper__term__paper` | `[1, 1, 6, 7]` |
| 3 | `paper__paper__paper__term__paper` | `[0, 0, 6, 7]` |
| 4 | `paper__paper__paper__term__paper` | `[0, 1, 6, 7]` |
| 5 | `paper__paper__paper__term__paper` | `[1, 0, 6, 7]` |

Relations `0` and `1` are the two directions of the paper–paper citation edge, so these are six
different typed routes that differ in the direction of each citation hop. No two groups share a
`(node_type_sequence, relation_ids)` signature, and their `face_node_index` tensors differ.

The identical `|F| = 8,000` that made them look like copies is just
`max_instances_per_schema: 8000` binding on all six.

**Consequences:** there is no duplication to remove, and therefore no memory to reclaim here. Any
plan that budgeted a ~67% face-branch reduction from deduplicating ACM is wrong.

What is real is the reason the alarm happened at all — see §1.

---

## 1. `template_name` is lossy, and it is what produced the false alarm

`FaceGroup.template_name` is derived from `node_type_sequence` alone
(`src/data/structures.py:12-19`). Every `paper → paper → term → paper` route renders as
`paper__paper__term__paper` no matter which relation was traversed at each hop.

**Impact.** Selected schemas are indistinguishable in `run.log`, `metrics.jsonl`, W&B run
comparisons, and any table derived from them. It is currently impossible to state in the paper
*which* schemas AMPSS chose — the identifier does not determine the schema. It also cost this
project one false bug report and a round of memory planning against a phantom.

**Fix.** Make the name carry the relation sequence, e.g.
`paper -[cites]-> paper -[has_term]-> term -[term_of]-> paper`, or at minimum append the
relation-id tuple. Keep it stable and human-readable, because it lands in the paper. Check
`schema_discovery.py`, `graph_faces.py`, and `preprocessing.py` for other places the name is built
so all three enumerators agree.

**Cost.** Small. **Value.** High — it gates being able to report AMPSS results at all.

**Reporting convention.** Formally a meta-path is node types *linked by relation types*
(`A1 -R1-> A2 -R2-> A3`), so the ACM six are genuinely distinct schemas. But most meta-path papers
report the node-type shorthand (`APA`, `PSP`, `APTPA`), under which they read as duplicates. Do not
write them as six plain meta-paths. Either give the relation-typed form, or collapse to
`P-P-T-P (citation-direction variants)`. Note that after the §3 fix this is less pressing for ACM:
the uncapped selection has five distinct node-type patterns and reads normally either way.

**Precedent, and why it does not transfer.** PM-HGNN (`research_paper_reinforce/Personalised
Meta-path Generation for Heterogeneous GNNs.pdf`) defines meta-paths with relation types (Def. 2)
and its RL action space *is* the relation set — but its reported meta-path tables (Tables 3, 6, 7)
show node types only, under an explicit footnote: "We do not present the relation types for all
meta-paths in Table 3."

That is lossless **for their graphs and not for ours.** Their Table 4 lists every relation:
IMDb `{M-D, D-M, M-A, A-M}`, DBLP `{A-P, P-A, P-T, T-P, P-V, V-P}`, ACM `{A-P, P-A, P-S, S-P}` —
exactly one relation per *ordered* node-type pair, so the node-type sequence determines the
relation sequence bijectively.

HGB ACM breaks that bijection: `paper -[cite]-> paper` and `paper -[ref]-> paper` are two distinct
relations on the same ordered type pair (5,343 edges each — a transpose pair of one citation
graph). Note also that PM-HGNN's ACM is *a different graph from ours*: 3,025 papers / 5,835 authors
/ 56 subjects, with **no term nodes and no citation edges**, against our 3,025 / 5,959 / 56 /
**1,902 terms** and 255,619 paper–term edges per direction.

So we cannot borrow PM-HGNN's shorthand, and we cannot cite it as precedent for dropping relations.
The honest framing is the reverse: relation-typed schemas are *necessary* on HGB-style graphs and
optional on the HAN/MAGNN-style graphs the meta-path literature reports on.

**Caveat to state alongside it.** Under the current cap the six ACM schemas differ *only* in which
direction of the same citation relation is taken at each hop. That is the weakest possible
demonstration of relation-awareness, and a reviewer will read it as such. Do not present it as
evidence that relation-typing matters — fix §3 first, then check whether the post-fix pool makes
the point better.

---

## 2. Redundancy is a soft penalty only, and cannot reject a near-duplicate

`topology_aware_score` (`schema_discovery.py:230-247`):

```python
redundancy = max((jaccard_endpoint_overlap(stats, s) for s in selected_stats), default=0.0)
base = (disc ** alpha_disc) * (cov ** alpha_cov) * (div ** alpha_div)
return base / (1.0 + redundancy)
```

`jaccard_endpoint_overlap` is bounded in `[0, 1]`, so the divisor is bounded in `[1, 2]`. **A
perfectly redundant candidate — Jaccard exactly 1.0 — has its score halved, not rejected.** If its
base score is more than twice the runner-up's, it is still selected.

Two further limits worth knowing before tuning this:

- Overlap is measured on `end_node_sample` only. Two routes that reach the same endpoints by
  entirely different relational paths score as redundant; two routes that differ only in direction
  may reach *different* endpoint samples and score as non-redundant. It is a proxy for endpoint
  coverage, not for schema similarity.
- Stage II's `lambda_redundancy` cannot compensate. Preselection is non-differentiable and fixes
  the pool; the gates only reweight what preselection already chose. This is stated in the
  manuscript (§AMPSS, "These terms refine only the preselected pool") and is correct as written —
  but it means redundancy must be handled in Stage I or not at all.

**Status: measured on ACM 2026-07-27 — the term is inert, not merely weak.** Pairwise
`jaccard_endpoint_overlap` among the six selected schemas is **0.63–0.68 for every pair**. The
divisor therefore sits at ≈1.65 near-uniformly across candidates, so it rescales the whole
candidate list by roughly the same factor and reorders essentially nothing. It is not that a
redundant candidate survives a halving — it is that every candidate is penalised equally, so the
penalty carries no selection signal at all.

This is a consequence of the first bullet above: on ACM, term and subject are hub types, so almost
any length-≥2 route reaches ~78% of papers and every pair of endpoint samples overlaps heavily.
`end_node_sample` cannot separate schemas on this graph.

ChatGPT's suggestion (see §7) to add a **node-type-level** redundancy check is well-aimed at the
symptom under the current cap — but see §3, where lifting the cap fixes the node-type collapse
without touching this term. Fix §3 first and re-measure before designing a new redundancy signal.

### 2a. AMP-HGNAS's dependency criterion, applied to our pools (measured 2026-07-27)

AMP-HGNAS (`research_paper/AMP-HGNAS.pdf`, Eq. 1) defines meta-path dependency as a **product of
two Jaccards** — one over covered nodes `N(P)`, one over traversed edges `E(P)` — adds a dependency
graph edge when `D > delta` (their Fig. 2 uses `delta = 0.7`), and then takes a **maximum
independent set** (Eq. 4, Alg. 1). It is a *hard* constraint: two dependent meta-paths cannot both
be selected.

Applying their `D` to the pools AMPSS actually selects:

| dataset | dependent pair | `D` | their MIS would keep |
|---|---|---|---|
| dblp | `APA` ~ `APAPA` | **0.804** | 3/4 — drops `APAPA` |
| imdb | `MKM` ~ `MKMKM` | **0.816** | 5/6 — drops `MKMKM` |
| acm | none (max 0.60) | — | 6/6 |

**So yes, AMPSS selects dependent schemas**, including AMP-HGNAS's own textbook example
(`APA` + `APAPA` on DBLP) and its IMDB analogue (a path and its self-concatenation).

**But their criterion is not sufficient either.** On ACM the six selected schemas have
`Jac(N) = 0.70–0.92` yet `Jac(E) = 0.21–0.70`, so the product stays under `delta` and they are
declared independent — because `cite` and `ref` carry different edge ids despite being transposes
of the same 5,343 citation links. Our endpoint Jaccard flags exactly this case at 0.66–0.69 and
then does nothing about it.

The two measures fail on opposite modes:

- theirs catches self-concatenation, misses transpose-relation variants;
- ours catches endpoint-reach overlap, misses self-concatenation, and never enforces.

**Therefore the fix is not "adopt Eq. (1)".** Two changes are needed: a measure covering both modes,
and a hard constraint replacing the soft divisor.

### 2b. The sheaf-native measure this should be

What a face group contributes to this model is **rank in `delta1`**. Since `L1 = delta1^T delta1`,
two groups whose boundary vectors span nearly the same subspace of the edge cochain space are
redundant no matter what their node or edge sets look like: the second adds no rank, only cost.

This dominates both Jaccards — node-Jaccard asks whether they touch the same nodes, edge-Jaccard
whether they traverse the same edges, but only **cycle-space overlap** predicts the effect on the
operator. Note `delta1`'s support *is* the boundary edge set, so AMP-HGNAS's `E(P)` term is a
first-order proxy for the quantity we can state exactly.

Cheap to compute pre-training: build the sparse boundary matrix `B_sigma` (path edges + closure
edge, already materialized) and score dependency as
`||B_i B_j^T||_F / (||B_i||_F ||B_j||_F)`; enforce via MIS or a rank-greedy pick.

This is also the answer to the standing critique that Stage I reads as detachable preprocessing:
selecting 2-cells is not selecting aggregation channels, and a selector that ignores boundary span
can return six faces that add one dimension of curl.

---

## 3. RESOLVED (diagnosed 2026-07-27): the candidate cap truncates in DFS pre-order

All six ACM routes go through citation hops and `term`. AMPSS never selects
`paper-author-paper` or `paper-subject-paper` — the canonical ACM meta-paths, and the ones
`configs/experiments/acm_h2snn.yaml` uses manually.

The diagnostic in §6b was run. **Both hypotheses were right, one for each meta-path:**

| meta-path | DFS index | in cap=64? | rank by `q0` | verdict |
|---|---|---|---|---|
| PAP `paper -[to]-> author -[to]-> paper` | 58 | yes | 20/64 (37/88 uncapped) | generated, **loses on scoring** |
| PSP `paper -[to]-> subject -[to]-> paper` | 68 | **no** | 8/88 when generated | **never generated** |

### The mechanism

`generate_schema_candidates` (`schema_discovery.py:54-85`) emits candidates in **DFS pre-order**
and hard-stops at `max_candidates`. It does not sample. ACM's relation ids are
`0:cite, 1:ref, 2:to author, 3:to paper, 4:to subject, 5:to paper, 6:to term, 7:to paper`, so the
DFS exhausts the entire `cite` subtree (indices 0–28) and the entire `ref` subtree (29–57) before
reaching `to author` at 58 — and never reaches `to subject`. **58 of the 64 slots are spent on
citation-prefixed routes.** The cap is not a random subsample of the schema space; it is a bias
toward whichever relation happens to hold the lowest id.

### Effect of lifting the cap

The full space is only **88 candidates** — the cap saves nothing meaningful. Setting
`max_schema_candidates: 88` (or higher) changes the selected pool completely:

| | selected pool (k=6) | distinct node-type patterns |
|---|---|---|
| `max_schema_candidates: 64` (current) | 6× `P →(cite\|ref)→ P → T → P` variants | **2** |
| uncapped (88) | `PTP`, `PTPSP`, `PAPTP`, `PSPTP`, `P-ref-PTP`, `P-cite-PTP` | **5** |

Plain `PTP` becomes the top-ranked candidate; `PSP` reaches rank 8/88; `PAPTP` and `PSPTP` are
both selected. The node-type-level collapse that §7 worries about **disappears without touching
the redundancy term** — which is why §2 should not be redesigned before this is fixed.

`PAP` still loses (37/88), on `end_coverage` (0.418 vs ~0.78 for term-mediated routes) despite
carrying the highest `type_diversity` in the pool (0.667). That is a defensible outcome to report
honestly — but see §3b, because `end_coverage` is estimated on a biased sample.

### Scope: ACM only

DBLP and IMDB were checked with the same script and **are not truncated**:

| dataset | cap | candidates generated | truncated? |
|---|---|---|---|
| acm | 64 | 88 | **yes, 24 cut** |
| dblp | 96 | 4 | no |
| imdb | 64 | 12 | no |

ACM is the only one of the three with a self-relation on the target type (`paper -cite/ref-> paper`),
and that is what makes its schema space explode from ~10 to 88. So this is not a generic AMPSS bug —
it is a bug that fires exactly when the target type has self-relations, which is also exactly when
the schema space is most worth searching.

Two side observations from the same run, not yet investigated:

- DBLP sets `selected_schema_k: 8` but only 4 candidates exist, so it always selects all 4 and the
  greedy selection is a no-op there. Any DBLP ablation over `selected_schema_k` is measuring nothing.
- IMDB selects reversal pairs at the node-type level (`M-A-M-K-M` and `M-K-M-A-M`; likewise for
  director). These are distinct *schemas*, but a face is a closed cycle, so it is not obvious they
  induce distinct *2-cells*. Worth checking whether the two groups' materialised cycles coincide as
  undirected edge sets before claiming they carry independent signal. **Do not assert either way
  without running it.**

**Fix.** Raise `max_schema_candidates` past the full enumeration size for each dataset (ACM needs
≥88), or replace the hard pre-order stop with breadth-first emission / reservoir sampling so the cap
degrades gracefully instead of amputating a relation subtree. Prefer the second: the first only
works because these three schema spaces happen to be tiny, and it will fail silently on the next
dataset.

---

## 3b. Schema stats are estimated on a 5.5% deterministic prefix of anchors

Found while running §3's diagnostic; not previously recorded.

`estimate_schema_stats` (`schema_discovery.py:153-155`) walks anchors in **ascending node id** and
breaks at `_MAX_STATS_SAMPLE = 5000` accumulated paths. With `max_instances_per_anchor: 30`, that
is at most **~167 of ACM's 3,025 papers — always the lowest ids**.

Two consequences:

- `end_coverage` (the `C` factor in `q`) is measured over that prefix, so it partly reflects how
  fast a schema burns its path budget rather than how much of the graph it reaches. High-branching
  hub routes exhaust 5000 paths in fewer anchors than PAP does; this is exactly the axis on which
  PAP loses.
- `num_instances` saturates at the cap for **87 of 88 candidates**, so it is a near-constant that
  still feeds `_schema_feature_vector` (via `log1p`) and `estimated_cost`. It carries no
  information and should either be normalised per anchor or dropped.

**Fix.** Sample anchors uniformly at random (seeded) instead of taking a prefix, and report
`num_instances` per visited anchor. Cheap, and it makes `C` mean what the manuscript says it means.

**Blocks:** any claim in the paper that `q` ranks schemas by graph coverage.

---

## 4. Instance caps are calibrated to a pre-simple-path-filter world

Carried over from handoff item #6, unchanged and still open.

Road A made faces simple paths (no revisited nodes, endpoints included). Measured interior-repeat
rates *before* the filter:

| group | L | interior repeats |
|---|---|---|
| all manual L=2 (PAP, PSP, APA, MAM, MDM) | 2 | 0.00% |
| dblp APTPA (manual) | 4 | 0.93% |
| dblp author-paper-**author**-paper-author (AMPSS) | 4 | **83.01%** |
| imdb, all six AMPSS groups | 4 | **30.6 – 44.4%** |

The DFS backfills its per-anchor budget with valid simple paths, so `|F|` still hits the cap — but
the *pool it samples from* changed drastically for AMPSS. `max_instances_per_schema` (8000 for
acm/imdb, 12000 for dblp) and `selected_schema_k` (6/6/8) were set before that filter existed.

**Until these are re-tuned, AMPSS instance-level numbers should not be reported.**

---

## 5. Priority

Revised 2026-07-27 after the §3 diagnostic.

1. **§3 candidate cap** — now the top item. It is a one-line config change per dataset (ACM: 64 →
   ≥88) that changes *which schemas the method selects*, and it invalidates every AMPSS pool
   recorded so far. Nothing else should be tuned or reported until this is settled.
2. **§3b anchor sampling** — small, and `end_coverage` is the factor PAP loses on, so it is
   entangled with any honest claim about §3's residual PAP result.
3. **§1 template names** — small, unblocks reporting, prevents a repeat of §0.
4. **§4 instance caps** — required before any AMPSS instance-level number goes in the paper, and
   must be re-tuned *after* §3 since the selected pool changes.
5. **§2 redundancy** — last. §3's fix removes the node-type collapse on its own; re-measure the
   pairwise Jaccard on the post-fix pool before designing anything new here.

**Do not report any AMPSS schema pool — ACM, DBLP, or IMDB — until §3 is fixed and the pools are
regenerated.** The pools currently in `experiments/` are artefacts of the DFS truncation.

---

## 6. Reproducing

Dump the selected pool with relation signatures (this is what refuted §0):

```python
# scratch script; conda run cannot take a multi-line python -c, write a file
import sys, yaml
from pathlib import Path
ROOT = Path("g:/My Drive/H2S-NN_test"); sys.path.insert(0, str(ROOT))
from src.configs.config import TrainConfig
from src.data.dataset import build_dataset

payload = yaml.safe_load((ROOT / "configs/experiments/acm_h2snn_ampss.yaml").read_text(encoding="utf-8"))
payload["device"] = "cpu"
data = build_dataset(TrainConfig.from_dict(payload))
for i, g in enumerate(data.face_groups):
    rel = g.relation_index
    per_pos = [sorted(set(rel[:, p].tolist())) for p in range(rel.size(1))]
    print(i, g.template_name, g.node_type_sequence, per_pos, g.num_faces)
```

Run through the `h2snn` env:
`"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python <file>`

---

## 6b. The §3 diagnostic

`scripts/diagnose_ampss_candidates.py` is the script §3 asked for, kept in the repo because it must
be re-run after any change to generation or scoring. It runs Stages B–D only — CPU, no faces
materialised, ~2 min on ACM.

```bash
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts/diagnose_ampss_candidates.py \
    --config configs/experiments/acm_h2snn_ampss.yaml --top_n 15
```

It prints, per dataset: the relation table with ids; capped vs uncapped candidate counts and the
exact list of candidates the cap cuts (with their DFS pre-order index); every candidate ranked by
`q0` with its `D`/`C`/`V`; the greedy selection under both pools; the node-type-pattern histogram of
the selection; the pairwise endpoint-Jaccard matrix (§2); and the §3b sampling-coverage summary.

**Read the node-type-pattern count first.** "2 distinct patterns across 6 slots" is the signature of
this whole class of problem.

---

## 7. Provenance

§0 was written after `docs/roadA_memory_handoff.md` item #7 turned out to be a false alarm.

§3, §3b, and the §2 measurement come from a 2026-07-27 review prompted by an external LLM read of
this file. That review made two checkable code claims — "PAP is generated but loses scoring" and
"PSP appears only if the candidate cap is raised beyond 64" — and **both were confirmed exactly** by
the diagnostic above. Its other points (relation-typed vs node-type meta-path identity; the lossy
`template_name`) restated §0 and §1.

Its remaining suggestion — add a redundancy check at the node-type level — is aimed at a real
symptom but at the wrong layer: the node-type collapse is downstream of the DFS truncation, and
lifting the cap resolves it without a new penalty term. Kept in §2 as a fallback if the post-fix
pools are still redundant.

Recording this because the value came from the two falsifiable claims, not the recommendations. Ask
for those.
