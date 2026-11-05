# Handoff: H2SNN memory / OOM investigation (post-Road A)

Written 2026-07-26. Continues a session that (a) implemented **Road A** — making the face level a
genuine 2-cell structure with an exact cochain condition — and (b) profiled the resulting memory
behaviour. **Road A is finished and verified; do not redo it.** The open topic is memory, driven by
the question: *will H2SNN OOM at `L=6` with more AMPSS paths?* (Answer: yes.)

---

## 0. Read this first: measure, don't reason

During this session I predicted memory behaviour **four times from tensor-shape arithmetic and was
wrong every time**:

1. Predicted Road A would *save* 1.66 GB on DBLP. Wrong sign.
2. Corrected to "+2·|F|·d², memory is the binding constraint." Also wrong — restriction maps are not
   where the memory is.
3. Hypothesised autograd retention across the `K` Euler steps dominated. Measured: `K=1 → 3.06`,
   `K=8 → 3.71` GiB. Nearly flat. Gradient checkpointing the diffusion loop would save almost nothing.
4. Hypothesised the face-branch LSTM + token MLP dominated. Shrinking `d_h` 64→16 and
   `face_mlp_hidden_dim` 128→32 (4×) saved only 0.52 of 1.71 GiB.

The actual top line item on ACM turned out to be the **Householder orthogonal parameterization**,
in code neither of us had opened. **Profile before proposing anything.**

---

## 1. State of the code — Road A, complete

### What changed conceptually

`δ₁δ₀ = 0` now holds **identically** rather than being encouraged by a penalty. Three things had to
land together:

1. **Closure edges are first-class 1-cells.** Appended to the edge list with their own stalks, so
   they enter `δ₀`, `L₀`, and `δ₁`'s boundary. `CoboundaryOps.closure_L0` is **deleted**.
2. **Functorial (transport-composed) face maps.** The edge→face map is no longer a free MLP output;
   it is derived from one free per-instance map `A_f` plus the node→edge maps along the path.
3. **Simple-path faces.** Instances that revisit a node (endpoints included) are excluded, because
   the transport assigns one map per node *position*.

### The construction (`src/models/sheaf_maps.py::compose_face_transport`)

For boundary `v0 → v1 → … → vL → v0`:

```
T[v0]     = A_f                                        # one free d×d map per instance
T[vi]     = T[v(i-1)] @ inv(F_src(ei)) @ F_tgt(ei)
F_{ei◁f}  = T[v(i-1)] @ inv(F_src(ei))
# closure cell eC = (vL → v0), oriented ALONG the cycle so its sign is +1 like the rest:
F_tgt(eC) = I ;  F_src(eC) = holonomy = Π inv(F_src(ei)) @ F_tgt(ei) ;  F_{eC◁f} = A_f
```

Every boundary term collapses to `T[v]x_v − T[u]x_u`; the cycle telescopes to exactly 0, with **no
trivial-holonomy assumption**. `F_src(eC)` is the path holonomy — independent of `A_f`, zero new
parameters. Inverses are closed-form: maps are `λ·Q` (scaled orthogonal), so `F⁻¹ = Fᵀ/λ²` — no solve.

### Files changed

| file | change |
|---|---|
| `src/models/sheaf_maps.py` | `compose_face_transport`, `FaceTransport`, `EdgeToFaceOutput`, `_invert_path_maps`; `SequenceEdgeToFaceSheaf` attention-**pools** to one `A_f` (per-edge attention scaling removed — it breaks the telescoping); `NodeToEdgeSheafOutput.F_src_scale`; conformal scale clamped **below** at `-2.0` so `F⁻¹` can't explode |
| `src/models/h2snn_model.py` | `_encode_from_h0` extends `edge_index`/`edge_type`/`F_src`/`F_tgt` with closure cells; `closure_L0` call removed; `EncodedState.edge_index` + `.num_real_edges` added (consumers of `s_star` must use these, **not** `data.edge_index`); relation embedding gains one row (`closure_relation_id`) |
| `src/models/coboundary_ops.py` | `closure_L0` deleted |
| `src/data/face_geometry.py` | **new** — `validate_face_orientation`, `is_simple` |
| `src/data/preprocessing.py`, `schema_discovery.py`, `graph_faces.py` | revisit guard in each DFS (subsumes the endpoint check, now unconditional rather than config-gated); orientation assertion at face-tensor build |
| `src/trainers/regularizers.py` | **deleted** |
| `src/trainers/{trainer,link_prediction_trainer,graph_classification_trainer}.py` | all geometry losses removed |
| `src/configs/config.py` | `lambda_c`, `lambda_co`, `lambda_0`, `lambda_1` removed; `closure_scale` default `1e-6 → 1.0` |
| 22 YAML configs | those four lambda keys stripped |
| `scripts/{tune_h2snn,run_h2snn_wandb_tuning,h2snn_paper_studies}.py` | lambda keys removed from search spaces; `no_closure` ablation → `weak_closure` (`closure_scale: 0.1`), since `closure_scale` must be **> 0** |
| `tests/test_coboundary_chain.py` | **new** — the correctness gate |
| `tests/test_models.py` | three obsolete `closure_L0`/per-edge-map tests replaced |
| `CLAUDE.md`, `docs/h2snn_current_model_documentation.html`, `paper/h2snn_manuscript.tex` | claims updated |

### Objective, now

```
L = L_task + ρ(t)·L_disc          # manual schemas ⇒ task loss alone
```

All four geometry regularizers removed, each for a stated reason:

- **coboundary** (`λ_co`): obsolete — `δ₁δ₀ = 0` holds identically. It was also *unsatisfiable* before:
  on an open path the two endpoint blocks of `δ₁δ₀` each appear in one term and node→edge maps are
  invertible, so exact zero forced the face maps to zero. It acted as shrinkage on the face maps.
- **consistency** (`λ_c`): counterproductive. `x` diffuses under `L₀` alone and the edge channel never
  feeds back, so forcing `s* → δ₀x*` makes the edge cochain a deterministic image of the node cochain
  and faces contribute nothing.
- **smooth1** (`λ₁`): counterproductive one rank up. With the chain condition exact, `δ₁s*` **is** the
  curl component and the only part `δ₁ᵀδ₁` acts on — shrinking it drives the face branch inert.
- **smooth0** (`λ₀`): was already `0.0`; a knob advertised and never used.

### Verification status

- `tests/test_coboundary_chain.py`: residual `< 1e-9` at **L=2, 3, 4**; plus `δ₁ᵀ` adjointness with the
  extra boundary column; plus closure-map-is-holonomy and `F_{eC◁f} == A_f`.
- Real data, relative `‖δ₁δ₀h⁰‖/‖δ₀h⁰‖`: **acm 1.04e-07, dblp 1.55e-07, imdb 1.37e-07,
  acm_ampss 3.66e-08** — float32 roundoff.
- Full suite: 63 tests, **1 failure**, pre-existing and unrelated
  (`test_notebooks_reference_repeated_run_and_wandb_tuning_scripts` asserts notebooks carry no
  committed outputs).
- All three task paths smoke-run at `d=4` on CPU: node (acm), graph (mutag), link (lastfm).
- `paper/h2snn_manuscript.tex` compiles clean (`latexmk` exit 0, no undefined refs/citations).
  New `Proposition 1` (exact cochain condition) with proof at `\label{prop:chain}`.

### Simple-path filter — measured effect

Interior-repeat rates went to **0.00% everywhere**. Before the filter:

| group | L | interior repeats |
|---|---|---|
| all manual L=2 (PAP, PSP, APA, MAM, MDM) | 2 | 0.00% |
| dblp APTPA (manual) | 4 | 0.93% |
| dblp author-paper-**author**-paper-author (AMPSS) | 4 | **83.01%** |
| imdb, all six AMPSS groups | 4 | **30.6 – 44.4%** |

DBLP APTPA lost only **18 of 202,625** instances — the DFS backfills its per-anchor budget with valid
simple paths rather than just dropping the bad ones. **AMPSS is far more affected**; its instance caps
should be re-tuned before its numbers mean anything.

---

## 2. Memory: measured attribution

Hardware: **NVIDIA RTX 4060 Laptop, 8.00 GiB**. One forward+backward, fresh process per row.

### ACM, d=8, K=1 — where the 8.41 GiB lives

| component | GiB | how measured |
|---|---|---|
| **total (full)** | **8.41** | — |
| Householder orthogonalization | **3.07** | noFaces 6.00 − noFaces+noOrth 2.93 |
| face branch (δ₁, closure, transport, LSTM) | 2.41 | full 8.41 − noFaces 6.00 |
| everything else (encoders, node→edge MLPs, δ₀/L₀, readout) | 2.93 | noFaces + `use_orthogonal=false` |

`orthogonal_map: householder → cayley` on noFaces: **6.00 → 3.87 GiB**.
Full model at K=4: **9.25 → 7.13 GiB** — ACM then *fits* in 8 GiB at d=8.

### Why Householder costs more

`Orthogonal._householder_orgqr` ([src/models/orthogonal.py:37-42](../src/models/orthogonal.py#L37-L42))
loops `r = d` times, and each iteration produces a new `[|E|, d, d]` `output` that is an input to the
next `bmm` — so autograd retains **all d of them**: `O(d · |E| · d²)`. Cayley
([orthogonal.py:90-93](../src/models/orthogonal.py#L90-L93)) is four tensors and no loop:
`O(|E| · d²)`. **The gap scales as ~d/2**, which is why d=32 was hopeless on ACM/DBLP.

At `|E| = 548k, d = 8`: one `[548k,8,8]` fp32 = 140 MB; Householder ≈ 2 tensors × 8 iters × 2 (src,tgt)
≈ 4.5 GB predicted vs 3.07 measured.

### Update 2026-07-27: the Householder loop was an implementation artifact, now fixed

`_householder_orgqr` was rewritten to dispatch to a **compact-WY builder**
(`Orthogonal._householder_wy`, [src/models/orthogonal.py](../src/models/orthogonal.py)) for square
power-of-two `d` — which is every `d` any sweep uses. The old loop survives as
`_householder_sequential` and still handles odd / non-power-of-two `d`.

The `O(d)` retained activations become `O(log d)`. Adapted from `fasth/fasthpp.py` (MIT;
Mathiasen et al., NeurIPS 2020), though **not for the reason that paper gives**: FastH optimizes
wall-clock for *one* large matrix whose `O(d)` sequential inner products starve a GPU. Here the
batch is already ~1e6 tiny maps and the GPU is saturated; the merge is used purely for its
autograd-retention profile.

**It is the same matrix, not a reparameterization** — max `|ΔQ|` ≤ 1.3e-15 and max gradient
difference ≤ 8e-12 in float64 at `d = 2,4,8,16,32`, asserted by
`tests/test_cuda_hardening.py::test_householder_wy_matches_the_sequential_builder`. So there is no
accuracy question, no checkpoint incompatibility, and no manuscript change (the paper says
"scaled orthogonal" and never names the parameterization).

| run | sequential | WY | memory | time |
|---|---|---|---|---|
| imdb d=8 K=4 | 3.34 GiB / 649 ms | 3.18 / 744 | −5% | **+15%** |
| acm d=8 K=1 | 8.40 / — | 7.47 / — | −11% | both thrash |
| imdb d=16 K=4 | 8.58 / — | 6.85 / — | −20% | both thrash |

Isolated over a batch of maps: **1.36× at d=8, 1.98× at d=16, 3.19× at d=32.**

**Read this carefully — the win is memory at large `d`, and it costs ~15% compute.** The only
clean timing (imdb d=8, both well under the card) says WY is *slower*. Rows above the card's 8 GiB
are dominated by WDDM spilling and are not compute measurements. imdb d=16 at 6.85 GiB still
thrashes on this card (median 9549 ms, min 5124, max 10118) — it did **not** become comfortable;
Cayley at 5.93 GiB is rock steady (898–911 ms). The 20% matters on a 16–40 GB card, not here.

Chain residual unaffected: acm 4.66e-08, imdb 6.57e-08, dblp 1.02e-07.

Also rejected on the way: `torch.linalg.householder_product` — likewise bit-identical and the
memory winner (0.93 GiB at d=8/B=548k), but **35 s** vs 0.72 s, because cuSOLVER loops over the
batch for tiny matrices.

### Householder vs Cayley — the tradeoff is real, but it cuts both ways

512 random draws, d=8, float64:

| | `max‖QᵀQ−I‖` | det | **min Re(eigenvalue)** |
|---|---|---|---|
| householder | 1.89e-15 | +1.0000 | **−1.0000** |
| cayley | 1.11e-15 | +1.0000 | **−0.8496** |

Both exactly orthogonal; both `det = +1` at **even** d. The gap is specifically that **Cayley cannot
produce an orthogonal matrix with eigenvalue −1**: if `Qv = −v` with `Q = (I−S/2)⁻¹(I+S/2)` then
`(I+S/2)v = −(I−S/2)v ⇒ 2v = 0 ⇒ v = 0`. Hard exclusion; approaching it needs `‖S‖→∞`.

**Why that matters here:** a restriction map with eigenvalue −1 flips the sign of a feature direction
across an edge — exactly the operation encoding endpoint *disagreement*, i.e. the heterophily case the
paper is about. So Cayley's blind spot is not a random measure-zero corner. **Do not switch the default
on memory grounds alone; settle it with an accuracy comparison.**

**Correction 2026-07-27 — Householder has the mirror-image blind spot, and the above overstates its
case.** Measured, d=8, float64, 8000 Adam steps fitting a target:

| target | householder | cayley |
|---|---|---|
| `+I` (perfect agreement) | 1.02e-01, `‖params‖`=44.9 ✗ | **1.5e-22**, `‖params‖`=0.01 ✓ |
| `−I` (total flip) | **6.2e-17**, `‖params‖`≈0 ✓ | 2.20e-01, `‖params‖`=104 ✗ |
| `diag(−1,−1,1…)` | 1.01e-01 | 1.73e-01 |

The two parameterizations have **opposite zero points**: at `params → 0`, Householder gives `−I` and
Cayley gives `+I` (at scale 1e-4: `‖Q+I‖`=0.0013 vs `‖Q−I‖`=0.0006). Householder cannot express
"these endpoints agree exactly" — only approach it as `‖params‖→∞` — which is the *homophily* case,
and plenty of relations in ACM/DBLP/IMDB are homophilous. It is a trade, not a free win, and it is
also an init-time bias: small-init MLPs start Householder edges near maximal disagreement.

Second correction: **Cayley's eigenvalue gap narrows with `d`** — min Re(eig) over 512 draws is
−0.809 (d=8), −0.907 (d=16), **−0.944 (d=32)**. The −0.85 figure above is a d=8 number, so the
expressivity argument for Householder is weakest exactly at the dims the sweeps use.

Net: the accuracy comparison is still the thing that settles this, but do not go into it assuming
Householder is the expressive one. With the WY builder shipped, memory is no longer forcing the
choice — so decide it on accuracy and throughput.

On throughput, the only clean end-to-end comparison this card can make is imdb d=8 K=4, where all
three fit comfortably: **cayley 2.99 GiB / 546 ms, WY 3.18 / 744, sequential 3.34 / 649.** Cayley is
~1.4× faster than WY there. At d=16 cayley is 5.93 GiB / 905 ms and rock steady while both
Householder builders thrash, so the d=16 gap cannot be measured here — it needs a bigger card.

What does **not** break with Cayley: orthogonality (so Road A's `F⁻¹ = Fᵀ/λ²` and the chain condition
are fine — verified, residual 3.41e-08 vs 8.33e-08); parameter count (`d(d−1)/2` for both, so no
architectural change and checkpoints stay shape-compatible); conditioning (`I − S/2` is never singular
for skew `S`); speed (Cayley is *faster* — 586 vs 622 ms per fwd+bwd on imdb d=8).

**Latent trap:** a product of `d` Householder reflectors has `det = (−1)^d`. At **odd** d that's −1, so
the identity map is unrepresentable and the sheaf could never have trivial restriction maps. Nothing
sweeps odd d today (`{16,32,64}`), but it would silently distort such a run. Worth a guard.
Confirmed at d=3 and d=5 (det = −1.000000 across 512 draws). The WY dispatch does **not** change
this — odd `d` falls back to `_householder_sequential`, which has the same trap. Still open (#9).

### What is *not* the problem

- **K (diffusion depth).** imdb d=8: K=1 → 3.06, K=2 → 3.15, K=4 → 3.34, K=8 → 3.71 GiB.
  ⇒ gradient checkpointing the Euler loop is near-worthless.
- **The face LSTM / token MLP.** imdb d=8 K=4: `(d_h=64, fh=128)` → 3.34;
  `(32,64)` → 2.99; `(16,32)` → 2.82. A 4× shrink buys 0.52 of a 1.71 GiB face branch. And these are
  the *d-independent* terms, so their share **shrinks** as d grows.
- **Restriction maps at small d.** At d=8 all maps together are ~164 MB of the face branch. They only
  dominate at d≥32 (see §3).

### Road A's own contribution to memory

Estimated **150–200 MB at d=8 (~5–7%)**: closure maps, one extra face-map column (`L+1` vs `L`), closure
entries in the edge cochain — partly offset because `map_head` now emits 1 map per instance instead of
`L`. **The 9.2 GiB on ACM/DBLP is overwhelmingly pre-existing.** Note this was never isolated against a
true pre-Road-A measurement (see §5).

### Pre-existing: the configs don't fit locally, and didn't before

`validate_node_edge_memory_budget` (guard fraction 0.20 = 1.60 GiB on this card):

| config | d=32 estimate | verdict |
|---|---|---|
| acm | 6.21 GiB | **REJECT** |
| dblp | 2.71 GiB | **REJECT** |
| imdb | 0.98 GiB | PASS |

The paper's runs were done on Colab (`experiments/colab/`). ACM and DBLP OOM at **d=16** too, with
Cayley. Local use is limited to `d≤8` smoke tests.

---

## 3. The L=6 / more-AMPSS-paths question — answered

Setup: `imdb_h2snn_ampss.yaml`, `min_schema_length = max_schema_length` to pin L,
`max_instances_per_schema = 8000`. Incidences = `|F| × (L+1)`.

| d | L | \|F\| | incidences | peak GiB |
|---|---|---|---|---|
| 8 | 2 | 24,000 | 72,000 | 2.02 |
| 8 | 4 | 32,000 | 160,000 | 2.49 |
| 8 | 6 | 32,000 | 224,000 | 2.83 |
| 16 | 6 | 32,000 | 224,000 | **7.46** |
| 16 | 6 | 64,000 (k=8) | 448,000 | **10.09** |
| 32 | 6 | 32,000 | 224,000 | **OOM** |
| 16 | 6 | ~128,000 (k=16) | ~896,000 | **OOM** |

(no-faces baseline for imdb d=8 ≈ 1.62 GiB. L=3 yields 0 groups on imdb — target-start/end forces even
length.)

**Face branch is dead linear in incidences**: 5.4 KB each at d=8 (constant across L=2/4/6), 11.7 KB at
d=16 as a marginal. Roughly linear in d too. Planning formula, reproduces every row within ~10%:

```
face branch bytes ≈ 0.73 KB × d × |F| × (L+1)
```

### Verdict

**Yes, it will OOM — and already does on the *smallest* dataset at mild settings.** imdb at
`d=16, L=6, k=8×8000` = 10.09 GiB.

Worse: the repo's AMPSS YAMLs set `max_instances_per_schema: 8000`, but the **dataclass default is
20000** with `selected_schema_k = 8`. At those defaults with `L=6`: `160,000 × 7 = 1.12M` incidences →
**~13 GiB at d=16, ~26 GiB at d=32**, face branch alone, before the `|E|`-side baseline. An A100-40GB is
tight; anything smaller won't start.

### The unavoidable tradeoff

`L=2 → 6` multiplies `(L+1)` by 2.33, so at fixed memory `|F|` must fall 2.33×. **`L` and `|F|` trade off
directly.** Not an implementation artifact — it is what the operator's support costs.

---

## 4. Options, with measured sizing

User constraints stated this session: **no gradient checkpointing** (time cost) and **no cutting `|F|`**
(feared loss of contribution).

| option | effect | notes |
|---|---|---|
| **`cache_face_maps=False`** — store only `A_f`, rebuild `F_{e◁f}` inside `δ₁` | ~34% of per-incidence cost at d=16; **~70% at d=32** | **Distinct from gradient checkpointing**: recomputes a few `d×d` matmuls per Euler step, *not* a network forward. Nearly free on GPU. Gets better exactly where needed. **Not implemented.** |
| **bf16 / AMP** | ~2× everything | **Do not do this naively — measured, it breaks Road A.** `max‖QᵀQ−I‖` at d=16 goes 2.1e-06 (fp32) → 1.2e-02 (fp16) → **9.7e-02 (bf16)**. `F⁻¹ = Fᵀ/λ²` and Prop. 1 assume exact orthogonality, so the ~1e-7 chain residual would become ~1e-2 and the paper's claim would be unsupportable. Autocast also **crashes today** at `orthogonal.py` (`Index put requires the source and destination dtypes match`). Any AMP must exclude the orthogonalization, transport, and coboundary ops — which is where the memory is, so the realistic win is far below 2×. Unmeasured. |
| **compact-WY Householder** (`_householder_wy`) | −5% at d=8, −20% at d=16; isolated 1.98×/3.19× at d=16/32 | **Done 2026-07-27.** Bit-identical math, so free of accuracy risk. Costs ~15% compute. Helps the `\|E\|` baseline only ⇒ **irrelevant to the L=6 problem.** |
| **`orthogonal_map: cayley`** | −2.1 GiB on ACM | Helps the `|E|` baseline, **not** the face branch ⇒ irrelevant to the L=6 problem. Expressivity cost (§2) — but see the 2026-07-27 correction: it cuts both ways. |
| reduce `d` | linear | d=16 instead of 32 halves it. |
| reduce `K` | negligible | measured flat. |
| shrink `d_h` / `face_mlp_hidden_dim` | 0.52 GiB at d=8 | d-independent terms; share shrinks as d grows. |
| cut `|F|` | linear — the biggest single dial | User ruled out. Worth noting `|F|` is *already* an arbitrary sample: the model keeps **0.5% of DBLP APTPA** (202k of 41.6M) and **6% of ACM PSP** (139k of 2.2M) available instances. `h2snn_paper_studies.py` already defines a `face_k{10,25,50,100}` sweep. |

Realistic recipe for `L=6` at `d=32`: `cache_face_maps=False` **and** AMP **and** a deliberate budget
(~30–50k total instances on a 40 GB card, by the formula).

---

## 5. Open items

1. ~~**Implement `cache_face_maps`**~~ — **refuted, closed 2026-07-27. Do not implement.** Measured
   at the exact L=6 shapes (`|F|`=64,000, L=6, d=16, K=4), isolating `compose_face_transport`:

   | mode | peak | time |
   |---|---|---|
   | cached (today: `build_maps` returns `cache.maps`) | **2.768 GiB** | 756 ms |
   | naive recompute inside `delta1` | 5.704 GiB | 15,773 ms |
   | recompute under `torch.utils.checkpoint` | 3.319 GiB | 2,360 ms |

   `build_maps` is called once per Euler step, so naive recompute makes autograd retain `K` copies
   of the transport scan — **2.1× worse memory, 21× slower**. Doing it properly with checkpointing
   is still **+20% memory and 3.1× slower** (at K=1 it is exactly break-even on memory and 64%
   slower). The estimate assumed the saving was the `[|F|, L+1, d, d]` output, but
   `compose_face_transport` takes three `[|F|, L, d, d]` inputs (`F_src_path`, `F_tgt_path`,
   `F_src_path_inv`) that are collectively ~3× larger and must stay live for backward regardless.
   This is also why §2's "memory is flat in K" holds today — precisely because maps are cached.

   **Do not confuse this with checkpointing the transport's single forward build**, which is a
   different change and *does* work: wrapping `compose_face_transport` in
   `torch.utils.checkpoint` gives imdb d=16 K=4 **5.65 -> 5.14 GiB for +3% step time**. The
   difference is that `cache_face_maps` rebuilds `K` times inside the diffusion loop, while this
   rebuilds once during backward.

   **Methodology warning, learned twice this session:** *isolated* micro-benchmarks of
   checkpointing are misleading here, because the recompute peak dominates when the checkpointed
   region is the whole graph. Isolated, checkpointing the orthogonalization looked worthless
   (1.733 vs 1.742 GiB) and checkpointing the transport looked actively harmful (3.319 vs 2.768);
   in the real model they save 1.09 GiB and 0.51 GiB respectively. **Measure checkpointing in the
   full model or not at all.**
2. **Preflight memory estimator.** Compute `0.73 KB × d × |F| × (L+1)` at dataset-build time and refuse
   early with numbers, mirroring `validate_node_edge_memory_budget`. Today a long AMPSS sweep discovers
   this by dying partway through a trial.
3. **Colab accuracy runs at d=32** — Road A vs the pre-Road-A commit, on acm/dblp/imdb. Never captured
   locally (guard rejects both at d=32; paper runs were Colab). Needed before any accuracy claim.
4. **Attribute the VRAM delta to Road A properly** — requires a pre-refactor measurement. Blocked
   locally: `git status` fails in the working tree (see §6).
5. **Cayley decision** — pending the accuracy comparison, not the memory number. **Memory is no
   longer forcing it** (WY builder shipped 2026-07-27), and the expressivity argument turned out to
   cut both ways, so decide on accuracy + throughput. The WY builder also made the first read
   runnable locally: **imdb d=16 K=4 now peaks at 6.85 GiB** (was 8.58) and acm d=8 at 7.47 (was
   8.40) — thrashy on this card but they complete, so a Householder-vs-Cayley A/B no longer has to
   wait for Colab. Keep d=32 on Colab.
6. **AMPSS instance caps** need re-tuning post-filter (30–83% of its L=4 instances were non-simple).
7. ~~**AMPSS duplicate schemas**~~ — **false alarm, closed 2026-07-27.** ACM's six groups have six
   *distinct* relation-id signatures (relations 0/1 are the two directions of the paper–paper
   citation edge); the identical `|F|` is just `max_instances_per_schema` binding on all six. What
   is real is that `FaceGroup.template_name` is built from node types only, so distinct routes log
   under the same name — a reporting defect, not a selection bug, and there is **no duplicated
   face memory to reclaim.** See [problem/ampss_issues.md](../problem/ampss_issues.md), which
   supersedes this item and carries the remaining AMPSS work (lossy names, soft-only redundancy
   penalty, PAP/PSP never selected, stale instance caps).
8. ~~**Paper-quality (not memory):** an LSTM over a 2-to-4 token sequence~~ — **done 2026-07-27.**
   Replaced by a positional-concat MLP, one per distinct path length, with schema identity entering
   through the relation embeddings (a face group fixes its relation sequence, so this costs no new
   parameters). `model.face_encoder='lstm'` retains the old encoder for the ablation.

   Profiling first showed why it was worth doing: at imdb `d=16` the sequence encoder was **35% of
   the face branch** — LSTM 417 MiB + token_mlp 261 + attention 184 = 862 MiB — to read a
   **two-element** sequence. The attention pool was also an order-agnostic weighted sum, so it
   discarded part of the order the LSTM had just injected.

   | run | lstm | concat | concat + triangular |
   |---|---|---|---|
   | imdb d=16 K=4 | 5.70 GiB / 1731 ms / 64,512 p | **5.11 / 1585 / 43,392** | 5.11 / 1627 / **27,912** |
   | acm d=8 K=1 | 6.04 GiB / 3112 ms / 49,984 p | **5.10 / 2904 / 15,552** | 5.10 / 2589 / **11,940** |

   Chain residual unchanged throughout (2.2e-08 … 2.6e-07). `face_map_param='triangular'` gauge-fixes
   `A_f` — the model consumes it only through `A_f^T A_f` — which cuts parameters again but **not**
   peak memory, since `A_f` is materialised as `d x d` either way.

   Cumulative on acm d=8 across this whole session: **8.40 → 5.10 GiB (−39%)**.
9. **Odd-`d` guard** for the Householder `det = (−1)^d` trap.

---

## 6. Environment gotchas

- **Always** run through the `h2snn` env: `"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python …`
  (MSVC activation hooks are required — `torch_householder` JIT-compiles on every import).
- **`conda run` cannot take a multi-line `python -c`** — it raises
  `NotImplementedError: Support for scripts where arguments contain newlines`. Write a file instead.
- **`git status` fails** in `g:\My Drive\H2S-NN_test` — `fatal: not a git repository`, from both Bash and
  PowerShell, despite the harness reporting a repo. So no diffing/reverting locally.
- CUDA measurements: **one dataset per process.** Memory is not reliably released after an OOM, which
  silently inflates the next measurement (this confounded my first pass).
- On Windows/WDDM a run can report `max_memory_allocated` **above** the 8 GiB physical limit — it spills
  to system RAM rather than hard-OOMing. Treat >8 GiB readings as "did not really fit."
- sklearn import intermittently fails with
  `ImportError: DLL load failed while importing _cython_blas: An Application Control policy has blocked
  this file` — environmental, unrelated, comes and goes between runs.

---

## 7. Measurement scripts

Scratchpad scripts from the original session are gone. Recreate as needed.

**Peak VRAM with knobs** (`diagnose.py <config> <d> <K> [no_delta1] [dh=N] [fh=N] [orth=name|none]`):

```python
import sys; from pathlib import Path
import torch, yaml
ROOT = Path("g:/My Drive/H2S-NN_test"); sys.path.insert(0, str(ROOT))
from src.configs.config import TrainConfig
from src.data.dataset import build_dataset
from src.models.h2snn_model import build_model

name, d, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
payload = yaml.safe_load((ROOT / "configs/experiments" / name).read_text(encoding="utf-8"))
m = payload.setdefault("model", {}); m["d"] = d; m["K"] = K
for flag in sys.argv[4:]:
    if flag == "no_delta1": m["ablate_no_delta1"] = True
    elif flag.startswith("dh="): m["d_h"] = int(flag[3:])
    elif flag.startswith("fh="): m["face_mlp_hidden_dim"] = int(flag[3:])
    elif flag == "orth=none": m["use_orthogonal"] = False
    elif flag.startswith("orth="): m["orthogonal_map"] = flag[5:]
payload["device"] = "cuda"
cfg = TrainConfig.from_dict(payload)
data = build_dataset(cfg).to("cuda"); model = build_model(cfg, data).to("cuda")
torch.cuda.reset_peak_memory_stats()
out = model(data); (out.logits.sum() + out.s_star.sum()).backward()
print(f"peak={torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")
```

**L / |F| scaling**: same, but also set
`payload["discovery"] = {"enabled": True, "mode": "auto", "min_schema_length": L,
"max_schema_length": L, "max_instances_per_schema": N, "selected_schema_k": k}`, and report
`sum(g.num_faces for g in data.face_groups)` and `sum(g.num_faces*(g.length+1) for g in ...)`.

**Chain residual on real data**: build model, call `model.encode(...)` under `no_grad`, report
`encoded.delta1_delta0_h0.norm() / encoded.delta0_h0.norm()`.

**Exact uncapped |F|** without enumerating: number of walks realising a type sequence is
`1ᵀ A₁A₂⋯A_L 1` — a chain of sparse matvecs over per-relation adjacency. (Counts include degenerate
`v₀==v_L` walks the enumerator rejects, so they are upper bounds.)

**Orthogonality / eigenvalue check**: instantiate `Orthogonal(d, kind)`, feed random
`[B, parameter_count]` float64, report `max|QᵀQ−I|`, `det`, and `min Re(eigvals)`.
