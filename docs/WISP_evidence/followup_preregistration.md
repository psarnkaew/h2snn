# Part I pre-registration (I0, 2026-11-09; written before any Part I code)
Authority: spec v1.4 (E1–E7), task card Part I, and the author's decisions of 2026-11-09:
1. flip the default (I2);
2. `g` reads `h₀` only in every ladder arm, while maps and gates keep `[h₀, c]` (I1);
3. do Part I;
4. push `d12-encoder` and `wisp` as backups, with no merge. Done in I0: `origin/d12-encoder` = `19a9d36`, `origin/wisp` = `94fc397`.

Branch `wisp-followup` starts at `94fc397`. Synthetic accuracies are pass/fail functional checks. A missed gate or expectation is recorded with numbers and a diagnosis, and the run continues. The stop conditions are the card's.

## Decisions fixed now (where the card leaves a choice)
- **D-I1a, one input rule for the path latent.** The path encoder (the first layer of `g`) reads `h₀` unless `ablation.path_weight_context: true`. This applies in every path mode, since the latent is `g`'s input everywhere:
  - in `endpoint_edges` and `free_path_maps`, their path-specific maps are built from the same latent, so they also read `h₀`; `free_maps`' per-node input keeps `[h₀, c]`;
  - the pair-attention arm's path latents also read `h₀`.
  - Fact maps, listen/broadcast gates and the context `c` are unchanged.
  - Alternative not taken: keeping `[h₀, c]` for the latent of those comparison arms. It would let their weight see `c`, against R12.
- **D-I1b, `path_weight_context: true` is allowed only** with `geometry_context: true` and `path_weight_input ∈ {full, endpoints}` (the spec's appendix pair). Otherwise it is rejected loudly. It is rejected in a tuning base config and in a search space.
- **D-I2a, the default `path_source` resolves by task when the key is not given:** `wisp` for node classification, `schemas` for link prediction and graph classification. Their samplers ignore it, and `wisp` is rejected for them. An explicit value is kept and validated as before.
- **D-I2b, `r` per dataset (spec §6: the incidence rule).** The node-classification presets carry `wisp_walks_per_anchor`: ACM 42 (H1), DBLP 65, IMDB 12 (`h6_calibrate_r.json`). They apply only under `wisp` and only when `r` is not given. These three are all the node-classification datasets.
- **D-I2c,** following the card's rule "comparison arms are never tuned": `path_source: schemas` for a node-classification tuning base config is rejected.

## I1: `g` reads `h₀` only
- **Change:**
  - the path latent's input becomes `h₀` (D-I1a);
  - the encoders' input width is `d`, and the `endpoints` width rule uses `node_dim = d`;
  - new switch `ablation.path_weight_context` (default `false`).
- **Tests:**
  - T-H7 rewritten under the **default** context (`geometry_context: true`). Perturbing an intermediate's `h₀` together with the `h₀` of all its neighbours outside the instance leaves `endpoints`, `schema` and `uniform` unchanged and changes `full`.
  - Perturbing a node outside the instance changes no arm.
  - Positions ℓ−1 and ℓ are perturbed separately: ℓ−1 is unchanged for `endpoints` and ℓ changes it, on length-2 and length-3 cells.
  - `path_weight_context: true` restores the v1.3 behaviour: the `endpoints` weight changes when an intermediate's neighbourhood moves.
  - The capacity rule still holds within 5 %. T-H5 and T-H6 still pass. The goldens (`geometry_context: false`) are bit-identical.
- **Mutations:**
  - the far-end off-by-one (`[:, [0, length-1]]`) must fail T-H7;
  - feeding `[h₀, c]` to `g` by default must fail the rewritten T-H7.
- **Expected:** all pass.
- **Wrong if:** any of the above fails after 3 diagnosed attempts (a stop), or a golden changes.

## I2: default flip
- **Change:** D-I2a, D-I2b, D-I2c. Tests and the golden harness that use schema-cell fixtures name `path_source: schemas` explicitly.
- **Tests:**
  - a default-config test: ACM, DBLP and IMDB resolve to `wisp` with r = 42 / 65 / 12;
  - link prediction and graph classification resolve to `schemas`;
  - an explicit `schemas` is kept and resolves its templates;
  - a `wisp` config round-trips through `save_config`;
  - the goldens are bit-identical.
- **Expected:** pass. No other test changes behaviour except by pinning.
- **Wrong if:** a golden changes, or a link-prediction or graph-classification config fails to load.

## I3: S1′ with a learnable-scale oracle and the learning mechanism
- **Graph:** H5's S1′ unchanged (500 targets, 8 authors × 8 cpapers, 4,000 decoys, 6-bit parity, 1 relevant instance in 64). Same split; `resample_paths: false` as in H5, so cell relevance is fixed per run.
- **The oracle:** `w_f = exp(b + β·o_f)`, with `o_f ∈ {0, 1}` the known relevance of cell f.
  - `b` and `β` are learnable scalars in the scalar optimizer group. `b₀ = log(path_weight_init)` = log 0.5 and `β₀ = 0`.
  - `g` is not used for weights in this arm. Everything else trains as in `full`.
  - Nothing is clamped.
- **Arms:**
  - `full` and the oracle, 3 seeds each;
  - `endpoints`, `uniform`, `no_path_cells`, and `full_no_h0` (the readout sees `[x_K, 0]`, i.e. without the `h₀` half; G0b hypothesis H1), 1 seed each;
  - all at the v1.4 defaults (`g` reads `h₀`).
- **Per-epoch log** (evaluation draw, frozen at each epoch end, no gradient):
  - the mean log-weight of relevant vs irrelevant cells;
  - the median over targets of the relevant cells' share of the target's scalar degree `D`, and of the path part's share of `D`;
  - train/val accuracy; `b` and `β` for the oracle.
- **Budget:** ≤ 100 epochs, validation selection, about 1 minute per run. `log_every` is set high (diagnostics only at epoch 0) to pay for the per-epoch log; the size is unchanged unless a run takes more than about 1 minute.
- **Expected:**
  - **Oracle ≥ 0.9 on 3/3 seeds** (capacity). Mechanism: β grows, the relevant cell's share of its target's `D` rises from about 0.03 % toward a dominant share, and normalisation by `D` suppresses the 64-instance edge route.
  - **`full` ≤ 0.65** with a relevant − irrelevant log-weight gap ≈ 0 (no learned selection; as in H5).
  - **`full_no_h0` ≤ 0.65.**
  - **Controls ≤ 0.65** (near chance; the leak rule as in Part H).
- **Wrong if:**
  - the oracle is < 0.9 on any seed. Then I diagnose structurally (the relevant share of `D`, the path vs edge part, β's trajectory) and do not redesign;
  - or `full` ≥ 0.9 (my mechanism story would be wrong);
  - or a control ≥ 0.9 (a leak).

## I4: S2 without the length confound
- **Graph** (one carrier type C with class one-hot plus noise; every other type is noise only):
  - Types: T, M1, M2, M3, N1, N2, N3, C.
  - Per target, one chain of each: T –a– M1 –b– M2 –c– M3, and T –f– N1 –g– N2 –h– N3. Plus a cross edge M2 –m– N3, and leaves M3 –d– C\* (class y), M3 –e– Cₑ (random class), N3 –k– Cₖ (random class).
  - The length-4 schemas in the draw: σ\* = a b c d, and the distractors a b c e, a b m h, a b m k, f g h k, f g h m. That is 6 schemas of length 4; only σ\* reaches the labelled carrier.
  - There are no dead ends before length 4.
  - Under `p`, σ\* is 1/8 of the length-4 walks, i.e. 1/24 of all walks (DERIVED: from T, 2 actions; at M2, {c, m}; at M3, {d, e}).
- **Arms:**
  - `full`, 3 seeds;
  - `uniform`, 1 seed, as a non-vacuity control: constant `w`, so every schema ties and σ\* must **not** be strictly first;
  - `ℓ_max = 4`; r large enough that σ\* is drawn for every target (coverage reported).
- **Criteria:**
  - **primary:** σ\* has the strictly highest mean learned log-weight per cell among all full-walk schemas (ties count against σ\*);
  - **secondary:** the strictly highest lift of `I_π`;
  - raw `I_π`, `I_σ` and the per-schema table are reported.
- **Expected:** primary met on 3/3 `full` seeds; `uniform` not strictly first.
- **Wrong if:** any `full` seed ranks another schema ≥ σ\*, or `uniform` ranks σ\* strictly first (a criterion bug).

## I5: test gaps from the H7 review
- **The tests:**
  - **Trainer-level:** an adaptive training draw (epoch ≥ 1) returns groups that differ from the evaluation draw, and leaves `prepared_data.face_groups` (the evaluation cells) bit-identical. The evaluation epoch uses those cells.
  - **Config validation in `__post_init__`:**
    - `ModelConfig`: `gamma_init ≠ 1` under `exp` and `path_weight_init ≠ 0.5` under `sigmoid_gamma` are rejected on direct construction.
    - `DataConfig`: the value checks of `path_source`, `wisp_max_length`, `r` and `wisp_proposal`; non-default `wisp_*` values under `path_source: schemas` are rejected.
  - **One test per never-tuned key,** in the search space and in the base config where it is an arm: `path_source`, `wisp_max_length`, `path_weight`, `path_weight_init`, `path_weight_input`, `wisp_proposal`, `path_weight_context`. The error message names `wisp_proposal`.
  - **T-H2 transported:** the Monte-Carlo mean of the dense transported `L̂_P = Σ m_f B_fᵀ(I − 11ᵀ/|f| ⊗ I_d)B_f` equals the exact enumeration within 4 s.e. (float64, fixture maps from `_chain_operators`), for `q = p` and for a biased `q`.
- **Mutations:** each new test fails under a targeted mutation:
  - evaluation on the training draw;
  - a `from_dict`-only check;
  - a dropped key in the rejection;
  - the IS ratio removed in the transported T-H2.
- **Expected:** pass, and every mutation is caught.

## I6: overlap sampling
- **Change:**
  - `Trainer.fit` draws epoch e+1's cells in one background thread while epoch e trains (node-classification trainer).
  - The draw seed stays `cfg.seed + e + 1`, and the CPU groups are moved to the device in the main thread.
  - Under `wisp_proposal: adaptive`, the next draw is submitted after the training step, which fixes `last_path_weights`. It then overlaps only with evaluation.
  - Switch: `data.overlap_sampling: true` (default). It changes no semantics and exists for the identity test and the measurement.
- **Bit-identity test** (a stop condition if it fails): on the fixture, 3 epochs with overlap on vs off, for the uniform and adaptive proposals. Every training draw's cells and `m_f`, and the training logits of every epoch, must be bit-identical (`torch.equal`).
- **Measurement:** ACM, DBLP, IMDB, `full`, at the D-I2b `r`, 2 epochs each. The exposed sampling time is the main thread's wait for the prefetched draw plus the device copy, as a share of epoch time.
- **Gate:** ≤ 25 %.
- **Expected:** pass on all three. Draws of 0.4–2.0 s (H6) overlap with the training step plus 2 evaluation forwards.
  - Risk: DBLP's draw (2.0 s) is longer than its training step (1.4 s). CPU and GIL contention may also slow both threads.
- **Wrong if:** > 25 % on any dataset; that is recorded with the diagnosis.

## I7
Report `docs/H2S-NN_WISP_followup_report.md` (Part A format, ≤ 2 pages). The full suite runs once, at I7. Commit; no push.
