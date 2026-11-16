# Card 4 report — before-tuning checks and ablation preparation (K0–K7)

**Card:** `docs/H2S-NN_card4_ablation_prep_card.md` (amended in `design-docs` `cd771dd`, merged). **Branch:**
`ablation-prep` from `v3b` `2ef9cac` + `design-docs` `9309297` (later `cd771dd`). **Evidence:**
`docs/card4_evidence/`. All runs on the lab cluster, env `h2snn-b`. No real-data metric appears in
this report; the K3–K5 accuracies are the pre-registered known-answer outcomes on synthetic graphs.

## Short verdict
- **No card stop fired.** K1: `full` bit-identical (golden and mb0 `differing: none`, before and after
  every `src/` change). K3: no warning signal (`full` + oracle at log 512: 1.000 / 0.994 / 0.994).
- **K2:** V(adaptive) / V(uniform) = **0.193 ≤ 0.8 → `adaptive` adopted** (B-10); reported in the chat; the
  switch (`config.py:213` and the default) is left to the separate prompt.
- **K3 S1′-B:** capacity shown; C12's known-answer part **not decidable**: one `covering_only` run (log 512,
  seed 1) stopped on the T_max condition; the 8 completed `covering_only` runs are ≤ 0.536.
- **K4 S3:** audit passed (exact tag balance, probes at chance, r 2560); oracle 1.000 / 0.996 / 0.996;
  controls at chance; **`full` 0.518 / 0.530 / 0.494 → "not shown at default settings"** (not a stop).
- **K5 S2′:** σ\* strictly first on 2/3 `full` seeds (third: rank 2, −0.104) → **criterion not met** →
  C2b's S2′ part fails as pre-registered.
- **K6:** driver, arm-tuning mode, parameter table, array script done; dry run of all 17 arms ok.
  **Stop item:** the tuning launcher cannot restrict `no_path_cells`' space. **Open:** Freebase training mode
  (card 2's dry-run result not yet available).

## K0 — preconditions
`docs/card4_evidence/k0_preconditions.md`: the map operator, Jacobi input, every §2 arm (all exist in
v3-B code: no stop item), Part K's oracle hooks and the H3 gate, with their file:line. Readings decided
before any code:
- **S2 and `a_e`:** in v3-B the fact weight is the map's scale, `a_e = c_e²` (`sheaf_energy.py:300`),
  made by `edge_scale_mlp` in the same module as the frames and reflectors. "a_e unchanged" and "the
  geometry network not called" hold together only if the geometry network is the part that makes the
  orthogonal factor `R H` (frames, `reflector_mlp`); S2 keeps `c_e` (from `[h0, c]`) and sets `R H = I`.
- **Key:** `ablation.fact_maps_identity: bool` (default false). K1 first used `ablation.maps: {learned, identity}`; in K7 the full suite showed two existing tests (`tests/test_v2_model.py` `test_every_ablation_trains_one_step`, `test_defaults_are_the_full_model`) that treat every ablation field except `coupling`/`path_weight_input` as a boolean switch, so the key became a boolean in K6 (`config.py` only; no existing test edited). `ablation.identity_maps` is a removed v2 key that raises, so it was not reused.
- **Checkpoints per seed (amendment 4):** `scripts/run_h2snn_runs.py` gives every seed its own
  `exp_dir` (`exp_name = <base>_run<i>`, seed `base_seed + i`, `:112-113`; `best.pt` under
  `exp_dir/checkpoints`, `src/builders.py:90`, `src/utils/checkpointer.py:25, 62`): no overwrite, no stop
  item. The main table's seeds come from the tuning launcher's `--final_seeds`, also one folder per seed
  (`run_h2snn_wandb_tuning.py:410-411, 711-718`).

## K1 — identity-map arms S2 / S0
`IdentityMapOperator` (`src/models/map_operators.py`): stores only `c_e` `[E]`; orthogonal applications
return their input, so every transport is exactly `I`. `FactorizedNodeToEdgeSheaf(identity_maps=True)`
builds every parameter as in `full` (same RNG order) and then removes the frames and the reflector MLP;
the model's solve passes only `c_e` and the cell weights as solve parameters. Config:
`ablation.fact_maps_identity` (validated; true only with `model.arm` full or no_path_cells, not with `h0_only_readout` or
`direction_blind_maps`; refused in a tuning base config and as a search-space key).

| Test (`tests/test_card4_identity_maps.py`) | Result |
|---|---|
| (1) S2 transports `== I` (`torch.equal`, `apply`, `adjoint_prefix`, holonomy) on a synthetic ACM-shaped WISP draw with backward hops | OK |
| (2) Jacobi diagonal under S2 vs dense identity maps `c_e I` (float64, atol 1e-12) | OK |
| (3) `full` bit-identical: `v3b_golden.py compare` → `compared 1365 tensors in 24 configurations; differing: none`; `mb0_reference.py compare` → `compared 139 tensors; differing: none` (g2n01, E, job 16741) | OK |
| (4) S2 and S0 one step finite, every parameter moves (S0: g has no gradient, as in learned `no_path_cells`); frames/reflector MLP absent, `edge_scale_mlp` present | OK |
| key validation and tuning refusal | OK |

Log: `docs/card4_evidence/k1_tests.log` (24 tests OK, with `test_sheaf_energy`, `test_factorized_maps`,
`test_anchored_solve`).

## K2 — proposal variance before tuning
`scripts/ablation/variance_gate.py` (H3's estimator unchanged; `L_P = Σ_f w_f m_f L_f` is
`PathCellTerm` on the fixed model's forced transports, as H3's `apply_path`). Fixed model: ACM, seed 1,
`d` 32, `ℓ_max` 3, `r` 16, lr 1e-3, weight decay 1e-4, encoder and classifier dropout 0.5; every other
key from `configs/experiments/acm_h2snn_v3b.yaml` and the HGB NC protocol (300 epochs, patience 30,
validation Macro-F1, uniform proposal, resampled training draws, AdamW, no scheduler, no clip);
selected checkpoint **epoch 24** (early stop at 54).

| | V | distinct cells per draw (mean) |
|---|---|---|
| uniform | 158,250.1 | 40,871.9 |
| adaptive | 30,516.3 | 27,866.9 |
| **ratio** | **0.193** (reduction 80.7 %) | |

**Verdict (fixed rule, B-10, amended card):** ratio ≤ 0.8 → **`adaptive` is adopted** for every later run.
Reported in the chat at once; the switch (`config.py:213` and the default) is not made here.
Fact for the design session: `q`'s length distribution from the checkpoint is
`q_len(2) = 0.987, q_len(3) = 0.013` (`k2_variance_gate.json`); the walk-level mixture keeps every ratio
≤ 2, but the adaptive draws hold about a third fewer distinct cells. Evidence:
`k2_variance_gate.json`, `k2_variance_gate.log` (filtered: no per-epoch metric).

## K3 — known-answer S1′-B (C12, v3-B capacity)
`scripts/known_answer/s1pb.py` (+ `common.py`: frozen settings, the fixed-β oracle of Part K ported to
v3-B; the graph is `docs/WISP_evidence/h5_synthetic.graph_s1p(n = 2000)`, imported unchanged). Settings as
frozen: AdamW lr 1e-3 (weight decay 1e-4, protocol), scalar group 1e-2 constant, no scheduler, no clip,
100 epochs (patience 100: no early stop), `d` 32, `tau_init` 1, `ℓ_max` 2, `r` 640, `resample_paths: false`,
50/25/25, validation-accuracy selection. Coverage of the relevant cells 1.0 in every run (127,995 cells).
18 runs, job 16741 (g2n01, 2080 Ti). Table: `docs/card4_evidence/k3/k3_table.md`; per run JSON (with the
per-epoch log) in `docs/card4_evidence/k3/`.

| arm | β | seed 1 | seed 2 | seed 3 | rel. share of D_P (median) | dispersion (s1/s2/s3) |
|---|---|---|---|---|---|---|
| `full` + oracle | 0 | 0.518 | 0.512 | 0.474 | 0.002–0.009 | 0.87 / 0.89 / 0.92 |
| `full` + oracle | log 64 | 0.936 | 0.698 | 0.730 | 0.47–0.49 | 0.28 / 0.50 / 0.42 |
| `full` + oracle | log 512 | **1.000** | **0.994** | **0.994** | 0.87 | 0.13 / 0.16 / 0.17 |
| `covering_only` + oracle | 0 | 0.536 | 0.512 | 0.516 | 0 (by construction) | 0.87 / 0.89 / 0.92 |
| `covering_only` + oracle | log 64 | 0.518 | 0.512 | 0.514 | 0 | 0.81 / 0.87 / 0.92 |
| `covering_only` + oracle | log 512 | **stopped: T_max** | 0.520 | 0.520 | 0 | — / 0.88 / 0.90 |

Final τ 1.8–2.9 in every run; −∂L/∂β > 0 on 0.50–1.00 of the epochs before the best epoch (β = 0, `full`,
seed 3: best epoch 0, n/a). "Relevant share of D_P" is the relevant cells' share of the target's scalar
degree `D_v = Σ a_e + Σ β_f` (node-cell term; under `covering_only` cells do not enter `D_v`, so it is 0
by construction); the relevant share of the cell mass at the target (the oracle's selection ratio) is
0.016 / 0.504 / 0.890 at β = 0 / log 64 / log 512 in both arms.

**Against the fixed criteria:**
- Leak (β = 0 ≥ 0.9 in either arm): **no**.
- Warning signal (`full` + oracle < 0.9 on any seed at log 512): **no** (1.000 / 0.994 / 0.994). v3-B's
  capacity is shown on S1′-B.
- C12 known-answer part (`full` ≥ 0.9 on 3/3 at log 512 **and** `covering_only` ≤ 0.65 at every β and
  seed): **not decidable.** `covering_only` + oracle, log 512, seed 1 stopped on the spec §6 stop condition
  T_max: the backward PCG reached 100 iterations (worst relative residual 1.010e-4, tolerance 1e-4) at the
  training step of epoch 90. Under the protocol a stopped run is a failed run (spec v1.16 K5); the solver
  settings are fixed and the run was not repeated, and no accuracy is read from its partial checkpoint.
  Every completed `covering_only` run is ≤ 0.536 (the expected ceiling), so the known-answer part
  holds on the 8 completed `covering_only` runs; the ninth is missing. Decision for the author (below).
- `covering_only` ≥ 0.9 at log 512 on 3/3: no.

**Launch note (deviation, no effect on results).** The first launch ran each stream under srun with
`--cpus-per-task=2` and no `--ntasks`, so SLURM started three copies of each stream (ntasks = 6/2). They
collided (CUDA out of memory, shared run folders) and were cancelled; no output was kept (moved to
`experiments/card4_known_answer_aborted_launch1`, outside git). All 18 runs were then launched once, from
scratch, with `--ntasks=1`, settings unchanged.

## K4 — known-answer S3 (learnability of `g`)
Generator `scripts/known_answer/s3_graph.py` (seed 0): per target P, 4 A × 4 B × 4 C, each C one Q, so 64
length-4 cells P–A–B–C–Q of one schema (relations r1–r4), every intermediate inside one target. Tags on
P, B, Q; relevant iff `t_P ⊕ t_B ⊕ t_Q = 0`; one relevant cell per target (its Q carries y), 63 with parity 1
and a random class. **Tag balance by an exact design:** every A has two B of each tag; the targets are split
evenly over (t_P, relevant-B tag) and y is balanced inside each cell of that design.

**Audit before any run** (`scripts/known_answer/s3_audit.py` → `docs/card4_evidence/k4/s3_audit.json`; passed):

| Check (fixed in the script before running) | Statistic | Result |
|---|---|---|
| (i) tag balance: own tag + per-(relation, role) context tag means, relevant vs other nodes | TV distance per type a / b / c / q | 0 / 0 / 0 / 0 (pass, ≤ 1e-12) |
| (ii) fact-input probe `[x_u, c_u, x_v, c_v, rel]`, per hop, training targets split in halves (K0 solver, λ 1e-3); n_test = 1,000 per hop | held-out accuracy r1 / r2 / r3 / r4 | 0.506 / 0.492 / 0.500 / 0.503, all in the binomial 99 % chance interval [0.459, 0.541] (pass) |
| (iii) reachability | b, θ_τ move ≤ Σ lr = 1.0; gap ≈ log 512 = 6.24 must come from `g` | stated |
| `r` (smallest of 640, 1280, 2560, … with coverage 1.0 of the 64 length-4 cells of every target, seeds 1–3) | | **r = 2560** (cells per draw: 32,000 / ≈128,000 / 128,000 of length 2 / 3 / 4) |

**Runs** (frozen settings, `ℓ_max` 4, 12 runs; `docs/card4_evidence/k4/k4_table.md`):

| arm | seed 1 | seed 2 | seed 3 | mean log-w gap, length-4 cells (relevant − irrelevant) |
|---|---|---|---|---|
| `full` + oracle (β = log 512) | 1.000 | 0.996 | 0.996 | 6.238 (= β) |
| `full` | 0.518 | 0.530 | 0.494 | 0.014 / 0.018 / 0.011 |
| `endpoints` | 0.496 | 0.542 | 0.496 | 0.006 / −0.003 / −0.002 |
| `uniform` | 0.490 | 0.488 | 0.532 | 0 |

**Against the fixed criteria:** no control ≥ 0.9 and the probe at chance → no leak; `full` + oracle ≥ 0.9
(capacity shown on this graph); **`full` ≤ 0.65 on all seeds → "not shown at default settings"** (not a
stop condition). With the C2 kill rule, the spec §6 / pre-registration §5 reading applies once the real-data
ablations exist. The all-cells gap (2.3 / 2.9 / −1.0 for `full`) compares length-4 relevant cells with shorter
cells of other projections and is not the S3 quantity; it is in the JSONs only.
Note: final τ reached 3.37 (`uniform`, seed 1; log τ = 1.21), above the pre-registered reach of 1.0 per
scalar (Σ lr). DERIVED: an Adam step can exceed the learning rate (|m̂/√v̂| up to (1 − β₁)/√(1 − β₂) ≈ 3.2
when the gradient shrinks), so Σ lr is not a strict bound; the gap needed (6.24) is still out of reach for
b and θ_τ, and no verdict changes.

**Runs and fixes.** K4/K5 ran from a detached snapshot worktree (`/mnt/data/peeradon/h2snn-ablprep-run`,
`src` = `f1601b8`, scripts copied, sha256 in every log) so that K6's `config.py` edits never reached a run;
tasks ran in job 16741 and arrays 16869/16884 (3090, g3n01). Two S3 runs started together OOM'd on the
2080 Ti in their first epochs (one S3 run needs ≈ 7 GB) and were discarded before any result; S3 then ran
one at a time. The oracle's relevance was keyed by group size (as Part K); at r = 2560 the length-3 and
length-4 groups both hold 128,000 cells on seeds 1 and 3, so those two oracle runs refused to start
("the oracle needs distinct group sizes"); the oracle is now keyed by cell length (same oracle) and those
two runs were run once. Seed 2's oracle run (distinct sizes) is the original.

## K5 — known-answer S2′ (C2b read-out), option (a)
`scripts/known_answer/s2p.py`: K2b's graph `i4_s2_unconfounded.graph_s2u(n = 2000)` unchanged, `r` 200,
I4's read-out ported (w_f from the checkpoint's `g` on the evaluation draw). `docs/card4_evidence/k5/k5_table.md`.

| arm | seed | test acc | σ\* coverage of the draw | strict rank of σ\* (mean log w) | σ\* − best other |
|---|---|---|---|---|---|
| `full` | 1 | 0.998 | 0.9995 | 1 | +0.533 |
| `full` | 2 | 1.000 | 1.0000 | 1 | +0.132 |
| `full` | 3 | 1.000 | 1.0000 | **2** | −0.104 (sibling `a b c e`) |
| `uniform` | 1 / 2 / 3 | 0.896 / 0.912 / 0.942 | 1.0 | 11 / 10 / 11 (all tie) | 0 |

**Against the fixed criterion** (σ\* strictly highest on 3/3 `full` seeds): **not met** (2/3). By
pre-registration B-8 / spec 1.23 X-2b, C2b then fails ("S2′ fails" → the read-out is reported without
interpretation, no interpretability claim). `uniform`: selection not required on this graph. Note: the
pre-registration states r = 200 gives coverage 1.0 (measured in Part K on seeds 42–44); with seeds 1–3, seed
1's evaluation draw misses σ\* for one of 2,000 targets (0.9995). The setting was not changed after the run.


## K6 — ablation driver (`scripts/ablation/`)
| Piece | File | State |
|---|---|---|
| (1) configurations (a): `full`'s configuration + one change, per arm | `arms.py` | 17 configurations: the 15 arms of §2 (incl. S2, S0) + the appendix pair (`appendix_full`, `appendix_endpoints`, `path_weight_context: true`); the fixed-literature-schema row excluded (amended card). Each is built by `TrainConfig.from_dict` |
| (2) configurations (b): base config + `tuning.arm` + sweep definition | `sweeps.py`; arm-tuning mode in `src/configs/config.py` | five arms; the launcher (not edited) accepts every (b) base config (`--validate_only`, `docs/card4_evidence/k6/k6_emit.log`). **Stop item:** the launcher builds its space itself; for `no_path_cells` the restricted space (without `wisp_max_length`, `wisp_walks_per_anchor`, which do not act there) cannot be passed. Removed keys: `no_path_cells` → those two; `endpoints`, `closure_only`, `instance_features`, S2 → none (every key acts) |
| (3) seed runs: seeds 1–10 through `run_h2snn_runs.py`; `full` seeds 1–4 looked up | `seed_runs.py` (`manifest.tsv`, `lookup.json`) | ACM manifest emitted with the placeholder `full` config: 27 rows (5 `tune`, 17 (a) `runs`, 5 (b) `runs`); `full` (a) runs seeds 5–10; seeds 1–4 at `<exp>/final_seeds/<exp>_tune_final_seed<s>_final_seed<s>/`. Every seed has its own `exp_dir` with `best.pt` and `config.yaml` (K0). A (b) `runs` row needs `tuned_config()` (the sweep's `best_tuned_run0.json` merged by the launcher's `apply_sweep_overrides`) before it runs |
| (3b) Freebase training mode | `seed_runs.py --freebase_mode` | **Open:** card 2's dry run has no Freebase-at-largest-configuration result yet (`origin/dry-run` `e419370`: D0/D1 only, no `docs/dryrun_evidence/cells.csv`). The driver refuses Freebase without an explicit `--freebase_mode`; set it from card 2's D-result (pre-registration §1: full batch if it fits) |
| (4) parameter table | `param_table.py` → `docs/card4_evidence/k6/param_table.json` | below |
| (5) SLURM array | `ablation_array.sbatch` | `bash -n` OK; `sbatch --test-only --array=1-27 … manifest.tsv` → "Job 16877 to start at 2026-11-17T08:48:41 … g2n01" (accepted; nothing submitted) |

**Arm-tuning mode** (`config.py`: `TuningConfig.arm`, `ARM_TUNING_KEYS`, `reject_never_tuned_settings`):
`tuning.arm ∈ {endpoints, closure_only, instance_features, no_path_cells, S2}`; under it exactly that
arm's key and value are accepted (and required), every other arm key, `path_weight_context`, `path_source:
schemas`, `adaptive` and the analysis flags are refused as before; without it every arm key is refused.
The launcher passes the flag through the base config file (no launcher change).

**Tests** (`tests/test_card4_ablation_driver.py`, with the K1 tests, `test_v2_model`, `test_search_spaces`,
`test_legacy_removal`, `test_notebook_configs`: 56 tests OK after the key rename, with golden and mb0
`differing: none`; `docs/card4_evidence/k6/k6_tests.log`):
(a) configurations differ from `full`'s in exactly the arm's keys (all 17); pairing: `full` and `uniform`
at one seed have identical evaluation and epoch-1 training draws (`torch.equal`; full batch has no batch
order); (i) a main-table tuning base config with any arm key (6 `model.arm` values, 3 ladder rungs,
`maps: identity`) is refused; (ii) arm-tuning mode with another arm's key (all 20 pairs), without its own
key, or naming a non-tuned arm is refused; (iii) arm-tuning mode builds for each of the five arms
(validation, `TrainConfig`, the launcher's per-trial merge).

**Parameter table** (placeholder configurations: `configs/experiments/{acm,dblp,imdb}_h2snn_v3b.yaml` and
card 2's Freebase D1 file copied to `docs/card4_evidence/k6/freebase_placeholder.yaml`; `d` 32, hidden 128;
re-run on `full`'s tuned files). Totals and the change from `full`:

| arm | ACM | DBLP | IMDB | Freebase | part removed / added |
|---|---|---|---|---|---|
| `full` | 488,678 | 359,895 | 792,248 | 7,089,226 | g 71,553; map network (frames + reflector MLP) 88,384 / 71,008 / 71,008 / 644,416; fact-weight MLP 76,033 / 59,649 / 59,649 / 600,321; head input 8,320 |
| `endpoints` | +13 | +13 | +13 | +13 | g re-widthed: 71,566 vs 71,553 (+0.02 %, within 5 %) |
| `schema` | −49,152 | −49,152 | −49,152 | −49,152 | g 22,401 (share of model 0.051 ACM) |
| `uniform` | −71,552 | −71,552 | −71,552 | −71,552 | g = one scalar b |
| `no_path_cells`, `closure_only`, `hub_A`, `covering_only`, `direction_blind_maps` | 0 | 0 | 0 | 0 | none (under `no_path_cells` g exists but receives no gradient) |
| `instance_features` | +4,096 | +4,096 | +4,096 | +4,096 | head input 2d → 3d |
| S2, S0 | −88,384 | −71,008 | −71,008 | −644,416 | map network removed (S0: g present, no gradient) |
| `free_cell_maps` | +64,128 | +55,936 | +55,936 | +326,272 | free incidence maps |
| geometry context off | −131,072 | −98,304 | −98,304 | −1,179,648 | map and fact-weight MLPs read `h0` only |
| `h0`-only | −4,097 | −4,097 | −4,097 | −4,097 | head input 2d → d, no θ_τ |
| appendix `full` / `endpoints` | +393,216 / +392,780 | +294,912 / +294,527 | +294,912 / +294,527 | +3,538,944 / +3,532,962 | g reads `[h0, c]` |

Configurations, sweeps and manifest were emitted again after the key rename (job 16886,
`docs/card4_evidence/k6/k6_emit.log`); the parameter table ran before the rename (key name only; the
counts do not depend on it).

**One-epoch dry run** of every (a) arm on ACM (placeholder configuration, job 16886;
`docs/card4_evidence/k6/dry_run_acm.json`): all 17 `ok`; peak GPU memory 0.2–3.0 GB; no metric read.

## K7 — suite and report
Full suite once on one node at the committed head `763adc4` (clean tree; job 16887, g2n01, pinned kernel
environment E): **`Ran 706 tests … OK`**; `v3b_golden.py compare` → `compared 1365 tensors in 24
configurations; differing: none`; `mb0_reference.py compare` → `compared 139 tensors; differing: none`
(`docs/card4_evidence/k7_suite.log`, filtered of INFO lines). The earlier suite run at the pre-rename state
(job 16741) had the two `test_v2_model` failures described under Deviations 1; the rename fixed them.

## Deviations (all stated; none changes a graph, seed, setting or criterion after a run)
1. **S2/S0 key** is the boolean `ablation.fact_maps_identity` (card example: `ablation.maps: identity`): a
   string key broke two existing `test_v2_model` tests that iterate the boolean ablation switches; renamed
   in `config.py` instead of editing those tests.
2. **K3 first launch aborted** (srun started three copies of each stream; OOM and shared folders); all 18 runs
   then launched once from scratch.
3. **K4 oracle key** changed from group size to cell length after two oracle runs refused to start
   (identical oracle; those two runs were run once). Two S3 runs that OOM'd when started together were
   discarded before any result.
4. **Commit order:** K5 (`7f3b918`) was committed before K4 (`49dd93b`) because K4's last oracle run was still
   queued; K6 after both.
5. **Job scripts:** `scripts/slurm/run_tests.sbatch` changes into the other agent's worktree, so this card
   used its own copies under `docs/card4_evidence/jobs/`. K4/K5 ran from a detached snapshot worktree
   `/mnt/data/peeradon/h2snn-ablprep-run` (at `f1601b8`, not a branch; left in place for the review, removable
   with `git worktree remove`).
6. **Logs with real-data metrics** (K2 training, K6 dry run) are committed only filtered; the raw logs stay in
   `H2S-NN/logs/` (git-ignored).

## Decisions for the author
| # | Question | Options | Recommendation and reason |
|---|---|---|---|
| D1 | `adaptive` adoption (K2, ratio 0.193) | (a) apply B-10 as fixed: switch default and `config.py:213` in the separate prompt; (b) look first at why: `q_len(3)` = 0.013, adaptive draws hold ≈ 32 % fewer distinct cells | (a): the gate and its consequence were fixed before the run; (b)'s facts are for the record (the defensive mixture keeps every ratio ≤ 2) |
| D2 | C12 known-answer part not decidable (one `covering_only` run stopped on T_max) | (a) report C12's known-answer part as "8 of 9 `covering_only` runs ≤ 0.536, one stopped (T_max)", decide C12 on the real-data comparison; (b) an author-approved extra seed (4) for that cell; (c) a solver change (not allowed by the protocol) | (a): no setting may change after a run; the completed runs all sit at the predicted ceiling |
| D3 | S3 `full` at chance ("not shown at default settings") | as pre-registered: the C2 × S3 table applies when the real-data C2 result exists | no action now; a warning signal is not triggered by the pre-registration |
| D4 | S2′ fails (rank 2 on seed 3) → C2b fails as pre-registered | (a) accept: no interpretability claim for the schema read-out; (b) nothing else allowed post hoc | (a) |
| D5 | `no_path_cells` (b) space: the launcher cannot drop `wisp_max_length`, `wisp_walks_per_anchor` | (a) a one-option launcher change (`--exclude_params`) in a later card; (b) tune `no_path_cells` with the two inert keys (budget wasted, results valid) | (a), small and keeps §1's rule |
| D6 | Freebase training mode | set from card 2's Freebase-at-largest-configuration result | `seed_runs.py --freebase_mode` refuses to guess |
| D7 | Two arm switches pass a main-table tuning base config | `ablation.direction_blind_maps: true` and `ablation.h0_only_readout: true` are refused only as search-space keys (`NEVER_TUNED_LEGACY_KEYS`), not in the base config (`reject_never_tuned_settings` never reads them; existing behaviour, outside this card's config scope) | refuse both in the base config in a later card (same pattern as the other arm keys) |

branch=ablation-prep sha=<see final message> card=docs/H2S-NN_card4_ablation_prep_card.md base=2ef9cac14ae03136efb07ad0c71233697ae138b1
