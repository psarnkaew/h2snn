# Report — Cluster setup (CS0–CS6)

**Branch** `cluster-setup` from `part-mb` at `ccbcf993548b47a6de07c8a50e0cf2f00aa47a24` (checked after `git fetch`; VERIFIED-RUN `git rev-parse HEAD`). One commit per unit, each pushed (`git push origin cluster-setup`); `v3b` not touched. Evidence: `docs/cluster_evidence/` (each job's script `csN_*job.sh` next to its output). Date 2026-11-14.

## Verdict

- **Done:** CS0–CS6. Torch 2.6.0+cu124 uses CUDA in `h2snn` (no stop). Server `data/` equals the laptop manifest (215/215 files) and the Part M M1 gate passes 69/69.
- **Not complete, reported (no guess, no reinstall):** `dgl` and `torch_householder` are **not installed** in `h2snn`; `data.dgl.ai` is unreachable (TLS EOF) from the login node and from a compute node. Four vendors have **no URL in the sources the card names** (PSHGCN, ChebNetII, Cooperative-Sheaf-NN, hgbi), so they are not cloned.
- **Full suite (job 16615):** 689 tests; 15 failures, 12 errors, 2 skips. Every failure has one of four causes: missing `dgl` / `torch_householder`, missing PSHGCN/ChebNetII clones, the known notebook check, or a **1-ulp cross-platform difference** in three bit-exact tests (golden file captured on Windows). No failure points to H2S-NN code. Nothing was fixed, loosened or skipped.
- No method code changed; no performance run; no metric reported.

## CS0 — Preconditions (commit `b22d120`)

**Done.** VERIFIED-RUN `srun --gres=gpu:1 --mem=8G --time=0:20:00 --cpus-per-task=2 bash docs/cluster_evidence/cs0_job.sh` → `cs0_preconditions.txt`. Job **16606**, node **g2tn01**, GPU **NVIDIA TITAN RTX** (24 GB).

| Check | Result |
|---|---|
| Driver / CUDA | 550.163.01 / 12.4 |
| gcc / g++ | Debian 14.2.0-19 (both present) |
| Python (`~/.conda/envs/h2snn/bin/python`) | 3.12.16 |
| torch | 2.6.0+cu124, `cuda_available True`, a CUDA tensor sums to 3.0 |
| torch_geometric / scatter / sparse / cluster | 2.8.0 / 2.1.2+pt26cu124 / 0.6.18+pt26cu124 / 1.6.3+pt26cu124 |
| dgl | **`ModuleNotFoundError`** (not installed; so not the `+cu124` build) |
| torch_householder | **`ModuleNotFoundError`** |
| `pip freeze` | 104 lines → `server_pip_freeze.txt` |

**DGL status.** `curl -sI https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html`: compute node g2tn01 `curl_exit=35`; login node login01 `000 TLS connect error: error:0A000126:SSL routines::unexpected eof while reading`, `curl_exit=35` (VERIFIED-RUN, both in `cs0_preconditions.txt`). Same failure as the author's of 2026-11-14. Per the card, the author copies `dgl-2.5.0+cu124-cp312-…manylinux…x86_64.whl` from the laptop; not reinstalled here.

**Deviations.** The check ran as a non-interactive `srun … bash <script>` (this agent cannot drive `--pty`); the job script is kept in evidence because a compute node cannot see the login node's `/tmp` (first attempt failed with `No such file or directory`). `_dgl_cuda_defect()` called directly printed "no CUDA support" although DGL is absent; in normal use it runs only after `dgl` resolves (`dependencies.py` group check), so this is not a finding.

## CS1 — Cluster section, SLURM scripts (commit `2a988aa`)

**Done.**
- `H2S-NN/CLAUDE.md`, `H2S-NN/AGENTS.md`: new section "Lab GPU cluster (Linux, SLURM)" after the Windows "Environment" section (Windows section unchanged): paths, storage, secrets in home, login-node rule, `--mem`/`--time` ≤ 72 h, GPU types, Miniforge line and why not `/opt/conda`, `/tmp` scratch, version lock, how to run tests, push rules (2026-11-14). Identical text in both files.
- `H2S-NN/scripts/slurm/run_tests.sbatch`: `--gres=gpu:1 --cpus-per-task=8 --mem=32G --time=4:00:00`, output `/mnt/data/peeradon/h2snn/H2S-NN/logs/slurm-%j.out`; `cd /mnt/data/peeradon/h2snn/H2S-NN`; `. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn`; arguments = unittest targets, default `discover -s tests -v`; prints job, node, git hash, GPU, torch/CUDA; exits with the suite's status.
- `H2S-NN/scripts/slurm/interactive.md` (the `srun … --pty bash` line and rules).
- `H2S-NN/.gitignore`: `/logs/*`, `!/logs/.gitkeep`; `H2S-NN/logs/.gitkeep` (SLURM does not create a missing output directory).
- `H2S-NN/tests/test_slurm_scripts.py` (2 tests): the script has `--mem`, `--time`, `--cpus-per-task`, `--gres`, the Miniforge line, `conda activate h2snn`, a `cd` into the repository, no `/opt/conda`, no `cd $HOME`; a second test checks the checker catches each violation.

**Tests (VERIFIED-RUN).** Login node: `bash -n` → exit 0; `sbatch --test-only scripts/slurm/run_tests.sbatch` → `Job 16607 to start … using 8 processors on nodes g2n01 in partition lab`. Job **16608** (g2tn01, TITAN RTX): `tests.test_slurm_scripts` → Ran 2, OK. **Mutation** (`#SBATCH --time` deleted in place, then restored): `AssertionError: Lists differ: ['missing #SBATCH --time'] != []`, `FAILED (failures=1)`; restored, `git diff` empty (`cs1_tests.txt`).

**Deviation.** The section names `requirements/linux-lock.txt` and `check_env.py`, which land in CS2 (the next commit).

## CS2 — Lock and environment check (commit `e7dda80`)

**Done.** `H2S-NN/requirements/linux-lock.txt`: the 27 exact pins of the card (torch cu124 index, PyG wheel index, DGL index); `numba` and HGB's tensorflow/gensim/rdflib/pynvml not pinned. `H2S-NN/scripts/setup/check_env.py`: reads the lock (`name==version`; options, comments, markers dropped; a non-exact pin raises), compares with `importlib.metadata` (PEP 503 names), prints `MISSING`/`DIFFERENT` lines and exits 1 on a difference. **Rule for local build tags:** an installed version equal to the pin plus a tag the pin does not state (`2.6.0+cu124` for `torch==2.6.0`) is printed as a `note`, not a difference; a pin that states a tag (`dgl==2.5.0+cu124`) must match exactly. `H2S-NN/tests/test_check_env.py` (9 tests: parse, equal, missing, different, extra local tag, locked tag must match, wrong version with tag, non-exact pin rejected, repository lock parses to 27 pins).

**Run (VERIFIED-RUN)** job **16609** (g2tn01, TITAN RTX), `cs2_check_env.txt`: `tests.test_check_env` Ran 9, OK; `python scripts/setup/check_env.py` → `differences: 2; notes: 4`, `check_env_exit=1`.

### Version differences (server vs lock)

| Package | Lock | Server | Kind |
|---|---|---|---|
| dgl | 2.5.0+cu124 | — | **missing** |
| torch_householder | 1.0.1 | — | **missing** |
| torch | 2.6.0 | 2.6.0+cu124 | note (local tag) |
| torch_scatter / torch_sparse / torch_cluster | 2.1.2 / 0.6.18 / 1.6.3 | +pt26cu124 each | note (local tag) |
| Python | 3.12.13 (laptop) | 3.12.16 | patch level; not in the lock |

All other 21 pins equal. Not reinstalled (card).

## CS3 — Vendor manifest and clones (commit `c0bfc63`)

**Done, with four vendors unresolved.**
- URLs were taken only from the sources the card names: code/docs grep (`github.com/...` in `src/`, `scripts/`, `docs/`, notebooks) and the reference PDFs. The `h2snn` env has no PDF reader and nothing was installed, so a standard-library scan of each PDF (raw bytes and FlateDecode streams) ran in job **16610/16611** (`cs3_pdf_urls_job.sh` → `cs3_pdf_urls.txt`).
- Clones made on the login node with git only (allowed there). Laptop-pinned vendors at the laptop commit; empty-on-laptop vendors at the latest default-branch commit.
- New `H2S-NN/vendor/VENDORS.tsv` (columns `name, url, commit, licence, used_by, kind`), `H2S-NN/scripts/setup/clone_vendors.sh`, `H2S-NN/tests/test_vendor_manifest.py` (4 tests).

### Vendor table

| Vendor | URL (source) | Commit | Licence | Used by | One-epoch / test result on the server |
|---|---|---|---|---|---|
| HGB | github.com/THUDM/HGB (HGB.pdf, code) | `ca6fd5bb0c1c…` | none (never committed) | hgb_gnn, hgb_han (GCN, GAT, HAN; LP oracle) | `test_hgb_oracle` 7/7 OK; GCN, GAT LP one-epoch **pass**; GCN, GAT NC and HAN NC/LP fits: `No module named 'dgl'` |
| SeHGNN | github.com/ICT-GIMLab/SeHGNN (LMSPS paper, references/metapath_search) | `e92bd37d0b80…` (latest master) | none | sehgnn | clone present; no one-epoch fit in these tests (Freebase excluded by D-P8) |
| H2SGNN | github.com/Lukangkang123/H2SGNN (its paper, references/background) | `93c33faf3a41…` (latest main) | none | h2sgnn | LP one-epoch **passes** |
| Hetero2Net | github.com/EdisonLeeeee/Hetero2Net (paper, code) | `49133947f33b…` (laptop) | none | hetero2net | NC Freebase, LP, data-root one-epoch **pass**; heterophily tests OK |
| H2GB | github.com/junhongmit/H2GB (paper, code) | `d3b49db63ad6…` (laptop) | MIT | h2gformer | NC, LP, width tests **pass**; heterophily tests OK |
| neural-sheaf-diffusion | github.com/twitter-research/neural-sheaf-diffusion (NSD paper) | `11e21b561d88…` (latest master) | Apache-2.0 | nsd | NC, LP: `No module named 'torch_householder'` |
| HINormer | github.com/Ffffffffire/HINormer (LMSPS paper, code) | `a29fcace78c6…` (in repo, `CLONED_COMMIT.txt`) | none | hinormer | NC Freebase: `No module named 'dgl'` |
| csnn | in this repo | `b0e0fab45868…` (last change here; upstream not recorded) | — | csnn | NC, LP: `No module named 'torch_householder'` |
| HINPool | in this repo | `b0e0fab45868…` (as csnn) | — | campaign (HINPool) | — |
| Heterogeneous-Sheaf-Neural-Networks | github.com/alessioborgi/… (HETSHEAF.pdf) | `6964e1e79a4c…` (laptop) | **CC BY-NC-ND 4.0** | none yet (later card) | — |
| deep-neural-sheaf-diffusion | github.com/remibourgerie/… (paper) | `3f706c2bff7e…` (laptop; default branch is now `034316e`) | none | none | — |
| HSB-GDL | github.com/mr49online/HSB-GDL (code) | `4126e64e7af9…` (laptop = latest) | none | none | **0 Python files** at this commit (data, readmes, a video): checked |
| fasth | github.com/alexandermath/fasth (Householder paper) | `b3fddf117f05…` (latest) | MIT | none (reference for `src/models/fasthpp.py`) | — |
| tudataset | github.com/chrsmrrs/tudataset (TUDataset.pdf) | `26e6ef6c5f63…` (latest) | none | none | — |
| dgl | github.com/dmlc/dgl (LMSPS paper) | `3d16000b4170…` (latest; shallow) | Apache-2.0 | none (DGL is used as a pip package) | — |
| HeaRT | github.com/Juanhui28/HeaRT | `c447cbf` (laptop) | — | removed protocol (v1.14) | not cloned (`kind=no-clone`) |
| **PSHGCN** | **not found** | — | — | pshgcn, _pshgcn_lp | not cloned; LP subtest skipped; 2 `test_baselines` errors |
| **ChebNetII** | **not found** | — | — | chebnetii | not cloned; NC subtest skipped |
| **Cooperative-Sheaf-NN** | **not found** (its PDF names only torch-householder) | — | — | none | not cloned |
| **hgbi** | **not found** | — | — | none | not cloned |

LatGRL (removed in v1.15) has no row: no clone and no commit. Licence = the clone's LICENSE file; "none" = no licence file found.

**Tests (VERIFIED-RUN).**
- Clone script on login01 (`cs3_clone_script.txt`): full run → every clone `PRESENT`, the rest `NOT CLONED (kind)`, exit 0; fresh clone of `fasth` into an empty scratch directory → `CLONED … b3fddf11…`, exit 0; second run → `PRESENT`, exit 0; a tracked file modified → `REFUSED fasth: dirty clone`, exit 1.
- Job **16612** (g2n01, RTX 2080 Ti; `cs3_vendor_tests.txt`): `available_vendors()` all true except `pshgcn`, `pshgcn_lp`, `chebnetii`. `test_vendor_manifest` Ran 4: 3 pass; **`test_every_row_has_a_commit` fails**, listing `['PSHGCN', 'ChebNetII', 'Cooperative-Sheaf-NN', 'hgbi']`. This failure is intended: it stays red until the author supplies the URLs. Skipped-for-clone tests rerun: `test_heterophily_metrics` 18 OK; `test_hgb_oracle` 7 OK; `test_partP_h2gformer` 10 OK; `test_partPfix_h2gformer_width` 3 OK; one-epoch fits as in the table.

**Deviations.**
- The extra column `kind` (`clone`, `in-repo`, `no-clone`, `unresolved`) tells the script what to clone.
- csnn and HINPool have no upstream commit anywhere in the sources; their row uses this repository's last commit for the directory.
- Laptop-pinned vendors kept their laptop commit even where newer exists (deep-NSD).
- `available_vendors()` is not all true (the card's test): PSHGCN and ChebNetII are unresolved.

## CS4 — Data (commit `dc526ba`)

**Done.** VERIFIED-RUN job **16613** (g2tn01; `cs4_data_check.txt`): `manifest: 215 files, 3270364442 bytes; server: 215 files, 3270364442 bytes` → **missing 0, extra 0, size-different 0**. (`du` shows 273 MB allocated against 3.1 GB apparent: the NFS server compresses. Spot check: PubMed `node.dat` ends in real feature text, and the tensors hold non-zero data to their last 4 MB.)

Job **16614** (g2n01, RTX 2080 Ti; `cs4_count_tests.txt`):
- `python ../docs/partM_evidence/m1_data_counts.py` → **0 failure(s)**, exit 0 (69 checks).
- The 12 M1 test modules → Ran 178; **1 failure**: `test_task_heads…test_outputs_and_gradients_are_bit_identical_to_the_pre_removal_capture`, `593 != 0`, first mismatches `init/node_encoder.encoders.0.0.weight max|diff|=2.980e-08`. These are 1-ulp differences already at initialisation against `docs/heads_evidence/v2_golden.pt`, which was captured on Windows. This is a platform difference, not a data difference, so CS4's stop rule (data difference) does not apply. Decision 3 below.

## CS5 — Full suite (commit `78b0881`)

VERIFIED-RUN `sbatch docs/cluster_evidence/cs5_job.sh`. It runs `bash scripts/slurm/run_tests.sbatch` inside one job (`--gres=gpu:1 --cpus-per-task=8 --mem=32G --time=4:00:00`), then the mb0 compare. Job **16615**, node **g2n01**, GPU **RTX 2080 Ti**, git `dc526ba`; `cs5_full_suite.txt`.

**`Ran 689 tests in 1053.566s — FAILED (failures=15, errors=12, skipped=2)`** (laptop Part MB-fix: 674 tests, failures=2, errors=2, skipped=7; +15 = the new CS1/CS2/CS3 tests).

| Cause | Tests | Count |
|---|---|---|
| Known laptop failures (still present) | `test_notebooks_are_committed_clean`; `test_all_declared_vendor_clones_are_present` (now missing only `pshgcn`, `pshgcn_lp`, `chebnetii`); `test_colliding_vendor_module_names_do_not_leak`, `test_vendor_model_init_is_reproducible_under_seed_everything` (both `FileNotFoundError` for PSHGCN) | 2 F + 2 E |
| `dgl` not installed | `DglCudaGateTests` (8 failing subtests/parent: e.g. `'CUDA' not found in 'missing requirement group dgl …'`); one-epoch fits gcn, gat, han, hinormer (NC Freebase), han (LP), han (data-root) | 9 F incl. 2 below + 6 E |
| `torch_householder` not installed | `DglCudaGateTests.test_only_dgl_backed_models_are_gated` (csnn, nsd: `missing requirement group sheaf`); one-epoch nsd, csnn (NC and LP) | (2 F in row above) + 4 E |
| 1-ulp platform difference in bit-exact tests | `test_task_heads` golden (593 tensors, ≤ 3e-8); `test_cooperative…test_T1_symmetric_reproduces_the_golden_outputs_bit_for_bit` (same golden file, same 593); `test_partMB_scopes…test_featureless_node_in_two_scopes_has_its_global_h0` | 3 F |
| New manifest test (4 unresolved URLs) | `test_vendor_manifest…test_every_row_has_a_commit` | 1 F |

**Skips (2):** one-epoch `pshgcn` (LP) and `chebnetii` (NC): `vendor clone … is empty on this machine`.

**Which laptop skips/failures disappear.** The laptop's 7 skips (empty vendor clones; no MSVC for CSNN) drop to 2. Tests that were skipped there now run. The H2GB/Hetero2Net heterophily tests (OK), the HGB oracle (OK) and the H2GB/Hetero2Net/H2SGNN fits (pass) now run green. The NSD and CSNN fits now reach import and stop on `torch_householder`. The laptop's 4 known failures all remain, each with a smaller cause (only PSHGCN/ChebNetII clones missing). The DGL-gate and DGL-fit failures are new on the server: the laptop has the Windows DGL 2.2.1 and they passed there.

**`test_partMB_scopes` diagnostic (VERIFIED-RUN, no code change; job 16616, g2tn01, i7-9700; `cs5_scope_diag.txt`).** 1 of 13 rows differs, `max|diff|=4.470e-08`, with 4 threads and with 1. The node encoder's output for a row depends on the number of rows in the CPU GEMM on this platform. The test asserts `torch.equal`.

**mb0 reference (information only).** `CUDA_VISIBLE_DEVICES="" python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference.pt` → `compared 139 tensors; differing: ['.eval_logits_1', '.eval_logits_2', '.h0', '.grads.theta_tau', '.grads.node_encoder.encoders.0.0.weight']` (the script prints the first 5 only and exits 0). The reference was captured on Windows; consistent with the 1-ulp differences above. Not a stop condition (card).

## Decisions for the author

| # | Decision | Options | Recommendation | Reason |
|---|---|---|---|---|
| 1 | DGL | (a) copy the `dgl-2.5.0+cu124` cp312 manylinux wheel from the laptop and `pip install --no-cache-dir` it in a job; (b) leave DGL out | (a) | 11 test failures/errors and GCN, GAT, HAN, HiNormer, SeHGNN depend on it; `data.dgl.ai` is unreachable from both node types |
| 2 | `torch_householder==1.0.1` | install in a job (gcc/g++ 14.2 present) | install | NSD and CSNN fits fail on import; it is in the lock |
| 3 | Bit-exact golden tests (`test_task_heads`, `test_cooperative` T1) and `mb0_reference.pt` | (a) re-capture `v2_golden.pt` and `mb0_reference.pt` on the cluster with a recorded node/CPU; (b) keep the Windows captures and accept known failures on Linux | (a), as a small card | Differences are ≤ 3e-8 at init (platform RNG/GEMM rounding), not code; the tests are only useful if they can be green on the machine that runs the campaign. Tolerances were not touched |
| 4 | `test_partMB_scopes…featureless…global_h0` asserts `torch.equal` across different row counts | decide whether this invariant is meant bit-exact on every platform | design-session decision | On this CPU one row differs by 4.5e-8; changing the assertion is outside this card |
| 5 | URLs for PSHGCN, ChebNetII, Cooperative-Sheaf-NN, hgbi | supply them (with the commit to pin) | supply; then `bash scripts/setup/clone_vendors.sh` | Not in any source the card names; not guessed |
| 6 | Upstream of `vendor/csnn`, `vendor/HINPool` | record upstream URL+commit, or keep "in-repo" | record if known | Manifest uses this repo's commit |
| 7 | deep-neural-sheaf-diffusion | keep laptop `3f706c2` or move to `034316e` | keep (same as laptop) | Not used by any adapter |
| 8 | HetSheaf licence CC BY-NC-ND 4.0 | note for the HetSheaf card | check before adapting its code | "NoDerivatives" may matter for a modified copy |
| 9 | Local build tags in `check_env.py` count as notes, not differences | keep / make strict | keep | A pin without a tag leaves the build to the index; a CPU `+cpu` torch still prints as a note, and CS0 checks CUDA directly |

## Out-of-scope findings (listed, not fixed)

- `mb0_reference.py compare` exits 0 when tensors differ and prints only the first 5 names.
- Vendor clone directories are untracked but not git-ignored (only `vendor/HGB/*` is); an accidental `git add -A` would add them as gitlinks.
- `torch_geometric` prints `'torch-cluster' is no longer necessary and is being ignored` (pyg-lib migration); harmless.
- CLAUDE.md/AGENTS.md "Hard rules" still say "Do not commit or push unless asked"; the new cluster section states that the 2026-11-14 push rules supersede it for agent work branches.
