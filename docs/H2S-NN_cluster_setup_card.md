# Task card — Cluster setup: the repository, environment, vendors and test suite on the lab GPU cluster

**Binding:** `docs/working_preferences.md` (section "Server and GitHub workflow", 2026-11-14); `docs/experiment_settings_decisions.md` rows of 2026-11-14; the lab manual (author's copy; facts below); `H2S-NN/AGENTS.md`/`CLAUDE.md` for coding and evidence tags. Spec v1.22 is the design authority and is not touched: this card changes no method code.
**Branch:** `cluster-setup`, created from `part-mb` at the commit given in the prompt. **Report:** `docs/H2S-NN_cluster_setup_report.md`; evidence `docs/cluster_evidence/`.

## Facts of the cluster (author, 2026-11-14; lab manual)
- Login node `login01` (Debian 12), reached as `ssh peeradon@ai1.net.comp.isct.ac.jp`. Claude Code runs here. **No computation on the login node** (tests, training, `pip`/`conda install`, preprocessing): use `srun ... --pty bash` or `sbatch`. Compute nodes are not reachable by ssh.
- Every job sets `--mem` and `--time` (≤ 72 h; longer work needs checkpoints). GPUs: `--gres=gpu:<type>:<n>` with types `2080ti` (11 GB), `titan_rtx` (24 GB), `3090` (24 GB), `a6000` (48 GB); `--cpus-per-gpu`. Driver 550.163.01, CUDA 12.4 (checked by the author on a 2080 Ti node).
- Conda: `. /opt/conda/etc/profile.d/conda.sh`; environment `h2snn` (Python 3.12) created by the author in his home from `H2S-NN/requirements.txt`.
- Storage: repo `/mnt/data/peeradon/h2snn` (project storage, NFS, readable by every lab member, no backup, no quota); home `/mnt/home/peeradon` (50 GB, private, no backup; manual: jobs should not run in `$HOME`); `/tmp` on a compute node is fast scratch deleted at job end.
- Secrets (W&B key, GitHub key) only under home, never under `/mnt/data`.

## Rules
- Correctness only; no performance run, no metric reported as a result, no comparison between methods.
- **Push rules (2026-11-14):** push `cluster-setup` after each unit (`git push origin cluster-setup`); never push or move `v3b`; never force-push; never commit vendor files (HGB has no licence), `data/`, secrets, `Claude outputs/`, `H2S-NN/paper/`.
- One commit per unit, explicit paths, tests with the change. Forbidden: loosening tolerances, deleting or skipping tests, silencing exceptions, editing the spec or the cards, changing method code.
- **Stop:** a requirement needs root or an administrator (report the exact error and stop that unit); the installed environment cannot import torch with CUDA (report, stop the card); a vendor URL or commit cannot be found in the sources named below (report it; do not guess a repository).
- **Not stop conditions:** a vendor's own one-epoch test failing (report it, do not fix it here); version differences between the server and the lock (report them; the author decides).

## Units

### CS0 — Preconditions (login node; checks inside one short `srun` job)
`git fetch`; check the base hash; create `cluster-setup`. In one job (`srun --gres=gpu:1 --mem=8G --time=0:20:00 --pty bash`, or an `sbatch` script): `nvidia-smi`; `gcc --version` and `g++ --version` (needed by `torch_householder` for CSNN/NSD); `conda activate h2snn`; `python -c "import torch, torch_geometric, torch_scatter, torch_sparse, dgl; print(...versions..., torch.cuda.is_available(), dgl CUDA support)"`; `pip freeze > docs/cluster_evidence/server_pip_freeze.txt`. Record the node, GPU, versions and whether DGL is the `+cu124` build (`src/baselines/runtime/dependencies.py` `_dgl_cuda_defect`).

### CS1 — Cluster section in `CLAUDE.md` and `AGENTS.md`
Keep the Windows section unchanged; add "Lab GPU cluster (Linux, SLURM)": the facts above (paths, conda line, login-node rule, `--mem`/`--time`, GPU types, `/tmp` scratch, secrets in home), the push rules, and how to run tests: new `scripts/slurm/run_tests.sbatch` (arguments: unittest targets, default the whole suite; `--mem`, `--time`, `--cpus-per-task` set; conda activated; runs from `H2S-NN/`; log to `logs/slurm-%j.out` under the repo, `logs/` git-ignored) and `scripts/slurm/interactive.md` (the `srun --pty bash` line for short work).
**Tests:** `bash -n` on the script; `sbatch --test-only scripts/slurm/run_tests.sbatch` succeeds; a unit test that reads the script and asserts `--mem`, `--time`, the conda line and no `cd $HOME`. **Mutation:** `--time` removed → the test fails.

### CS2 — `requirements/linux-lock.txt` and an environment check
Exact pins of the author's laptop environment `h2snn` (read 2026-11-14 from its `site-packages`; Python 3.12.13), except DGL: `torch==2.6.0` (cu124 index), `torch_geometric==2.8.0`, `torch_scatter==2.1.2`, `torch_sparse==0.6.18`, `torch_cluster==1.6.3` (PyG wheel index for torch 2.6.0+cu124), `numpy==2.4.6`, `scipy==1.18.0`, `scikit-learn==1.9.0`, `pyyaml==6.0.3`, `torchmetrics==1.8.2`, `wandb==0.28.0`, `pandas==3.0.3`, `tqdm==4.68.3`, `umap-learn==0.5.12`, `matplotlib==3.11.0`, `ogb==1.3.6`, `yacs==0.1.8`, `dill==0.4.1`, `pytorch_lightning==2.6.5`, `prettytable==3.18.0`, `gdown==6.1.0`, `performer-pytorch==1.1.4`, `seaborn==0.13.2`, `texttable==1.7.0`, `termcolor==3.3.0`, `torch_householder==1.0.1`, `dgl==2.5.0+cu124` (Linux; the laptop has the Windows CPU build 2.2.1). Not pinned: `numba` (only `scripts/lp/build_heart_negatives.py`, HeaRT removed in v1.14), HGB's `tensorflow`/`gensim`/`rdflib`/`pynvml` (methods not used). New `scripts/setup/check_env.py`: compares installed versions (importlib.metadata) with the lock and prints every difference; exit code 1 on a difference.
**Tests:** the comparator on a fake installed mapping (equal, missing, different, extra local build tag such as `+cu124`). Run `check_env.py` in a job; record its output. **Do not** reinstall to remove a difference: report it.

### CS3 — Vendor manifest and clones
New `H2S-NN/vendor/VENDORS.tsv` (committed): one row per directory in `src/baselines/runtime/vendor_paths.py` `VENDOR_ROOTS` and per vendor present on the laptop: name, repository URL, commit, licence note, used by (adapter). Laptop state (design session, 2026-11-14): own `.git` — `H2GB` `junhongmit/H2GB` `d3b49db`; `Hetero2Net` `EdisonLeeeee/Hetero2Net` `4913394`; `Heterogeneous-Sheaf-Neural-Networks` `alessioborgi/...` `6964e1e`; `deep-neural-sheaf-diffusion` `remibourgerie/...` `3f706c2`; `HSB-GDL` `mr49online/HSB-GDL` `4126e64` (0 Python files: check); `HeaRT` `Juanhui28/HeaRT` `c447cbf` (removed protocol; no clone needed). Files tracked in our repo: `HINormer` (22, `CLONED_COMMIT.txt`), `csnn` (21), `HINPool` (14). `HGB` cloned at `ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe` (`vendor/HGB/CLONED_COMMIT.txt`; never commit its files). **Empty on the laptop since 2026-11-05:** ChebNetII, Cooperative-Sheaf-NN, H2SGNN, PSHGCN, SeHGNN, neural-sheaf-diffusion, dgl, fasth, hgbi, tudataset, LatGRL (removed in v1.15: no clone), HSB-GDL. Find each URL in the adapter docstrings, `docs/`, `references/` or the method's paper; a URL that cannot be found there is reported, not guessed. Clone the latest default-branch commit and record it. New `scripts/setup/clone_vendors.sh`: clones every row at its commit (idempotent; skips present clones at the right commit; refuses a dirty clone).
**Tests:** the manifest parses; every `VENDOR_ROOTS` name has a row; every row has a 7–40 hex commit; `available_vendors()` is all true after the script (run in a job). Then run every test that was skipped for a missing clone (`tests/test_baselines.py` and the adapters' one-epoch tests) in one GPU job; report pass/fail per vendor. A vendor test failing is reported, not fixed.

### CS4 — Data on the server
The author copies `H2S-NN/data/` from the laptop (3.27 GB, 215 files; the H2GB data are not in it). Compare the server's `data/` with `docs/cluster_evidence/laptop_data_manifest.tsv` (path and size; written by the design session 2026-11-14): report missing, extra and size-different files. Then run the existing data-count tests (Part M M1 gate tests and the loaders' count tests) in a job. A difference stops this unit.

### CS5 — Full suite on the cluster
`sbatch scripts/slurm/run_tests.sbatch` on one GPU (`--gres=gpu:1`, `--cpus-per-task=8`, `--mem=32G`, `--time=4:00:00`). Laptop reference (Part MB-fix): 674 tests, the four known `test_baselines` failures (vendor clones, notebook check), 7 environment skips. Report the server result: every failure and skip with its reason, and which laptop failures/skips disappear. Run `docs/partMB_evidence/mb0_reference.py compare` on CPU in the same job and report the result as **information**: the reference was captured on Windows; a cross-platform difference is not a stop condition; whether to re-capture it on the cluster is a decision for the author.

### CS6 — Report
Per unit: done, VERIFIED-RUN commands (with job ids, node, GPU) and outputs, deviations; the version differences of CS2; the vendor table of CS3 (URL, commit, test result); "Decisions for the author"; out-of-scope findings.

## Out of scope (later cards)
HetSheaf (its `pyproject.toml` needs Python ≥ 3.13 and torch 2.8: separate environment, decision D-P4), the tuning launcher and ablation-row driver, H2GB data preprocessing and loaders (Part M3b), Part M3a (runs after this card), SeHGNN/PSHGCN/H2SGNN Freebase (D-P8).
