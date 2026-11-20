# Ops session handoff — 2026-11-20

For the next **ops session** (design session that owns run infrastructure). Talk to the author in Thai, short and plain; cards and agent prompts in English. Commit design docs yourself, stage by name, never `git commit -a`, rebase, reset or stash. `v3b` moves only by fast-forward after the author writes "approve <sha>"; log each in `docs/experiment_settings_decisions.md` (append-only). Deadline 2027-01-04.

## Read first
1. `docs/working_preferences.md`, then this file.
2. `docs/tsubame_guide.md` §9 (T6 checklist; where each step runs) and §0 (every reported number comes from TSUBAME H100; lab runs never enter a table).
3. `docs/H2S-NN_tsubame_card.md` (T0–T7) and `docs/H2S-NN_tsubame_report.md` on `v3b` (T6 commands, three chains).
4. Decisions log rows of 2026-11-19/20 (T7 approved, base fc0ebaa, rsync -H).

## Who owns what
- **Ops (this session):** cards 1e/1f (S1), card 2 (S2, TSUBAME scripts), card 3 (S3), card 6 (S5), the TSUBAME card; reviews and v3b fast-forwards for those.
- **Model session** (same worktree, other Claude): spec, pre-registration, card 5 (S4), card 7 (S6). Do not change `run_row.py` or card 7 phases of `make_cell_list.py` without telling it.

## State
- `v3b` = `fc0ebaa605280f2d987c4848cc3109c7d3b66680` (T7 + card 7; `src/`, `configs/` = H2S-NN freeze 365a757).
- S1, S3, S5: nothing pending. S2: idle after T7.
- Heterophily (card 6): complete for NC (12), LP (8, task H_edge + MDE per endpoint type on F17 files), GC (type statistics only, spec 1.23). `docs/heterophily_evidence/table.md`.

## TSUBAME (agent has access)
- `ssh t4` from the lab machine works (key `~/.ssh/id_ed25519_t4`). WORK = the TSUBAME work dir (`${HOME/home/work}`; path in the agent memory, not in git). W&B key is in TSUBAME `~/.netrc` (never copy it anywhere).
- Done 2026-11-20: checkout `$WORK/h2snn-runs/fc0ebaa…/H2S-NN` (vendors real clones; H2GB shows "dirty" = our patch, expected; `data` → `$WORK/h2snn-data`), DGL package in `~/wheels`, Miniforge in `~/miniforge3`.
- Started 2026-11-20 ~14:10: rsync of all `data/` with `-aLH` from the lab; `install_env.sh` job **9480778** (cpu_8, 2 h), log `$WORK/h2snn-out/fc0ebaa…/logs/install_env.sh.o9480778` must end `exit=0`.
- Next (allowed): when both finish, compare `du -sh $WORK/h2snn-data` with lab `du -shL data`; run `env_check.sh` (trial run, guide §9 step 3); write the log to `docs/tsubame_evidence/t6_env_check.txt` with the TSUBAME login and work path replaced by `<t4-login>` / `$WORK` (the repo is public).
- **Not allowed until the author says go:** §9 step 4 (launcher dry-run), step 5 (Freebase probe), step 6 (chains), any tuning or training. The author first re-checks how training and tuning work per task.

## Open items
1. Author's question (next): how tuning works per task, and which work runs on the lab cluster vs TSUBAME. Rule to keep: reported numbers (tuning sweeps, final seeds) only on TSUBAME H100; lab = tests, mutations, smoke/timing checks.
2. Freebase NC mode (card 5 D1): decided after the H100 probe.
3. H2GB trials > 24 h: decide (epoch resume vs node reservation) after first-trial timings.
4. `status.py` does not read `run_row.py` lines (card 7): small S2 task when chains run.
5. Phase-2 baseline card (Freebase, H2GB NC, large LP, GC) when the author asks.
6. Advise W&B key rotation after setup (key was pasted in chat).

## Update 2026-11-20 ~15:30: T6 steps 0–3 done
- install_env job 9480778 exit=0; env = lock (29/29 pins, DGL sha256 equal).
- env_check job 9480898 (group points, 20 min; the 3-min trial run 9480862 hit h_rt): **fails=0** on H100 (driver 580, CUDA 12.4, DGL/PyG CUDA OK, cuda_float32 test OK; W&B/PyPI/GitHub reachable from compute nodes). Evidence `docs/tsubame_evidence/t6_env_check.txt` (7f07c09).
- Data: 715 files, 198 hard links, apparent 35G both sides (lab plain du 16G = NFS compression). Quota 44/125 GB.
- W&B key in `~/.netrc`, HTTP 200. Steps 4–6 still wait for the author's go.
