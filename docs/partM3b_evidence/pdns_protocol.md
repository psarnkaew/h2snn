# B5 — PDNS: the protocol H2GB's code implements (written before any loader code; unit stopped)

**Stop (card B5 / working preferences "a card that is wrong or ambiguous makes the agent stop"):** the card and spec
§6 describe PDNS as temporal and inductive ("training graph over the training windows; one test graph per later window
with previously labelled domains removed"). H2GB's code trains and evaluates PDNS **transductively on a random 70/20/10
split of one graph** (windows 0–59); the per-window test graph it builds is never read, and with H2GB's arguments it
covers windows 0–5, i.e. it is not later than training. Following the card would deviate from H2GB's code; following
H2GB's code would contradict the card. No loader is written; the PDNS NC cell stays data-pending. Options are listed
under "Decision for the author".

## What H2GB's code does (VERIFIED-CODE, vendor `d3b49db`)
1. `load_dataset_master` → `preformat_PDNS(dataset_dir)` (`H2GB/loader/master_loader.py:604-617`):
   `PDNSDataset(root=<dir>/PDNS, start=0, end=60, domain_file='domains2.csv', transform=T.ToUndirected())`;
   `test_list` keeps its default `[6]` (`H2GB/datasets/pdns_dataset.py:47-50`).
2. `PDNSDataset.__init__` (`:47-55`) calls `_load_train_test` and stores **only its return value**
   (`self.data, self.slices = self.collate([processed])`, `:55`); `_load_train_test` returns `self.train_data`
   (`:247-249`; lines `:188-226` are a string literal and never run).
3. `self.train_data = _load_data(..., start=0, end=60, num_val=0.2)` (`:173`): the graph of the edge files of
   windows 0–59 (`edge_files[start:end]`, files sorted by window number, `:151-153, :278-280`); labels from
   `domains2.csv` (`type` malicious = 1, benign = 0, else −1, `:156-159`) kept only for domains seen in those windows
   (`_load_graph`, `:251-270`); split by an **unseeded** `torch.randperm` over the labelled domains: test 10 %, validation
   20 %, train 70 % (`_load_data`, `:302-312`). H2GB calls `seed_everything(cfg.seed)` just before `create_loader`
   (`H2GB/main.py:87-100`), so the split is a function of the run seed (torch's global generator).
4. `self.test_data` (`:175-187`): for each `test` in `test_list` = `[6]`, a graph over `edge_files[0:6]` (windows 0–5)
   whose labels are removed for every domain seen in `edge_files[0:5]` (windows 0–4), split with `num_val=1.0`.
   **Nothing in H2GB reads `test_data`** (grep of `H2GB/` for `test_data` and `dynamicTemporal`: only
   `pdns_dataset.py`; `dynamicTemporal` is never set on this class, so `master_loader.py:269` and
   `graphgym/loader.py:270` do not apply). Samplers read `dataset[0]` = the training graph (`custom_sampler.py:92, 330`).
5. No processed file: the class has no `process` and no `processed_file_names`, so every load re-reads `raw/` (52 s here).
   `scripts/data/h2gb_use_dataset.py` (B1) would load the dataset the same way and write nothing to `NC/PDNS/processed`;
   it was not run (deviation from the card's "run H2GB's own PDNS preprocessing": there is none). The data check below
   runs H2GB's `preformat_PDNS` itself.

## Data check (job 16773, g3an01, `b5_pdns_job.sh` → `b5_pdns_check.json`, log `b5_pdns_16773.log`; 52.3 s load, peak RSS 5.6 GiB; `NC/PDNS` before and after: `raw` only)
| | H2GB's loaded graph (`dataset[0]`, seed 0) | H2GB Table 1 (arXiv 2407.10916v2) |
|---|---|---|
| nodes (types) | 1,173,558 (2): domain 897,635, IP 275,923 | 1,173,558 (2) |
| edges (types) | 76,797,104 (4): resolves 37,390,522 ×2 (with `rev_resolves`), similar 1,228,824, apex 787,236 | 76,797,104 (4) |
| features | domain 10, IP 2 | 10 |
| labelled domains | 87,252 (malicious 58,273, benign 28,979: 2.0 : 1) | 2 (1:2) |
| split | train 61,077 / val 17,450 / test 8,725 (70 / 20 / 10, random) | "Time (70/20/10)" |
| class counts (benign, malicious) | train 20,304 / 40,773; val 5,815 / 11,635; test 2,860 / 5,865 | — |

The stored (pre-`ToUndirected`) graph has 41,501,735 edges in 3 relations (apex 1,609,183 and similar 2,502,030 as in the
CSVs; `ToUndirected` coalesces them). H2GB's unused test graph (windows 0–5): 3,997 labelled domains (val 3,598 / test 399,
benign 2,381 + 276, malicious 1,217 + 123).

### Time windows (`raw/timestamp_<t>_edges.csv`; full table in `b5_pdns_check.json` → `windows`)
| window | resolves | similar | apex | domains | IPs | malicious | benign | unlabelled | new domains | new labelled |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 235,304 | 34,903 | 23,923 | 69,407 | 41,840 | 8,586 | 3,301 | 57,520 | 69,407 | 11,887 |
| 6 | 756,431 | 33,599 | 21,757 | 74,148 | 92,472 | 8,502 | 6,261 | 59,385 | 15,887 | 2,610 |
| 59 | 254,796 | 33,807 | 22,345 | 73,854 | 47,528 | 8,908 | 1,741 | 63,205 | 7,786 | 271 |
| 60 (not used by H2GB) | 700,979 | 40,184 | 25,574 | 85,801 | 95,329 | 9,066 | 3,223 | 73,512 | 7,485 | 547 |

Labelled domains first seen in windows 0–59: 87,252 (= H2GB's labelled count); window 60 adds 547.

## Decision for the author (B5)
| Option | What it is | Cost / consequence |
|---|---|---|
| **(a) H2GB's code exactly** (recommended) | PDNS is transductive: one graph over windows 0–59, H2GB's random 70/20/10 split of its labelled domains, drawn from the run seed as H2GB does (`seed_everything` before loading). The loader is the B2 wrapper (no processed file: `raw/` every load, ≈ 1 min). Spec §6 and the decisions-log row ("temporal inductive") are corrected to the code. | Follows "follow the source's code exactly" and is the protocol behind H2GB's published PDNS numbers; PDNS is then not an inductive dataset (one claim of the spec goes). |
| (b) the spec's temporal protocol | Training graph = windows 0–59 (H2GB's `start`, `end`), one test graph = windows 0–60 with the 87,252 domains labelled in 0–59 removed (547 test targets); validation from the training graph. Stated deviation from H2GB's code (file:line above). | The trainer needs nothing new: `MiniBatchNodeTrainer.fit(graph, eval_graph)` already evaluates on a second graph, and scopes are cut from the graph given (N6). 547 test targets (benign/malicious mix not yet counted) give a noisy test metric. |
| (c) H2GB's `test_data` as built | The graph H2GB builds with `test_list=[6]` (windows 0–5, 399 test targets). | Earlier than the training windows (not a temporal test); never used by H2GB. Not recommended. |

Recommendation: (a), because the card's rule binds us to H2GB's code and the published numbers come from it; (b) only if
the inductive claim is wanted for its own sake, then as a stated deviation.
