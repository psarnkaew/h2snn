"""Card 7 amendments 4 and 7 (author 2026-11-19): the configuration of each perturbation baseline on ACM and DBLP.

Fixed per method (never retuned on a perturbed graph):
- GCN: HGB's published run command (`vendor/HGB/NC/benchmark/methods/GNN/README.md`), flags not given there at
  `run.py`'s argparse defaults; mapped onto our adapter's keys below (`HGB_COMMANDS`, `MAPPING`).
- HAN: the defaults of `vendor/HGB/NC/benchmark/methods/HAN/main.py` that its README runs (`python main.py --dataset
  ACM|DBLP`): `main.py`'s argparse (epochs, patience) and `utils.py`'s `default_configure` (lr, heads, hidden size,
  dropout, weight decay), which `setup()` applies.
- SeHGNN, Hetero2Net: their published configurations (`configs/baselines/<model>/node_classification/<dataset>.yaml`,
  read by the adapter's `published_params`, run by `scripts/run_baseline_published.py`).
- HetSheaf: its main-table ACM/DBLP tuning (seed-0 sweep, `run_baseline_tuning.py --final_seeds`); until that sweep
  exists its rows are marked "needs tuned config".
- CSNN (amendment 7): no published ACM/DBLP configuration exists (`CSNN_PUBLISHED`), so it is treated like HetSheaf.

HGB settings with no counterpart key in our adapters are listed in `UNMAPPED` (no key is added in `src/`).
"""

from __future__ import annotations

from typing import Any

HGB_GNN_README = "vendor/HGB/NC/benchmark/methods/GNN/README.md"
HGB_GNN_RUN = "vendor/HGB/NC/benchmark/methods/GNN/run.py"
HGB_HAN_README = "vendor/HGB/NC/benchmark/methods/HAN/README.md"
HGB_HAN_MAIN = "vendor/HGB/NC/benchmark/methods/HAN/main.py"
HGB_HAN_UTILS = "vendor/HGB/NC/benchmark/methods/HAN/utils.py"

#: configuration source per perturbation baseline
CONFIG_SOURCE: dict[str, str] = {"gcn": "hgb_command", "han": "hgb_command", "sehgnn": "published",
                                 "hetero2net": "published", "hetsheaf": "tuned", "csnn": "tuned"}
NEEDS_TUNED = "needs tuned config"

_GCN_DEFAULTS = {"hidden_dim": 64, "num_layers": 2, "dropout": 0.5, "epochs": 300, "early_stopping": 30}
_HAN = {"lr": 0.005, "weight_decay": 0.001, "hidden_dim": 8, "num_heads": 8, "dropout": 0.6, "epochs": 200,
        "early_stopping": 100}

#: (model, dataset) -> the HGB command, where it is published, and the adapter parameters it maps to
HGB_COMMANDS: dict[tuple[str, str], dict[str, Any]] = {
    ("gcn", "acm"): {
        "command": "python run.py --dataset ACM --model-type gcn --weight-decay 1e-6 --lr 1e-3 --feats-type=0",
        "source": f"{HGB_GNN_README} (line 10); other flags at {HGB_GNN_RUN} argparse defaults (lines 148-167)",
        "params": {"lr": 1e-3, "weight_decay": 1e-6, "feats_type": 0, **_GCN_DEFAULTS}},
    ("gcn", "dblp"): {
        "command": "python run.py --dataset DBLP --model-type gcn --weight-decay 1e-6 --lr 1e-3",
        "source": f"{HGB_GNN_README} (line 7); other flags at {HGB_GNN_RUN} argparse defaults (lines 148-167; "
                  "--feats-type 3)",
        "params": {"lr": 1e-3, "weight_decay": 1e-6, "feats_type": 3, **_GCN_DEFAULTS}},
    ("han", "acm"): {
        "command": "python main.py --dataset ACM",
        "source": f"{HGB_HAN_README}; {HGB_HAN_MAIN} argparse (--num_epochs 200, --patience 100); "
                  f"{HGB_HAN_UTILS} default_configure (lines 96-102)",
        "params": dict(_HAN)},
    ("han", "dblp"): {
        "command": "python main.py --dataset DBLP",
        "source": f"{HGB_HAN_README}; {HGB_HAN_MAIN} argparse (--num_epochs 200, --patience 100); "
                  f"{HGB_HAN_UTILS} default_configure (lines 96-102)",
        "params": dict(_HAN)},
}

#: the mapping, one row per HGB setting: (model, HGB setting, HGB value, adapter key, adapter value)
MAPPING: tuple[tuple[str, str, str, str, str], ...] = (
    ("gcn", "--model-type", "gcn", "(registry model)", "gcn"),
    ("gcn", "--lr", "1e-3", "lr", "1e-3"),
    ("gcn", "--weight-decay", "1e-6", "weight_decay", "1e-6"),
    ("gcn", "--feats-type", "ACM 0 (command); DBLP 3 (default)", "feats_type", "ACM 0; DBLP 3"),
    ("gcn", "--hidden-dim", "64 (default)", "hidden_dim", "64"),
    ("gcn", "--num-layers", "2 (default)", "num_layers", "2"),
    ("gcn", "--dropout", "0.5 (default)", "dropout", "0.5"),
    ("gcn", "--epoch", "300 (default)", "epochs", "300"),
    ("gcn", "--patience", "30 (default)", "early_stopping", "30"),
    ("han", "lr", "0.005 (default_configure)", "lr", "0.005"),
    ("han", "weight_decay", "0.001 (default_configure)", "weight_decay", "0.001"),
    ("han", "hidden_units", "8 (default_configure)", "hidden_dim", "8"),
    ("han", "num_heads", "[8] (default_configure; one HAN layer)", "num_heads", "8 (the adapter builds [num_heads])"),
    ("han", "dropout", "0.6 (default_configure)", "dropout", "0.6"),
    ("han", "--num_epochs", "200 (main.py default)", "epochs", "200"),
    ("han", "--patience", "100 (main.py default)", "early_stopping", "100"),
    ("han", "feat_type", "0 (main.py:33)", "(none: the adapter's vendor loader)", "0, same vendor loader"),
)

#: HGB settings without a counterpart key in our adapters (reported, not added)
UNMAPPED: tuple[tuple[str, str, str], ...] = (
    ("gcn, han", "optimizer", "HGB: torch.optim.Adam (L2 weight decay); adapters: AdamW (decoupled), fixed in code"),
    ("gcn", "early-stopping criterion", "HGB: validation loss; adapter: validation macro-F1 (protocol, adapter "
                                        "docstring deviation 2)"),
    ("han", "early-stopping criterion", "HGB/DGL HAN: validation loss and accuracy; adapter: validation macro-F1 "
                                        "(protocol)"),
    ("gcn", "--repeat", "HGB: repeats in one process; ours: seeds 1-5 as separate units"),
    ("han", "--num_layers", "main.py parses it (default 2) but HAN is built from num_heads=[8]: one layer; not used "
                            "by HGB either"),
    ("gcn, han", "split", "HGB: unseeded 20 % validation carve-out; ours: the project's deterministic carve-out "
                          "(adapter deviation 1)"),
)

#: where a published CSNN configuration for ACM/DBLP was looked for (none found)
CSNN_PUBLISHED: tuple[str, ...] = (
    "vendor/csnn/exp/scripts/: run_amazon, run_enzymes, run_minesweeper, run_questions, run_roman, run_squirrel, "
    "run_tolokers (.sh); no ACM, DBLP or IMDB script",
    "vendor/Cooperative-Sheaf-NN/README.md: points to exp/scripts only",
    "references/cooperative_sheaf_neural_network.pdf (arXiv preprint, 3 Feb 2026): experiments on roman-empire, "
    "amazon-ratings, minesweeper, tolokers, questions, the small heterophilic sets and Penn94; 'ACM' occurs only as "
    "the ACM-GCN baseline and in citations",
    "configs/baselines/csnn/: graph_classification only (ENZYMES)",
    "src/baselines/adapters/csnn.py docstring: node-classification cells are tuned",
)


def source_for(model: str) -> str:
    return CONFIG_SOURCE[model]


def hgb_command_params(model: str, dataset: str) -> dict[str, Any]:
    if (model, dataset) not in HGB_COMMANDS:
        raise KeyError(f"no HGB command mapped for {model}/{dataset}")
    return dict(HGB_COMMANDS[(model, dataset)]["params"])


def markdown_mapping() -> str:
    lines = ["| model | HGB setting | HGB value | adapter key | adapter value |", "|---|---|---|---|---|"]
    lines += [f"| {m} | `{s}` | {v} | `{k}` | {a} |" for m, s, v, k, a in MAPPING]
    lines += ["", "| model | HGB setting without a counterpart key | note |", "|---|---|---|"]
    lines += [f"| {m} | {s} | {n} |" for m, s, n in UNMAPPED]
    return "\n".join(lines) + "\n"
