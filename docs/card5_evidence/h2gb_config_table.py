"""Card 5 U1: the eight H2GB NC experiment configs as resolved by TrainConfig.from_dict (epochs, patience, eval_period).
Run from H2S-NN/: python ../docs/card5_evidence/h2gb_config_table.py"""
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path.cwd()))
from src.configs.config import TrainConfig, protocol_budget  # noqa: E402

print("| dataset | epochs | early_stopping_patience | eval_period | training_mode | batch_size | epoch_targets | in YAML |")
print("|---|---|---|---|---|---|---|---|")
for name in ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns"):
    path = Path(f"configs/experiments/{name}_h2snn_v3b.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = TrainConfig.from_dict(raw)
    explicit = [k for k in ("epochs", "early_stopping_patience") if k in raw] + \
               [k for k in ("eval_period", "epoch_targets") if k in (raw.get("data") or {})]
    print(f"| {name} | {cfg.epochs} | {cfg.early_stopping_patience} | {cfg.data.eval_period} | "
          f"{cfg.data.training_mode} | {cfg.data.batch_size} | {cfg.data.epoch_targets} | {explicit or 'none (protocol budget)'} |")
budget = protocol_budget("node_classification", "ogbn-mag")
print(f"\nprotocol budget H2GB NC: epochs {budget.epochs}, patience {budget.patience}, monitor {budget.monitor}")
