from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import TrainConfig
from src.utils.misc import save_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--decoder_type", type=str, default=None, choices=["dot", "distmult", "concat_mlp"])
    return parser.parse_args()


def load_config(path: Path) -> TrainConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return TrainConfig.from_dict(payload)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    if args.device is not None:
        cfg.device = args.device
    elif cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
    if args.decoder_type is not None:
        cfg.data.decoder_type = args.decoder_type

    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.exp_dir / "config.yaml")
    print(
        "Resolved run: "
        f"device={cfg.device} orthogonal_map={cfg.model.orthogonal_map} "
        f"d={cfg.model.d} exp_dir={cfg.exp_dir}",
        flush=True,
    )
    seed_everything(cfg.seed, cfg.device)

    from src.builders import build_experiment_components

    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    try:
        logger.info("Dataset: %s | Task: %s", cfg.data.dataset_name, cfg.task_type)
        if cfg.task_type == "node_classification":
            logger.info("Target node type: %s", cfg.data.target_node_type)
            logger.info("Metapaths: %s", [item.name for item in cfg.data.metapath_templates])
        elif cfg.task_type == "link_prediction":
            logger.info("Target relation: %s | Decoder: %s", cfg.data.target_relation, cfg.data.decoder_type)
        elif cfg.task_type == "graph_classification":
            logger.info(
                "Split protocol: %s | Run: %d/%d | Fold: %d/%d",
                cfg.data.split_protocol,
                cfg.data.run_index,
                cfg.data.num_runs,
                cfg.data.fold_index,
                cfg.data.num_folds,
            )
        logger.info("Model parameters: %d", sum(parameter.numel() for parameter in model.parameters()))
        trainer.fit(data)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
