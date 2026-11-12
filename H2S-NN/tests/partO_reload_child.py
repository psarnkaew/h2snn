"""Fresh-process half of the Part O reload test (card O1.2; `tests/test_partO_reload.py`).

    python -m tests.partO_reload_child <run_dir> <graph.pt> <out.pt>

Reloads the run from `<run_dir>/config.yaml` and `<run_dir>/checkpoints/best.pt` through
`src.analysis.model_loading.load_trained_model` (the evaluation draw is rebuilt from the saved base
seed by `build_dataset`), with `build_dataset` reading the synthetic graph `<graph.pt>` instead of
the ACM files, and writes the rebuilt cells and `w_f` to `<out.pt>`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def main(argv: list[str]) -> None:
    run_dir, graph_path, out_path = (Path(value) for value in argv)
    from src.analysis.model_loading import (
        evaluation_cell_weights,
        load_trained_model,
        nonempty_groups,
    )
    from tests.partO_fixtures import load_hetero, patched_acm_loader

    with patched_acm_loader(load_hetero(graph_path)):
        model, data, cfg = load_trained_model(run_dir, device="cpu")
    weights = evaluation_cell_weights(model, data)
    torch.save({"cells": cell_tensors(nonempty_groups(data.face_groups)),
                "weights": list(weights), "seed": cfg.seed}, out_path)


def cell_tensors(groups) -> list[dict[str, torch.Tensor]]:
    """The tensors that define each group's cells: node, fact and sign tensors, relation and
    signature ids, and m_f."""
    return [{"nodes": g.face_node_index, "facts": g.face_edge_index, "signs": g.face_edge_sign,
             "relations": g.relation_index, "m_f": g.path_weight,
             "signature_index": g.signature_index} for g in groups]


if __name__ == "__main__":
    main(sys.argv[1:])
