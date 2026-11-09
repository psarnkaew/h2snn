"""J5.4 diagnosis: per relation, the training facts the split holds vs the message-graph edges the
loader emits (data.edge_type), with the target-only file. Explains why the message graph grows by
less than the non-target validation rows. Read only. Run from H2S-NN/."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(".").resolve()))
from src.configs.config import TrainConfig  # noqa: E402
from src.data.link_prediction_dataset import (  # noqa: E402
    _load_local_hgb_link_data, build_link_prediction_dataset)

for dataset in ("pubmed", "lastfm"):
    config = TrainConfig.from_dict({"device": "cpu", "data": {"dataset_name": dataset,
                                                             "root": "data"}})
    data = build_link_prediction_dataset(config)
    split = _load_local_hgb_link_data(config)
    counts = torch.bincount(data.edge_type).tolist()
    names = getattr(data, "relation_names", None) or getattr(data, "edge_type_names", None)
    print(f"{dataset}: fact_layer {config.data.fact_layer}; message edges {data.edge_index.size(1)}; "
          f"edge_type counts {counts}; relation names {names}")
    for rel, e in sorted(split.train_edge_index_by_relation.items(), key=lambda r: r[0][1]):
        u = e if e.numel() else e
        loops = int((u[0] == u[1]).sum()) if u.numel() else 0
        uniq = len({(a, b) for a, b in u.t().tolist()})
        unordered = len({(min(a, b), max(a, b)) for a, b in u.t().tolist()})
        print(f"  {rel}: training rows {u.size(1)}, unique {uniq}, unordered {unordered}, loops {loops}")
