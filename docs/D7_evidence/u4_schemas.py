"""U4 evidence: default schemas resolved under tied and untied (relation names + direction)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.configs.config import TrainConfig
from src.data.dataset import _load_hetero_data
from src.data.fact_layer import build_fact_layer
from src.data.hetero_flatten import flatten_edges, stack_node_features
from src.data.schemas import resolve_schemas

for name in ("acm", "dblp", "imdb"):
    config = TrainConfig.from_dict({"data": {"dataset_name": name}})
    data = _load_hetero_data(config)
    x, _, _, slices, _ = stack_node_features(data)
    ei, et, names, tuples = flatten_edges(data, slices)
    for mode in ("tied", "untied"):
        layer = build_fact_layer(ei, et, names, x.size(0), mode)
        schemas = resolve_schemas(config.data.metapath_templates, layer, tuple(tuples[r] for r in layer.kept))
        rendered = "; ".join(f"{s.name}: " + ", ".join(f"{layer.kept_names[r]}{'+' if d > 0 else '-' if d < 0 else '0'}" for r, d in s.hops) for s in schemas)
        print(f"{name} {mode:6s} kept={len(layer.kept)}  {rendered}")
