import sys
import yaml
from pathlib import Path

ROOT = Path(r"G:/My Drive/H2S-NN_test")
sys.path.insert(0, str(ROOT))

from src.configs.config import TrainConfig
from src.data.dataset import _load_hetero_data, _prepare_target_labels
from src.data.hetero_flatten import flatten_edges, stack_node_features
from src.data.preprocessing import label_propagation
from src.data.schema_discovery import (
    estimate_schema_stats,
    generate_schema_candidates,
    select_schema_pool,
    topology_aware_score,
)


payload = yaml.safe_load(
    (ROOT / "configs/experiments/acm_h2snn_ampss.yaml").read_text(encoding="utf-8")
)
payload["device"] = "cpu"
cfg = TrainConfig.from_dict(payload)

data = _load_hetero_data(cfg)
raw_x, node_type, node_type_names, node_slices, feature_dims = stack_node_features(data)
edge_index, edge_type, relation_names, relation_tuples = flatten_edges(data, node_slices)

(
    y,
    train_mask,
    val_mask,
    test_mask,
    target_indices,
    target_y,
    target_train_mask,
    num_classes,
    is_multilabel,
) = _prepare_target_labels(
    data=data,
    node_slices=node_slices,
    target_node_type=cfg.data.target_node_type,
    val_ratio=cfg.data.acm_val_ratio,
    split_seed=cfg.data.acm_split_seed,
)
train_indices = target_indices[target_train_mask]
train_labels = target_y[target_train_mask]
propagated_labels = label_propagation(
    edge_index=edge_index,
    train_indices=train_indices,
    train_labels=train_labels,
    num_nodes=raw_x.size(0),
    num_classes=num_classes,
    num_steps=cfg.data.num_label_prop_steps,
)

candidates = generate_schema_candidates(
    relation_tuples=relation_tuples,
    target_node_type=cfg.data.target_node_type,
    min_length=cfg.discovery.min_schema_length,
    max_length=cfg.discovery.max_schema_length,
    require_target_start=cfg.discovery.require_target_start,
    require_target_end=cfg.discovery.require_target_end,
    allow_repeated_node_types=cfg.discovery.allow_repeated_node_types,
    max_candidates=cfg.discovery.max_schema_candidates,
)
stats = estimate_schema_stats(
    candidates=candidates,
    node_slices=node_slices,
    edge_index=edge_index,
    edge_type=edge_type,
    propagated_labels=propagated_labels,
    target_node_type=cfg.data.target_node_type,
    max_instances_per_anchor=cfg.discovery.max_instances_per_anchor,
    split_seed=cfg.data.acm_split_seed,
)
selected, selected_stats, final_scores = select_schema_pool(
    candidates, stats, cfg.discovery.selected_schema_k
)

print("target", cfg.data.target_node_type)
print("relations")
for i, rel in enumerate(relation_tuples):
    print(i, rel)
print("num_candidates", len(candidates))
print("selected")
for c, s, score in zip(selected, selected_stats, final_scores.tolist()):
    print(
        c.schema_id,
        c.name,
        c.node_types,
        c.relation_ids,
        c.relation_names,
        "instances",
        s.num_instances,
        "D",
        round(s.endpoint_label_disc, 6),
        "C",
        round(s.end_coverage, 6),
        "V",
        round(s.type_diversity, 6),
        "score",
        round(score, 8),
    )
print("PAP_PSP_candidates")
for c, s in zip(candidates, stats):
    if c.node_types in [("paper", "author", "paper"), ("paper", "subject", "paper")]:
        print(
            c.schema_id,
            c.name,
            c.node_types,
            c.relation_ids,
            c.relation_names,
            "instances",
            s.num_instances,
            "D",
            round(s.endpoint_label_disc, 6),
            "C",
            round(s.end_coverage, 6),
            "V",
            round(s.type_diversity, 6),
            "initial_score",
            round(topology_aware_score(s, []), 8),
        )

initial_scores = [
    (topology_aware_score(s, []), c.schema_id, c.name, c.node_types, c.relation_ids)
    for c, s in zip(candidates, stats)
]
print("top_initial")
for score, schema_id, name, node_types, relation_ids in sorted(initial_scores, reverse=True)[:12]:
    print(schema_id, name, node_types, relation_ids, round(score, 8))

more_candidates = generate_schema_candidates(
    relation_tuples=relation_tuples,
    target_node_type=cfg.data.target_node_type,
    min_length=cfg.discovery.min_schema_length,
    max_length=cfg.discovery.max_schema_length,
    require_target_start=cfg.discovery.require_target_start,
    require_target_end=cfg.discovery.require_target_end,
    allow_repeated_node_types=cfg.discovery.allow_repeated_node_types,
    max_candidates=512,
)
print("PAP_PSP_with_512_cap")
for c in more_candidates:
    if c.node_types in [("paper", "author", "paper"), ("paper", "subject", "paper")]:
        print(c.schema_id, c.name, c.node_types, c.relation_ids, c.relation_names)
