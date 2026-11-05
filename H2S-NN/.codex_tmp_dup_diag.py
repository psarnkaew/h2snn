import sys
import yaml
from collections import Counter
from pathlib import Path

ROOT = Path(r"G:/My Drive/H2S-NN_test")
sys.path.insert(0, str(ROOT))

from src.configs.config import TrainConfig
from src.data.dataset import build_dataset


payload = yaml.safe_load(
    (ROOT / "configs/experiments/acm_h2snn_ampss.yaml").read_text(encoding="utf-8")
)
payload["device"] = "cpu"
data = build_dataset(TrainConfig.from_dict(payload))

all_node_keys = []
all_edge_keys = []
all_node_rel_keys = []
all_full_keys = []

print("groups", len(data.face_groups))
for i, group in enumerate(data.face_groups):
    node_keys = [tuple(row.tolist()) for row in group.face_node_index]
    edge_keys = [tuple(row.tolist()) for row in group.face_edge_index]
    rel_keys = [tuple(row.tolist()) for row in group.relation_index]
    node_rel_keys = list(zip(node_keys, rel_keys))
    full_keys = list(zip(node_keys, edge_keys, rel_keys))

    def dup_count(keys):
        counts = Counter(keys)
        return sum(v - 1 for v in counts.values() if v > 1), len(counts)

    node_dups, node_unique = dup_count(node_keys)
    edge_dups, edge_unique = dup_count(edge_keys)
    node_rel_dups, node_rel_unique = dup_count(node_rel_keys)
    full_dups, full_unique = dup_count(full_keys)
    rel_signature = sorted(set(rel_keys))

    print(
        i,
        group.template_name,
        "faces",
        group.num_faces,
        "rel",
        rel_signature,
        "node_dups",
        node_dups,
        "edge_dups",
        edge_dups,
        "node_rel_dups",
        node_rel_dups,
        "full_dups",
        full_dups,
        "node_unique",
        node_unique,
    )

    all_node_keys.extend((node_key,) for node_key in node_keys)
    all_edge_keys.extend((edge_key,) for edge_key in edge_keys)
    all_node_rel_keys.extend(node_rel_keys)
    all_full_keys.extend(full_keys)

def print_total(label, keys):
    counts = Counter(keys)
    dup = sum(v - 1 for v in counts.values() if v > 1)
    print(label, "total", len(keys), "unique", len(counts), "duplicates", dup)

print_total("all_node_paths", all_node_keys)
print_total("all_edge_paths", all_edge_keys)
print_total("all_node_rel_paths", all_node_rel_keys)
print_total("all_full_paths", all_full_keys)
