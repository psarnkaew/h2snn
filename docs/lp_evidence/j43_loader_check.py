"""J4.3 (Part J): the real freebase_bb through `build_link_prediction_dataset` with every existing
leak check (and J4.2's guard), plus the pair-level checks on the loaded tensors.
Run from H2S-NN/: python ../docs/lp_evidence/j43_loader_check.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.configs.config import TrainConfig  # noqa: E402
from src.data.link_prediction_dataset import build_link_prediction_dataset  # noqa: E402

t0 = time.time()
config = TrainConfig.from_dict({"device": "cpu", "seed": 3, "data": {
    "dataset_name": "freebase_bb", "task_type": "link_prediction", "root": "data",
    "target_relation": ["book", "rel_0", "book"], "fact_layer": "tied", "path_sampler": "walk",
    "face_sample_k": 4}})
data = build_link_prediction_dataset(config)
parts = {}
for name in ("train", "val", "test"):
    index = getattr(data, f"{name}_pos_edge_index")
    rows = {tuple(p) for p in index.t().tolist()}
    parts[name] = {(min(u, v), max(u, v)) for u, v in rows}
    print(f"{name}: {index.size(1)} positive rows, {len(parts[name])} unordered pairs, "
          f"self-loops {int((index[0] == index[1]).sum())}")
print("pairs shared between parts:", len(parts["train"] & parts["val"]),
      len(parts["train"] & parts["test"]), len(parts["val"] & parts["test"]))
message = {tuple(p) for p in data.edge_index.t().tolist()}
held = {tuple(p) for p in data.val_pos_edge_index.t().tolist()} | \
    {tuple(p) for p in data.test_pos_edge_index.t().tolist()}
print("held-out rows in the message graph: as stored", len(held & message), "reversed",
      len({(v, u) for u, v in held} & message))
b0 = data.src_node_offset
loops = sum((b0 + i, b0 + i) in message for i in range(int(data.num_src_nodes)))
print(f"self-loops of books in the message graph: {loops}; message-graph edges "
      f"{data.edge_index.size(1)}; loaded with every leak check passing ({time.time() - t0:.0f} s)")
