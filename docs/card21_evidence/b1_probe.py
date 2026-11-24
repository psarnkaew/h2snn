"""Card 21 B1 evidence: PyG HGTLoader with unequal per-type hop lists (H-Pokec shape: target 4 hops, other type 3).

Unpadded: torch_sparse `hgt_sample` reads `num_samples[item][3]` past the end of a 3-element list. Evidence is either
the raise ("cannot sample n_sample <= 0 samples") or more `item` nodes in a batch than 3 hops can give (3 x 4 = 12).
Padded (`equal_hop_fanout`): no raise, at most 4 x 4 = 16 items. Run from H2S-NN/: python ../docs/card21_evidence/b1_probe.py
"""
import sys
from collections import Counter

import torch
from torch_geometric.loader import HGTLoader

sys.path.insert(0, ".")
from src.baselines.minibatch import equal_hop_fanout  # noqa: E402
from tests.test_card21_loader import B1_FANOUT, b1_graph  # noqa: E402

data = b1_graph()
seeds_in = ("user", data["user"].train_mask)
for label, fanout in (("unpadded", B1_FANOUT), ("padded", equal_hop_fanout(B1_FANOUT))):
    raises, item_counts, messages = 0, Counter(), Counter()
    batches = 0
    for seed in range(10):
        torch.manual_seed(seed)
        loader = HGTLoader(data, num_samples=fanout, input_nodes=seeds_in, batch_size=8, shuffle=True)
        for _ in range(3):  # 3 passes of 30 batches
            it = iter(loader)
            while True:
                try:
                    batch = next(it)
                except StopIteration:
                    break
                except RuntimeError as err:
                    raises += 1
                    messages[str(err).splitlines()[0][:90]] += 1
                    break  # the iterator is dead after a raise; next pass
                batches += 1
                item_counts[int(batch["item"].num_nodes)] += 1
    over = sum(c for k, c in item_counts.items() if k > 12)
    print(f"{label}: fanout={fanout}")
    print(f"  batches={batches} raises={raises} messages={dict(messages)}")
    print(f"  item nodes per batch: min={min(item_counts)} max={max(item_counts)}; batches with > 12 items "
          f"(more than 3 hops x 4 can give) = {over}")
