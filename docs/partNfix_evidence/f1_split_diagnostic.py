"""F1 diagnostic: is the split difference only the draw rule (u < 0.1 -> valid, ours; u >= 0.9
-> valid, HGB) and our fallback? Replays HGB's rule with random.Random(seed) on our sorted pairs."""
import random, sys
sys.path.insert(0, ".")
from tests.test_hgb_oracle import DATASETS, HGBOracleTests, SEEDS, _hgb_module, _load_hgb, _write
from pathlib import Path
import tempfile, torch
from src.data.link_prediction_dataset import _split_edge_index_hgb_by_head

hgb = _hgb_module()
tmp = Path(tempfile.mkdtemp())
rows = []
for name, build in DATASETS.items():
    folder = tmp / name; _write(folder, build())
    for seed in SEEDS:
        dl = _load_hgb(hgb, folder, seed)
        first = next(iter(dl.links["data"]))
        stored = sorted({(h, t) for h, t, r in build()["links"] if r == first})
        rng = random.Random(seed); last = None; replay = []
        for h, t in stored:
            if h != last:
                last = h; continue
            if not rng.random() < 0.9:
                replay.append((h, t))
        hgb_valid = [(int(h), int(t)) for h, t in zip(*dl.valid_pos[first])]
        _, ours = _split_edge_index_hgb_by_head(torch.tensor(stored).t(), 0.10, seed)
        rows.append((name, seed, len(stored), len(hgb_valid), ours.size(1), replay == hgb_valid))
for r in rows:
    print("%-26s seed %d  facts %2d  HGB valid %2d  ours valid %2d  HGB rule replayed == HGB: %s" % r)
print("all replays equal:", all(r[-1] for r in rows))
