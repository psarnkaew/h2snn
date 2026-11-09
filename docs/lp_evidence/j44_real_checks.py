"""J4.4: the card's negative properties on the real files (`<task>/heart/`), in raw HGB ids:
shape (n x 500 x 2), positives aligned with the loader's split, none is a target-relation positive
of any split in either orientation, every corrupted endpoint has the required type, no duplicate
in a list, the first 250 keep the source and the last 250 keep the destination, no self-pair.
Run from H2S-NN/: python ../docs/lp_evidence/j44_real_checks.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from scripts.lp import build_heart_negatives as hn  # noqa: E402

ROOT = Path("data")
ok_all = True
for name, task in hn.HEART_TASKS.items():
    data = hn.load_task(ROOT, task)
    src_t, _, dst_t = task.target
    raw_type = {}
    for t, ids in data.local_to_raw.items():
        for r in ids.tolist():
            raw_type[r] = t
    positives = set()
    for p in np.concatenate([data.train_pos, data.valid_pos, data.test_pos]):
        positives.add((int(data.local_to_raw[src_t][p[0]]), int(data.local_to_raw[dst_t][p[1]])))
    both = positives | {(v, u) for u, v in positives}
    for split, local_pos in (("valid", data.valid_pos), ("test", data.test_pos)):
        pos = np.load(data.folder / "heart" / f"heart_{split}_pos.npy")
        neg = np.load(data.folder / "heart" / f"heart_{split}_samples.npy")
        checks = {
            "shape": neg.shape == (len(local_pos), 500, 2) and pos.shape == (len(local_pos), 2),
            "positives = loader split": bool((pos == hn.to_raw(data, local_pos)).all()),
            "first half keeps source": bool((neg[:, :250, 0] == pos[:, None, 0]).all()),
            "second half keeps destination": bool((neg[:, 250:, 1] == pos[:, None, 1]).all()),
            "source type": all(raw_type[u] == src_t for u in np.unique(neg[..., 0]).tolist()),
            "destination type": all(raw_type[v] == dst_t for v in np.unique(neg[..., 1]).tolist()),
            "no self-pair": not bool((neg[..., 0] == neg[..., 1]).any()),
        }
        pairs = [tuple(x) for x in neg.reshape(-1, 2).tolist()]
        checks["none is a positive (either orientation)"] = not any(p in both for p in pairs)
        flat = neg[..., 0].astype(np.int64) * (10 ** 7) + neg[..., 1]
        checks["no duplicate in a list"] = all(len(np.unique(row)) == 500 for row in flat)
        ok = all(checks.values())
        ok_all &= ok
        print(f"{name} {split}: {len(local_pos)} lists, {'ALL PASS' if ok else 'FAIL'} "
              f"{ {k: v for k, v in checks.items() if not v} or ''}", flush=True)
print("all tasks pass:", ok_all)
