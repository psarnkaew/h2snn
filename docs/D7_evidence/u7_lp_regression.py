"""U7 evidence, run once with the pre-U7 code and once with the current code:
  python u7_lp_regression.py dump <H2S-NN dir> <out.pt>
  python u7_lp_regression.py compare <a.pt> <b.pt>
Fixtures (written to a temp dir): 'plain' = the tiny LastFM of tests/test_task_heads.py (no
inverse relations); 'tied' = the T11 fixture (rel 3 exact reverse of rel 0, rel 1
self-inverse). Recorded per fixture with fact_layer untied + path_sampler dfs: every split
tensor, negatives, message graph, face groups, and the number of held-out (val+test)
user->artist pairs whose REVERSE orientation is a message edge (a fact-level leak)."""
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
FIELDS = ("edge_index", "edge_type", "train_pos_edge_index", "val_pos_edge_index",
          "val_neg_edge_index", "test_pos_edge_index", "test_neg_edge_index")


def dump(repo: Path, out: Path) -> None:
    sys.path.insert(0, str(repo))
    sys.path.insert(1, str(HERE.parents[2] / "H2S-NN"))  # tests/ for the fixture writers
    from src.configs.config import TrainConfig
    from src.data.link_prediction_dataset import build_link_prediction_dataset
    from tests.test_paths_and_facts import _write_tied_lastfm
    from tests.test_task_heads import _write_tiny_lastfm

    record = {}
    for name, writer in (("plain", _write_tiny_lastfm), ("tied", _write_tied_lastfm)):
        with tempfile.TemporaryDirectory() as tmp:
            writer(Path(tmp))
            config = TrainConfig.from_dict({"device": "cpu", "data": {
                "dataset_name": "lastfm", "root": tmp}})
            data = build_link_prediction_dataset(config)
        for field in FIELDS:
            record[f"{name}/{field}"] = getattr(data, field)
        for g, group in enumerate(data.face_groups):
            record[f"{name}/group{g}/nodes"] = group.face_node_index
            record[f"{name}/group{g}/edges"] = group.face_edge_index
        held = torch.cat([data.val_pos_edge_index, data.test_pos_edge_index], dim=1)
        message = {(int(a), int(b)) for a, b in data.edge_index.t().tolist()}
        reverse_leaks = sum((int(v), int(u)) in message for u, v in held.t().tolist())
        record[f"{name}/reverse_leaks"] = reverse_leaks
        print(f"{name}: held-out pairs {held.size(1)}, reverse orientation in message graph "
              f"{reverse_leaks}, message edges {data.edge_index.size(1)}")
    torch.save(record, out)


def compare(a: Path, b: Path) -> None:
    x, y = torch.load(a, weights_only=False), torch.load(b, weights_only=False)
    for prefix in ("plain", "tied"):
        keys = sorted(k for k in set(x) | set(y) if k.startswith(prefix + "/"))
        bad = [k for k in keys if k not in x or k not in y or not (
            torch.equal(x[k], y[k]) if isinstance(x[k], torch.Tensor) else x[k] == y[k])]
        print(f"{prefix}: {len(keys)} entries, differing {len(bad)}: {bad}")


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(Path(sys.argv[2]).resolve(), Path(sys.argv[3]))
    else:
        compare(Path(sys.argv[2]), Path(sys.argv[3]))
