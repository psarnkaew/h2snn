"""J5.4 loader check (spec v1.9 F13): PubMed and LastFM with a target-only link.dat.valid.
`before` saves the loader's tensors with the current (J4.2, every-relation) file; `after`, run
once the target-only file is written, compares: the target's train/validation/test positives
must be identical tensors; every non-target relation must have all its link.dat facts in the
message graph (none carved to validation). The 1:1 negatives are reported (they may change,
since they are drawn against the message graph). Read only.
Run from H2S-NN/: python ../docs/lp_evidence/j54_loader_check.py {before|after} <snapshot dir>"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(".").resolve()))
from src.configs.config import TrainConfig  # noqa: E402
from src.data.link_prediction_dataset import (  # noqa: E402
    _LOCAL_HGB_LINK_DATASETS, _load_local_hgb_link_data, build_link_prediction_dataset)

mode, snap = sys.argv[1], Path(sys.argv[2])
POS = ("train_pos_edge_index", "val_pos_edge_index", "test_pos_edge_index")
NEG = ("val_neg_edge_index", "test_neg_edge_index")
ok = True
for dataset in ("pubmed", "lastfm"):
    config = TrainConfig.from_dict({"device": "cpu", "data": {"dataset_name": dataset,
                                                             "root": "data"}})
    data = build_link_prediction_dataset(config)
    split = _load_local_hgb_link_data(config)
    tensors = {name: getattr(data, name) for name in POS + NEG}
    tensors["edge_index"] = data.edge_index
    counts = {rel[1]: (int(split.train_edge_index_by_relation[rel].size(1)),
                       int(split.valid_edge_index_by_relation[rel].size(1)),
                       int(split.link_edge_index_by_relation.get(
                           rel, torch.empty(2, 0)).size(1)))
              for rel in split.train_edge_index_by_relation}
    path = snap / f"j54_{dataset}_before.pt"
    if mode == "before":
        torch.save({"tensors": tensors, "counts": counts}, path)
        print(f"{dataset}: saved; per relation (train, valid, link.dat rows): {counts}")
        continue
    before = torch.load(path)
    target = config.data.target_relation[1]
    for name in POS:
        same = torch.equal(before["tensors"][name], tensors[name])
        ok &= same
        print(f"{dataset} {name}: identical to before {same} ({tensors[name].size(1)})")
    for name in NEG:
        same = torch.equal(before["tensors"][name], tensors[name])
        print(f"{dataset} {name}: identical to before {same} (reported, not gated)")
    for rel, (train, valid, rows) in sorted(counts.items()):
        b_train, b_valid, _ = before["counts"][rel]
        if rel == target:
            same = (train, valid) == (b_train, b_valid)
            ok &= same
            print(f"{dataset} {rel} (target): train {train}, valid {valid}; before {b_train}, "
                  f"{b_valid}: {'same' if same else 'DIFFERENT'}")
        else:
            complete = valid == 0 and train == b_train + b_valid
            ok &= complete
            print(f"{dataset} {rel}: train {train} (before {b_train} + valid {b_valid}), valid "
                  f"{valid}; link.dat rows {rows}: {'complete' if complete else 'INCOMPLETE'}")
    print(f"{dataset} message graph edges: before {before['tensors']['edge_index'].size(1)}, "
          f"after {tensors['edge_index'].size(1)}")
if mode == "after":
    print("ALL PASS" if ok else "SOME FAIL")
