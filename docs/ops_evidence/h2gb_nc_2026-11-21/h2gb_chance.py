"""Chance levels of the H2GB NC validation metrics (ops, 2026-11-21). Run from the H2S-NN checkout root.

For each dataset: classes, split sizes, and on the B1a validation subset (seed 0) and the whole validation split:
- majority: always predict the most frequent training class -> accuracy, Macro-F1 (classes present in the split)
- prior: predict at random with the training class frequencies -> expected accuracy, expected Macro-F1
- uniform: predict uniformly at random over all classes -> expected accuracy (1/C), expected Macro-F1
Expected F1 of class c for a random predictor with rate q_c on true share p_c: 2 p_c q_c / (p_c + q_c).
Binary datasets also give the positive-class F1 (H2GB's `f1`, class 1).
"""
import json
import sys

import torch

sys.path.insert(0, ".")
from src.data.h2gb_nc import ALL_H2GB_NC_SPECS, load_h2gb_nc  # noqa: E402
from src.evaluation.validation_subset import validation_subset  # noqa: E402

DATASETS = sys.argv[1:] or ["ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns"]


def stats(y_split, prior, majority, n_classes):
    present = torch.unique(y_split).tolist()
    n = y_split.numel()
    p = torch.bincount(y_split, minlength=n_classes).double() / n
    out = {"n": int(n), "classes_present": len(present), "largest_class_share": float(p.max())}
    # majority predictor
    f1 = []
    for c in present:
        if c == majority:
            f1.append(2 * p[c] / (1 + p[c]))
        else:
            f1.append(torch.tensor(0.0, dtype=torch.double))
    out["majority_acc"] = float(p[majority])
    out["majority_macro_f1"] = float(torch.stack(f1).mean())
    for name, q in (("prior", prior), ("uniform", torch.full((n_classes,), 1.0 / n_classes, dtype=torch.double))):
        f1 = [2 * p[c] * q[c] / (p[c] + q[c]) if (p[c] + q[c]) > 0 else torch.tensor(0.0) for c in present]
        out[f"{name}_acc"] = float((p * q).sum())
        out[f"{name}_macro_f1"] = float(torch.stack([torch.as_tensor(v, dtype=torch.double) for v in f1]).mean())
    if n_classes == 2:
        out["majority_pos_f1"] = float(2 * p[1] / (1 + p[1])) if majority == 1 else 0.0
        out["prior_pos_f1"] = float(2 * p[1] * prior[1] / (p[1] + prior[1]))
        out["all_positive_f1"] = float(2 * p[1] / (1 + p[1]))
    return out


result = {}
for ds in DATASETS:
    spec = ALL_H2GB_NC_SPECS[ds]
    data = load_h2gb_nc(ds, "data", seed=0)
    store = data[spec.target_node_type]
    y = store.y.view(-1).long()
    n_classes = int(y[y >= 0].max()) + 1
    idx = {s: store[f"{s}_mask"].nonzero().view(-1) for s in ("train", "val", "test")}
    y_train = y[idx["train"]]
    prior = torch.bincount(y_train, minlength=n_classes).double() / y_train.numel()
    majority = int(prior.argmax())
    sub = validation_subset(ds, idx["val"], run_seed=0)
    result[ds] = {"n_classes": n_classes, "sizes": {s: int(v.numel()) for s, v in idx.items()},
                  "val_subset": stats(y[sub], prior, majority, n_classes),
                  "val_full": stats(y[idx["val"]], prior, majority, n_classes)}
    print(ds, json.dumps(result[ds]), flush=True)
    del data
json.dump(result, open(f"logs/h2gb_chance_{'_'.join(DATASETS) if len(DATASETS) < 8 else 'all'}.json", "w"), indent=1)
