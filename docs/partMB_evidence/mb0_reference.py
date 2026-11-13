"""Part MB, MB0: full-batch reference outputs saved before any change (bit-identity checks of
MB2/MB3/MB4), and compared after each unit.

    python ../docs/partMB_evidence/mb0_reference.py save|compare <file.pt>

Saved: (1) full-batch WISP draws, merged into cell groups (`build_wisp_path_groups`), on the
`test_wisp` fixture (l_max 2/3/4, with and without an adaptive proposal) and on a synthetic
ACM-shaped graph; (2) a full-batch anchored v3-B model on that graph: evaluation logits of two
consecutive evaluations (the second one takes the warm start), one training step's loss and every
parameter gradient, and h0 of a featureless type. Synthetic data only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "H2S-NN"))
import torch  # noqa: E402

from src.data.wisp_sampler import build_wisp_path_groups  # noqa: E402
from tests.partO_fixtures import acm_like_hetero, patched_acm_loader  # noqa: E402
from tests.test_wisp import ANCHORS, EDGE_INDEX, EDGE_TYPE, NODE_TYPE, SELF_INVERSE, _biased_proposal  # noqa: E402

GROUP_FIELDS = ("face_node_index", "face_edge_index", "relation_index", "face_edge_sign",
                "path_weight", "signature_index")


def _groups(groups):
    return [{f: getattr(g, f) for f in GROUP_FIELDS} | {"table": torch.tensor(g.signature_table)}
            for g in groups]


def _config(featureless: bool = False):
    from src.configs.config import TrainConfig

    return TrainConfig.from_dict({
        "device": "cpu", "seed": 3,
        "data": {"dataset_name": "acm", "wisp_max_length": 3, "wisp_walks_per_anchor": 5},
        "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                  "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full"},
        "ablation": {"coupling": "symmetric"}})


def references() -> dict:
    out: dict = {}
    for max_length in (2, 3, 4):
        for name, proposal in (("p", None), ("q", _biased_proposal(max_length))):
            groups = build_wisp_path_groups(EDGE_INDEX, EDGE_TYPE, NODE_TYPE, SELF_INVERSE,
                                            ANCHORS, 40, max_length, 11, proposal)
            out[f"draw_L{max_length}_{name}"] = _groups(groups)

    from src.data.dataset import build_dataset
    from src.models.h2snn_model import build_model

    hetero = acm_like_hetero(seed=9)
    del hetero["term"].x  # one featureless type (learned per-node rows)
    cfg = _config()
    with patched_acm_loader(hetero):
        data = build_dataset(cfg)
    out["acm_draw"] = _groups(data.face_groups)
    torch.manual_seed(0)
    model = build_model(cfg, data)
    model.eval()
    with torch.no_grad():
        out["eval_logits_1"] = model(data).logits.clone()
        out["eval_logits_2"] = model(data).logits.clone()  # warm-started evaluation
        out["h0"] = model.encode_nodes(data.x, data.node_type).clone()
    model.train()
    torch.manual_seed(1)
    logits = model(data, epoch=0).logits
    loss = torch.nn.functional.cross_entropy(logits[data.target_train_mask],
                                             data.target_y[data.target_train_mask])
    loss.backward()
    out["train_loss"] = loss.detach().clone()
    out["grads"] = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return out


def _flatten(prefix, value, into):
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}", v, into)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _flatten(f"{prefix}[{i}]", v, into)
    else:
        into[prefix] = value


if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "save":
        torch.save(references(), path)
        print("saved", path)
    else:
        ref, now = {}, {}
        _flatten("", torch.load(path), ref)
        _flatten("", references(), now)
        bad = [k for k in ref if k not in now or ref[k].dtype != now[k].dtype
               or not torch.equal(ref[k], now[k])]
        missing = sorted(set(now) - set(ref))
        print(f"compared {len(ref)} tensors; differing: {bad[:5] or 'none'}"
              f"{'' if not missing else f'; new keys {missing[:3]}'}")
