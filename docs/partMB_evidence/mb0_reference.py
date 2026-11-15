"""Part MB, MB0: full-batch reference outputs saved before any change (bit-identity checks of
MB2/MB3/MB4), and compared after each unit.

    python ../docs/partMB_evidence/mb0_reference.py save|compare <file.pt>

Saved: (1) full-batch WISP draws, merged into cell groups (`build_wisp_path_groups`), on the
`test_wisp` fixture (l_max 2/3/4, with and without an adaptive proposal) and on a synthetic
ACM-shaped graph; (2) a full-batch anchored v3-B model on that graph: evaluation logits of two
consecutive evaluations (the second one takes the warm start), one training step's loss and every
parameter gradient, and h0 of a featureless type. Synthetic data only.

`compare` prints every differing tensor name and exits 1 if any tensor differs (cluster-setup2 CS7-4;
H2S-NN/tests/test_mb0_reference_compare.py).

Cluster reference (cluster-setup3 card, CS8-1/CS8-2): `docs/partMB_evidence/mb0_reference_cluster.pt`,
saved on node g2n01 (Intel i9-10900K) in env h2snn-b (torch 2.4.1+cu124), CPU only, with the pinned
kernel environment E (ATEN_CPU_CAPABILITY=avx2, MKL_CBWR=AVX2, ONEDNN_MAX_CPU_ISA=AVX2) and 1 thread;
job 16721 step 13, git e0a7ca12fd37a7915d44740bf453926accd8526a; sha256
83f5f57d4feee19624f5bb9e38e935032813546c7a159c4f523d36d183c91700 (copy of
docs/cluster_evidence3/mb0_reference_E.pt). It is portable with E and 1 thread (0/139 differing on
g2n01 again and on g3an01, i9-10900X). `mb0_reference.pt` is the Windows capture, kept as history.
Rule R2 of that card: `compare` refuses to run (exit 2) unless E is in effect, and the references
are computed with 1 thread (restored after).
"""

import os
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


PINNED_MESSAGE = ("bit-exact test needs the pinned kernel environment: "
                  "run through scripts/slurm/run_tests.sbatch")


def kernel_path_is_pinned() -> bool:
    """cluster-setup3 CS8-2: E is in effect (torch's CPU kernels and MKL on the AVX2 path)."""
    return (torch.backends.cpu.get_cpu_capability() == "AVX2"
            and os.environ.get("MKL_CBWR") == "AVX2")


def references_one_thread() -> dict:
    """`references()` with 1 thread (rule R2 of the cluster-setup3 card), restored after."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return references()
    finally:
        torch.set_num_threads(threads)


if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "save":
        torch.save(references_one_thread(), path)
        print("saved", path)
    else:
        if not kernel_path_is_pinned():
            print(f"{PINNED_MESSAGE} (cpu_capability={torch.backends.cpu.get_cpu_capability()}, "
                  f"MKL_CBWR={os.environ.get('MKL_CBWR')})")
            sys.exit(2)
        ref, now = {}, {}
        _flatten("", torch.load(path), ref)
        _flatten("", references_one_thread(), now)
        bad = [k for k in ref if k not in now or ref[k].dtype != now[k].dtype
               or not torch.equal(ref[k], now[k])]
        missing = sorted(set(now) - set(ref))
        # cluster-setup2 CS7-4: every differing name is printed and any difference exits 1.
        print(f"compared {len(ref)} tensors; differing: {len(bad) or 'none'}"
              f"{'' if not missing else f'; new keys {missing}'}")
        for k in bad:
            print(f"  DIFF {k}")
        sys.exit(1 if bad else 0)
