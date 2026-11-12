"""P0b evidence: WISP draws on fixed fixtures (with and without an adaptive proposal), saved before
and compared after the step refactor (bit-identity of WISP's main-model output).

    python ../docs/partP_evidence/p0b_wisp_reference.py save|compare <file.pt>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "H2S-NN"))
import torch  # noqa: E402

from src.data.wisp_sampler import build_action_graph, sample_wisp_walks  # noqa: E402
from tests.test_wisp import EDGE_INDEX, EDGE_TYPE, NODE_TYPE, SELF_INVERSE, ANCHORS, _biased_proposal  # noqa: E402


def draws():
    graph = build_action_graph(EDGE_INDEX, EDGE_TYPE, int(NODE_TYPE.numel()), SELF_INVERSE)
    out = {}
    for max_length in (2, 3, 4):
        for name, proposal in (("p", None), ("q", _biased_proposal(max_length))):
            d = sample_wisp_walks(graph, NODE_TYPE, ANCHORS, 200, max_length,
                                  torch.Generator().manual_seed(7), proposal)
            out[f"L{max_length}_{name}"] = {f: getattr(d, f) for f in d.__slots__}
    return out


if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "save":
        torch.save(draws(), path)
        print("saved", path)
    else:
        ref, now = torch.load(path), draws()
        bad = [(k, f) for k in ref for f in ref[k]
               if ref[k][f].dtype != now[k][f].dtype or not torch.equal(ref[k][f], now[k][f])]
        n = sum(len(v) for v in ref.values())
        print(f"compared {n} tensors in {len(ref)} draws; differing: {bad or 'none'}")
