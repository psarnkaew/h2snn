"""T11: bit-for-bit regression of the v1 (map_family=dense) forward/backward pass.

Usage (from the H2S-NN repo root, in the h2snn env):
    python <this> capture <out.pt>    # run against the PRE-change code
    python <this> compare <ref.pt>    # run against the POST-change code

Each case seeds the global RNG, builds data + model, runs one eval forward and one train
forward+backward, and records outputs plus every parameter gradient. `compare` requires
torch.equal (bit-for-bit) on every tensor and identical state_dict key lists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.configs.config import TrainConfig  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402
from synthetic_graph import make_random_pap_data  # noqa: E402
from tests.test_models import _make_prepared_data  # noqa: E402

SMALL = {"d": 4, "d_r": 2, "d_h": 8, "K": 2, "classifier_hidden_dim": 8}
MEDIUM = {"d": 8, "d_r": 4, "d_h": 8, "K": 3, "classifier_hidden_dim": 16}


def _cases() -> list[tuple[str, dict, object]]:
    discovery = {
        "enabled": True,
        "mode": "auto",
        "max_total_instances": 10,
        "max_total_incidences": 30,
    }
    return [
        ("toy_acm", {"model": SMALL}, lambda: _make_prepared_data("acm")),
        ("toy_dblp", {"model": SMALL}, lambda: _make_prepared_data("dblp")),
        ("toy_imdb", {"model": SMALL}, lambda: _make_prepared_data("imdb")),
        (
            "toy_acm_auto",
            {"model": dict(SMALL, K=1), "discovery": discovery},
            lambda: _make_prepared_data("acm", with_discovery=True),
        ),
        ("pap", {"model": MEDIUM}, lambda: make_random_pap_data(seed=3)),
        ("pap_lstm_tri", {"model": dict(MEDIUM, face_encoder="lstm", face_map_param="triangular")},
         lambda: make_random_pap_data(seed=4)),
        ("pap_unconstrained", {"model": dict(MEDIUM, use_orthogonal=False)},
         lambda: make_random_pap_data(seed=5)),
        ("pap_identity", {"model": dict(MEDIUM, ablate_identity_maps=True)},
         lambda: make_random_pap_data(seed=6)),
        ("pap_no_delta1", {"model": dict(MEDIUM, ablate_no_delta1=True)},
         lambda: make_random_pap_data(seed=7)),
    ]


def _run_case(raw: dict, make_data) -> dict[str, object]:
    torch.manual_seed(1234)
    data = make_data()
    cfg = TrainConfig.from_dict(dict(raw, device="cpu", checkpoint_metric="acc"))
    torch.manual_seed(42)
    model = build_model(cfg, data)
    record: dict[str, object] = {"state_keys": list(model.state_dict().keys())}
    for name, value in model.state_dict().items():
        record[f"init/{name}"] = value.detach().clone()
    model.eval()
    with torch.no_grad():
        out = model(data)
    for field in ("logits", "x_star", "s_star", "delta0_h0", "delta1_delta0_h0", "raw_face_curl"):
        record[f"eval/{field}"] = getattr(out, field).detach().clone()
    model.train()
    torch.manual_seed(7)
    out = model(data)
    loss = out.logits.square().sum() + out.s_star.square().mean() + out.x_star.square().mean()
    loss.backward()
    record["train/logits"] = out.logits.detach().clone()
    for name, parameter in model.named_parameters():
        record[f"grad/{name}"] = (
            None if parameter.grad is None else parameter.grad.detach().clone()
        )
    return record


def main() -> None:
    mode, path = sys.argv[1], Path(sys.argv[2])
    results = {name: _run_case(raw, make) for name, raw, make in _cases()}
    if mode == "capture":
        torch.save(results, path)
        total = sum(len(r) for r in results.values())
        print(f"captured {len(results)} cases, {total} entries -> {path}")
        return
    reference = torch.load(path, weights_only=False)
    mismatches = 0
    compared = 0
    for case, record in results.items():
        ref = reference[case]
        if set(ref) != set(record):
            print(f"[{case}] key sets differ: {sorted(set(ref) ^ set(record))[:10]}")
            mismatches += 1
            continue
        for key, value in record.items():
            compared += 1
            expected = ref[key]
            if isinstance(value, torch.Tensor):
                same = isinstance(expected, torch.Tensor) and value.dtype == expected.dtype
                if same:
                    try:
                        # Exact (rtol=atol=0) on finite entries; NaN positions must coincide.
                        torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0, equal_nan=True)
                    except AssertionError:
                        same = False
            else:
                same = value == expected
            if not same:
                mismatches += 1
                detail = ""
                if isinstance(value, torch.Tensor) and isinstance(expected, torch.Tensor) and value.shape == expected.shape:
                    detail = f" max|diff|={float((value - expected).abs().max()):.3e}"
                print(f"[{case}] MISMATCH {key}{detail}")
    print(f"compared {compared} entries across {len(results)} cases; mismatches={mismatches}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
