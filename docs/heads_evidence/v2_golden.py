"""Golden outputs of the v2 node-classification model (Phase 2 step 0).

    python <this> capture <out.pt>    # run BEFORE the v1 removal (pre-change code)
    python <this> compare <ref.pt>    # run after; exits 1 on any mismatch
    python <this> recapture <ref.pt> <out.pt>   # re-record outputs on ref's stored fixtures

Each case seeds the RNG, builds the v2 model on a stored fixture, records the initial
state_dict, eval logits, train-mode logits and every parameter gradient. `capture` also
stores the fixture tensors, so `compare` does not depend on test-fixture code. `compare`
requires torch.equal (bit-for-bit) on every tensor and identical key sets.

Cases are described abstractly (model overrides + ablation names); `build_config` maps them
to whichever config format the current code has (switches inside `model:` before the move,
an `ablation:` block after it).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2] / "H2S-NN"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.configs.config import AblationConfig, TrainConfig  # noqa: E402
from src.data.structures import FaceGroup, PreparedH2SData  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402

SMALL = {"d": 4, "d_r": 2, "K": 2, "classifier_hidden_dim": 8, "face_mlp_hidden_dim": 8,
         "edge_mlp_hidden_dim": 8}
MEDIUM = {"d": 8, "d_r": 4, "K": 2, "classifier_hidden_dim": 16, "face_mlp_hidden_dim": 16,
          "edge_mlp_hidden_dim": 16}

# Ablation name -> pre-move model keys (D4 unit 5 format).
OLD_ABLATION_KEYS = {
    "no_path_cells": {"ablate_no_path_cells": True},
    "uniform_path_gate": {"uniform_path_gate": True},
    "free_path_maps": {"path_maps": "free"},
    "endpoint_edges": {"path_mode": "endpoint_edges"},
    "holonomy_edges": {"path_mode": "holonomy_edges"},
    "per_side_scale": {"map_scale": "per_side"},
    "h0_only_readout": {"readout": "h0_only"},
}

CASES: list[tuple[str, str, dict, tuple[str, ...]]] = [
    ("toy_acm", "toy_acm", SMALL, ()),
    ("toy_dblp", "toy_dblp", SMALL, ()),
    ("toy_imdb", "toy_imdb", SMALL, ()),
    ("pap", "pap", MEDIUM, ()),
    ("pap_dense", "pap", dict(MEDIUM, map_family="dense"), ()),
    ("pap_no_checkpointing", "pap", dict(MEDIUM, checkpointing=False), ()),
    *[(f"pap_{name}", "pap", MEDIUM, (name,)) for name in OLD_ABLATION_KEYS],
]


def build_config(model: dict, ablations: tuple[str, ...]) -> TrainConfig:
    raw: dict = {"device": "cpu", "checkpoint_metric": "acc",
                 "model": {"map_family": "factorized", **model}}
    if "ablation" in TrainConfig.__dataclass_fields__:  # after the move (Phase 2 step 2)
        raw["ablation"] = {name: True for name in ablations}
        # D10: the golden outputs are the pre-D10 (symmetric) model; the default is now
        # cooperative, so the harness names the symmetric mode explicitly.
        if "coupling" in AblationConfig.__dataclass_fields__:
            raw["ablation"]["coupling"] = "symmetric"
        # D12: the golden outputs are the D11 geometry (from h0 alone) and the first-order
        # update; the defaults are now geometry_context and filter_bank, named explicitly here.
        from src.configs.config import ModelConfig

        if "geometry_context" in ModelConfig.__dataclass_fields__:
            raw["model"]["geometry_context"] = False
        if "encoder" in ModelConfig.__dataclass_fields__:
            raw["model"]["encoder"] = "first_order"
    else:
        raw["model"]["architecture"] = "v2"
        for name in ablations:
            raw["model"].update(OLD_ABLATION_KEYS[name])
    return TrainConfig.from_dict(raw)


# ----------------------------------------------------------------------------- fixtures
def _fixture(name: str) -> PreparedH2SData:  # capture only
    sys.path.insert(0, str(REPO))
    if name.startswith("toy_"):
        from tests.test_models import _make_prepared_data

        return _make_prepared_data(name.removeprefix("toy_"))
    from tests.test_factorized_maps import _random_pap_data

    return _random_pap_data(seed=3)


def _serialize(data: PreparedH2SData) -> dict:
    fields = {name: getattr(data, name) for name in PreparedH2SData.__slots__}
    fields["face_groups"] = [
        {name: getattr(group, name) for name in FaceGroup.__slots__}
        for group in data.face_groups
    ]
    fields["node_slices"] = dict(data.node_slices)
    fields["feature_dims"] = dict(data.feature_dims)
    if fields.get("discovery") is not None or fields.get("selector_report") is not None:
        raise ValueError("golden fixtures must not carry AMPSS metadata")
    fields.pop("discovery", None)
    fields.pop("selector_report", None)
    return fields


def _deserialize(fields: dict) -> PreparedH2SData:
    group_slots = set(FaceGroup.__slots__)
    groups = tuple(
        FaceGroup(**{k: v for k, v in group.items() if k in group_slots})
        for group in fields["face_groups"]
    )
    data_slots = set(PreparedH2SData.__slots__)
    kwargs = {k: v for k, v in fields.items() if k in data_slots and k != "face_groups"}
    return PreparedH2SData(face_groups=groups, **kwargs)


# ----------------------------------------------------------------------------- runs
def run_case(data: PreparedH2SData, model_overrides: dict, ablations: tuple[str, ...]) -> dict:
    config = build_config(model_overrides, ablations)
    torch.manual_seed(42)
    model = build_model(config, data)
    record: dict[str, object] = {}
    for name, value in model.state_dict().items():
        record[f"init/{name}"] = value.detach().clone()
    model.eval()
    with torch.no_grad():
        record["eval/logits"] = model(data).logits.detach().clone()
    model.train()
    torch.manual_seed(7)
    output = model(data)
    (output.logits.square().sum() + output.x_star.square().mean()).backward()
    record["train/logits"] = output.logits.detach().clone()
    for name, parameter in model.named_parameters():
        record[f"grad/{name}"] = None if parameter.grad is None else parameter.grad.detach().clone()
    return record


def compare(reference_path: Path) -> tuple[int, int, list[str]]:
    """(entries compared, mismatches, messages) against a captured reference."""
    reference = torch.load(reference_path, weights_only=False)
    compared = 0
    messages: list[str] = []
    for case, fixture_name, model_overrides, ablations in CASES:
        entry = reference[case]
        data = _deserialize(entry["data"])
        record = run_case(data, model_overrides, ablations)
        expected = entry["record"]
        if set(record) != set(expected):
            messages.append(f"[{case}] key sets differ: {sorted(set(record) ^ set(expected))[:8]}")
            continue
        for key, value in record.items():
            compared += 1
            want = expected[key]
            same = (value is None and want is None) or (
                isinstance(value, torch.Tensor) and isinstance(want, torch.Tensor)
                and value.dtype == want.dtype and value.shape == want.shape
                and torch.equal(value, want)
            )
            if not same:
                detail = ""
                if isinstance(value, torch.Tensor) and isinstance(want, torch.Tensor) \
                        and value.shape == want.shape:
                    detail = f" max|diff|={float((value - want).abs().max()):.3e}"
                messages.append(f"[{case}] MISMATCH {key}{detail}")
    return compared, len(messages), messages


def main() -> None:
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "capture":
        torch.manual_seed(1234)
        fixtures = {name: _fixture(name) for name in ("toy_acm", "toy_dblp", "toy_imdb", "pap")}
        results = {}
        for case, fixture_name, model_overrides, ablations in CASES:
            data = fixtures[fixture_name]
            results[case] = {"data": _serialize(data),
                             "record": run_case(data, model_overrides, ablations)}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(results, path)
        total = sum(len(entry["record"]) for entry in results.values())
        print(f"captured {len(results)} cases, {total} entries -> {path}")
        return
    if mode == "recapture":
        # Re-record outputs on the fixtures stored in an existing reference (the fixture
        # builder `tests.test_models` no longer exists). Only the records change.
        out = Path(sys.argv[3])
        reference = torch.load(path, weights_only=False)
        results = {}
        for case, _, model_overrides, ablations in CASES:
            data = _deserialize(reference[case]["data"])
            results[case] = {"data": reference[case]["data"],
                             "record": run_case(data, model_overrides, ablations)}
        torch.save(results, out)
        total = sum(len(entry["record"]) for entry in results.values())
        print(f"re-captured {len(results)} cases, {total} entries on stored fixtures -> {out}")
        return
    compared, mismatches, messages = compare(path)
    for message in messages[:40]:
        print(message)
    print(f"compared {compared} entries across {len(CASES)} cases; mismatches={mismatches}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
