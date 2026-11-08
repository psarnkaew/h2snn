"""K1: parameter counts of the arms that the two-latent change affects (T11 fixture, the
`_v2_config` model block). Run from H2S-NN/: python ../docs/WISP_evidence/k1_param_counts.py TAG"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from tests.test_v2_model import _float64_pap, _v2_model  # noqa: E402

data = _float64_pap()
arms = {"full": {}, "endpoint_edges": {"endpoint_edges": True},
        "free_path_maps": {"free_path_maps": True},
        "pair_attention": {"coupling": "pair_attention"}}
out = {}
for context in (True, False):
    for name, ablation in arms.items():
        model = _v2_model(data, ablation=ablation, geometry_context=context)
        encoders = [m for m in (model.path_encoder, getattr(model, "map_encoder", None)) if m is not None]
        out[f"{name}/context_{context}"] = {
            "total": sum(p.numel() for p in model.parameters()),
            "path_encoders": sum(p.numel() for m in encoders for p in m.parameters()),
            "num_path_encoders": len(encoders)}
print(json.dumps({sys.argv[1]: out}, indent=1))
