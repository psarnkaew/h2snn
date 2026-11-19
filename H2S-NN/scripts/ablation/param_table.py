"""Card 4 K6 (4): parameter table of every ablation arm (pre-registration §4.8, B-13) on ACM, DBLP, IMDB and
Freebase at `full`'s configuration: total and the part each arm removes or adds. Parts:
- `g`: path encoder + gate (`path_encoder.*`, `path_gate.*`, or the uniform arm's `path_log_weight`);
- map network (spec 1.25 TD-1): frames, `W_r` and the one per-edge MLP (`node_to_edge.frame_params`,
  `node_to_edge.subspace_params`, `node_to_edge.edge_mlp.*`); the MLP also gives c_e (a_e = c_e^2);
- fact-weight MLP: none since 1.25 (the scale is one output of the map network's MLP; reported as 0);
- head input: the first readout Linear (`readout.0.*`; 3d input under `instance_features`);
- free cell maps: `free_cell_maps.*` (arm free_cell_maps only).
`endpoints`' g must be within 5 % of `full`'s (`endpoint_hidden_width`); `uniform` and `schema` report
g's share of the model. Until `full` is tuned the configuration is the experiment file given (a
placeholder; d and the hidden widths fix the counts, r does not).

    python scripts/ablation/param_table.py --config acm=configs/experiments/acm_h2snn_v3b.yaml ... --out <json>
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402

PARTS = {
    "g": ("path_encoder.", "path_gate.", "path_log_weight"),
    "map_network": ("node_to_edge.frame_params", "node_to_edge.subspace_params",
                    "node_to_edge.edge_mlp."),
    "fact_weight_mlp": (),  # spec 1.25: c_e is an output of node_to_edge.edge_mlp (map_network)
    "head_input": ("readout.0.",),
    "free_cell_maps": ("free_cell_maps.",),
}


def counts(model) -> dict[str, int]:
    out = {"total": sum(p.numel() for p in model.parameters())}
    for part, prefixes in PARTS.items():
        out[part] = sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefixes))
    return out


def table_for(dataset: str, config: Path) -> dict:
    raw_full = yaml.safe_load(config.read_text(encoding="utf-8"))
    data = None
    rows = {}
    for arm in arms.ARM_CHANGES:
        raw = arms.apply_change(raw_full, arm)
        raw["device"] = "cpu"
        cfg = TrainConfig.from_dict(copy.deepcopy(raw))
        if data is None:
            data = build_dataset(cfg)       # one draw: the counts do not depend on the arm's draw
        rows[arm] = counts(build_model(cfg, data))
    full = rows["full"]
    for arm, row in rows.items():
        row["delta_total_vs_full"] = row["total"] - full["total"]
        row["g_share_of_total"] = row["g"] / row["total"]
    rows["endpoints"]["g_relative_to_full_g"] = rows["endpoints"]["g"] / full["g"] - 1.0
    rows["endpoints"]["within_5_percent"] = abs(rows["endpoints"]["g_relative_to_full_g"]) <= 0.05
    return {"config": str(config), "d": raw_full["model"]["d"], "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True, help="dataset=path ...")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for item in args.config:
        dataset, path = item.split("=", 1)
        result[dataset] = table_for(dataset, Path(path))
        e = result[dataset]["rows"]["endpoints"]
        print(dataset, {a: r["total"] for a, r in result[dataset]["rows"].items()},
              "endpoints g vs full:", round(e["g_relative_to_full_g"], 4), flush=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
