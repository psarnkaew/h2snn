"""Schema read-out and coverage of trained node-classification runs (spec v1.19 §6; card Part O, O2).
Evaluation only: loads each run (`config.yaml` + a checkpoint, `src.analysis.model_loading`),
rebuilds its fixed evaluation draw from the saved base seed and computes, per sub-path schema,
c_sigma, I_sigma, rho_sigma = I_sigma / c_sigma and the mean g(xi_f) (`src.analysis.schema_readout`);
across the runs of a dataset, Kendall tau of their I_sigma rankings. Delta L_sigma is not computed.

    python scripts/schema_readout.py --runs <run_dir> [<run_dir> ...] --out_dir <dir> [--device cpu]

Output: `<out_dir>/<dataset>/<run>.json` per run and `<out_dir>/<dataset>/schema_readout.md` per
dataset (listed meta-paths per seed; top ten length-2 sub-path schemas by I_sigma; number of
distinct schemas; Kendall tau between runs).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.analysis.model_loading import (  # noqa: E402
    evaluation_cell_weights,
    load_trained_model,
    nonempty_groups,
)
from src.analysis.schema_readout import (  # noqa: E402
    kendall_matrix,
    literature_signatures,
    run_record,
    schema_statistics,
)

LOGGER = logging.getLogger("schema_readout")
TOP = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="Run directories (each with config.yaml and checkpoints/<tag>.pt).")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tag", type=str, default="best", help="Checkpoint tag (NC: best).")
    parser.add_argument("--permutations", type=int, default=1000,
                        help="Permutations of the Kendall tau null.")
    return parser.parse_args(argv)


def read_run(run_dir: Path, device: str, tag: str) -> dict[str, Any]:
    model, data, cfg = load_trained_model(run_dir, device=device, tag=tag)
    if cfg.task_type != "node_classification":
        raise ValueError(f"{run_dir}: the read-out is defined for node classification, got "
                         f"{cfg.task_type!r}.")
    checkpoint = torch.load(run_dir / "checkpoints" / f"{tag}.pt", map_location="cpu",
                            weights_only=False)
    groups = nonempty_groups(data.face_groups)
    weights = evaluation_cell_weights(model, data)
    stats = schema_statistics(groups, weights, data.node_type, data.fact_layer.self_inverse)
    literature = literature_signatures(cfg.data.dataset_name, data.fact_layer,
                                       data.relation_tuples, data.node_type_names)
    meta = {"run": run_dir.name, "run_dir": str(run_dir), "dataset": cfg.data.dataset_name,
            "seed": cfg.seed, "path_source": cfg.data.path_source,
            "wisp_max_length": cfg.data.wisp_max_length,
            "wisp_walks_per_anchor": cfg.data.wisp_walks_per_anchor,
            "checkpoint": tag, "checkpoint_epoch": int(checkpoint.get("epoch", -1))}
    return run_record(stats, data.node_type_names, data.relation_names, literature, meta)


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def dataset_table(records: list[dict[str, Any]], permutations: int) -> str:
    """Markdown table of one dataset's runs (ordered by seed)."""
    records = sorted(records, key=lambda r: r["seed"])
    lines = [f"# Schema read-out — {records[0]['dataset']}", "",
             "Evaluation draw of each run (base seed), checkpoint `"
             f"{records[0]['checkpoint']}`. Descriptive; no test (spec §6).", ""]
    if records[0]["literature"] is not None:
        lines += ["## Listed meta-paths (c, I, rho per seed)", "",
                  "| meta-path | schema | " + " | ".join(f"seed {r['seed']} c / I / rho"
                                                        for r in records) + " |",
                  "|---|---|" + "---|" * len(records)]
        for i, item in enumerate(records[0]["literature"]):
            cells = []
            for r in records:
                listed = r["literature"][i]
                cells.append(f"{_number(listed['c'])} / {_number(listed['I'])} / "
                             f"{_number(listed['rho'])}" + ("" if listed["in_draw"] else
                                                          " (absent)"))
            lines.append(f"| {item['listed']} | `{item['relations']}` | " + " | ".join(cells)
                         + " |")
        lines.append("")
    lines += ["## Top ten length-2 sub-path schemas by I", ""]
    for r in records:
        lines += [f"### seed {r['seed']} (`{r['run']}`)", "",
                  "| rank | schema | I | c | rho | mean g | cells |", "|---|---|---|---|---|---|---|"]
        top = [s for s in r["schemas"] if s["span"] == 2][:TOP]
        for rank, s in enumerate(top, start=1):
            lines.append(f"| {rank} | `{s['relations']}` | {_number(s['I'])} | {_number(s['c'])} | "
                         f"{_number(s['rho'])} | {_number(s['mean_g'])} | {s['num_cells']} |")
        lines.append("")
    lines += ["## Distinct sub-path schemas drawn", "", "| seed | all spans | length 2 |",
              "|---|---|---|"]
    lines += [f"| {r['seed']} | {r['num_distinct_schemas']} | {r['num_distinct_length2_schemas']} |"
              for r in records]
    lines.append("")
    if len(records) > 1:
        for title, span in (("all sub-path schemas", None), ("length-2 sub-path schemas", 2)):
            lines += [f"## Kendall tau of the I rankings between seeds ({title})", "",
                      "| run a | run b | tau | p (one-sided, permutation) |", "|---|---|---|---|"]
            lines += [f"| {row['a']} | {row['b']} | {_number(row['tau'])} | {_number(row['p'])} |"
                      for row in kendall_matrix(records, span=span, permutations=permutations)]
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for run_dir in args.runs:
        record = read_run(run_dir, args.device, args.tag)
        out = args.out_dir / record["dataset"] / f"{record['run']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", out)
        by_dataset.setdefault(record["dataset"], []).append(record)
    for dataset, records in by_dataset.items():
        out = args.out_dir / dataset / "schema_readout.md"
        out.write_text(dataset_table(records, args.permutations), encoding="utf-8")
        LOGGER.info("wrote %s", out)


if __name__ == "__main__":
    main()
