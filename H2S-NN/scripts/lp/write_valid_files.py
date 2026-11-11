"""Write `link.dat.valid` for HGB PubMed and LastFM: the target relation's rows of the loader's
carve (spec v1.9 F13; task card Part J, J5.4), or every relation's (spec v1.8 F9, J4.2 (c)).

F13 (default, `--mode target`): the file lists only the target relation's validation facts, the
same facts the J4.2 file listed for it, so every non-target relation stays complete in the
message graph, as in the derived tasks. `--mode carve` keeps the J4.2 file (every relation).

The loader carves validation from `link.dat` for every relation (`_split_validation` without a
file: `_split_edge_index_hgb_by_head`, ratio `link_val_ratio` of the dataset's preset, seed
`acm_split_seed + relation_id * 9973`). This script runs exactly that carve and writes every
relation's validation rows, in raw HGB ids with the row's own fourth column, so that the loader,
which now reads the file, reproduces today's split with no numeric change (checked by
`tests/test_lp_valid_files.py`). An existing file is overwritten only when it is one of the two files this script writes (the
J4.2 carve or the target-only file); anything else raises.

Run from H2S-NN/: python scripts/lp/write_valid_files.py [--datasets pubmed lastfm] [--mode target]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import LINK_PREDICTION_PRESETS, TrainConfig  # noqa: E402
from src.data.data_layout import REPO_DATA_ROOT, lp_root  # noqa: E402
from src.data.fact_layer import inverse_relations_typed  # noqa: E402
from src.data.link_prediction_dataset import (  # noqa: E402
    _LOCAL_HGB_LINK_DATASETS,
    _load_local_hgb_nodes,
    _resolve_target_relation_ids,
    _read_local_hgb_link_file,
    _relation_id,
    _split_validation,
)


def carve_rows(data_root: Path, dataset: str) -> tuple[bytes, dict]:
    """The bytes of `link.dat.valid` for `dataset` (the carve, every relation) and a summary."""
    spec = _LOCAL_HGB_LINK_DATASETS[dataset]
    folder = lp_root(data_root) / str(spec["folder"])
    config = TrainConfig.from_dict({"device": "cpu", "data": {"dataset_name": dataset,
                                                             "root": str(data_root)}})
    preset = LINK_PREDICTION_PRESETS[dataset]
    if config.data.link_val_ratio != preset["link_val_ratio"]:
        raise ValueError(f"{dataset}: config link_val_ratio {config.data.link_val_ratio} is not "
                         f"the preset's {preset['link_val_ratio']}.")
    nodes = _load_local_hgb_nodes(folder, spec["node_type_names"])
    link = _read_local_hgb_link_file(folder / "link.dat", nodes)
    test = _read_local_hgb_link_file(folder / "link.dat.test", nodes)
    relations = sorted(set(link) | set(test))
    type_sizes = {name: int(nodes.data[name].num_nodes) for name in nodes.data.node_types}
    rho = inverse_relations_typed(link, type_sizes)
    _, valid = _split_validation(link, rho, relations, None,
                                 val_ratio=config.data.link_val_ratio,
                                 split_seed=config.data.acm_split_seed)

    to_raw = {(nodes.raw_to_type_name[raw], local): raw
              for raw, local in nodes.raw_to_local.items()}
    fourth: dict[tuple[int, int, int], str] = {}
    with (folder / "link.dat").open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            fourth.setdefault((int(parts[0]), int(parts[1]), int(parts[2])), parts[3])
    rows = []
    for relation in sorted(valid, key=_relation_id):
        rel_id = _relation_id(relation)
        for u, v in valid[relation].t().tolist():
            src, dst = to_raw[(relation[0], u)], to_raw[(relation[2], v)]
            rows.append(f"{src}\t{dst}\t{rel_id}\t{fourth[(src, dst, rel_id)]}\n")
    summary = {
        "dataset": dataset, "folder": str(spec["folder"]),
        "link_val_ratio": config.data.link_val_ratio,
        "acm_split_seed": config.data.acm_split_seed,
        "rows_per_relation": {relation[1]: int(valid[relation].size(1))
                              for relation in sorted(valid, key=_relation_id)},
    }
    return "".join(rows).encode("utf-8"), summary


def target_rows(data_root: Path, dataset: str) -> tuple[bytes, dict]:
    """The target relation's rows of `carve_rows` (F13), byte-identical, and a summary."""
    content, summary = carve_rows(data_root, dataset)
    config = TrainConfig.from_dict({"device": "cpu", "data": {"dataset_name": dataset,
                                                             "root": str(data_root)}})
    targets = {str(r) for r in _resolve_target_relation_ids(
        _LOCAL_HGB_LINK_DATASETS[dataset], tuple(config.data.target_relation))}
    rows = [r for r in content.decode("utf-8").splitlines(keepends=True)
            if r.split("\t")[2] in targets]
    summary = {**summary, "mode": "target", "target_relation_ids": sorted(int(t) for t in targets),
               "carve_md5": hashlib.md5(content).hexdigest(),
               "rows_per_relation": {f"rel_{t}": summary["rows_per_relation"].get(f"rel_{t}", 0)
                                     for t in sorted(targets)}}
    return "".join(rows).encode("utf-8"), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=REPO_DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["pubmed", "lastfm"],
                        choices=["pubmed", "lastfm"])
    parser.add_argument("--mode", choices=["target", "carve"], default="target")
    args = parser.parse_args()
    for dataset in args.datasets:
        carve, summary = carve_rows(args.data_root, dataset)
        target, target_summary = target_rows(args.data_root, dataset)
        content = target if args.mode == "target" else carve
        if args.mode == "target":
            summary = target_summary
        path = lp_root(args.data_root) / summary["folder"] / "link.dat.valid"
        if path.exists() and path.read_bytes() not in (carve, target):
            raise FileExistsError(f"{path} is neither the carve nor the target-only file; "
                                  "not overwritten.")
        if path.exists():
            summary["previous_md5"] = hashlib.md5(path.read_bytes()).hexdigest()
        path.write_bytes(content)
        summary["md5"] = hashlib.md5(content).hexdigest()
        summary["path"] = str(path.relative_to(args.data_root.parent))
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
