"""J1.1 (Part J): relation inventory of the HGB node-classification releases for link prediction.

The files are read with the link-prediction loader's own readers, `_load_local_hgb_nodes` and
`_read_local_hgb_link_file` (`src/data/link_prediction_dataset.py:187,245`). That confirms the
format the derived tasks will be loaded with. `rho` and the facts after tying come from the fact
layer (`src/data/fact_layer.py`: `inverse_relations_typed`, `build_fact_layer`, `fact_mask`).
Run from H2S-NN/: python ../docs/lp_evidence/j11_inventory.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from src.data.fact_layer import build_fact_layer, fact_mask, inverse_relations_typed  # noqa: E402
from src.data.link_prediction_dataset import (  # noqa: E402
    _load_local_hgb_nodes, _read_local_hgb_link_file)

OUT = Path("../docs/lp_evidence")
RAW = {
    "acm": Path("data/acm/acm/raw/ACM"),
    "dblp": Path("data/dblp/dblp/raw/DBLP"),
    "imdb": Path("data/imdb/imdb/raw/IMDB"),
    "freebase": Path("data/freebase/freebase/raw/Freebase"),
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def info(dataset: str, folder: Path) -> tuple[dict[int, str], dict[int, str]]:
    """Node-type names and relation meanings from info.dat (JSON for ACM/DBLP/IMDB, a text table
    for Freebase, parsed as PyG's HGBDataset does)."""
    text = (folder / "info.dat").read_text(encoding="utf-8")
    if dataset == "freebase":
        lines = text.split("\n")
        start = lines.index("TYPE\tMEANING") + 1
        end = lines[start:].index("")
        types = {int(k): v.lower() for k, v in (row.split("\t\t") for row in lines[start:start + end])}
        start = lines.index("LINK\tSTART\tEND\tMEANING") + 1
        end = lines[start:].index("")
        meanings = {}
        for key, row in enumerate(lines[start:start + end]):
            fields = [v for v in row.split("\t")[1:] if v != ""]
            meanings[key] = fields[2]
        return types, meanings
    data = json.loads(text)
    types = {int(k): v for k, v in data["node.dat"]["node type"].items()}
    meanings = {int(k): v["meaning"] for k, v in data["link.dat"]["link type"].items()}
    return types, meanings


def main() -> None:
    report = {}
    for dataset, folder in RAW.items():
        t0 = time.time()
        type_names, meanings = info(dataset, folder)
        rows_ok = all(len(line.rstrip("\n").split("\t")) == 4
                      for line in (folder / "link.dat").open(encoding="utf-8"))
        nodes = _load_local_hgb_nodes(folder, type_names)
        links = _read_local_hgb_link_file(folder / "link.dat", nodes)
        type_sizes = {name: int(nodes.data[name].num_nodes) for name in nodes.data.node_types}
        rho = inverse_relations_typed(links, type_sizes)

        def rel_id(relation):
            return int(relation[1].removeprefix("rel_"))

        relations = sorted(links, key=rel_id)
        offsets, total = {}, 0
        for name in sorted(type_sizes):
            offsets[name] = total
            total += type_sizes[name]
        parts, types = [], []
        for index, relation in enumerate(relations):
            local = links[relation]
            parts.append(torch.stack([local[0] + offsets[relation[0]], local[1] + offsets[relation[2]]]))
            types.append(torch.full((local.size(1),), index, dtype=torch.long))
        edge_index, edge_type = torch.cat(parts, dim=1), torch.cat(types)
        layer = build_fact_layer(edge_index, edge_type, [r[1] for r in relations], total, "tied")
        kept_mask = fact_mask(layer, edge_index, edge_type)
        rows = []
        for index, relation in enumerate(relations):
            local = links[relation]
            partner = rho[relation]
            rows.append({
                "relation_id": rel_id(relation),
                "meaning": meanings[rel_id(relation)],
                "src_type": relation[0], "dst_type": relation[2],
                "stored_distinct_edges": int(local.size(1)),
                "self_loops": int((local[0] == local[1]).sum()) if relation[0] == relation[2] else 0,
                "rho": None if partner is None else rel_id(partner),
                "facts_after_tying": int(kept_mask[edge_type == index].sum()),
            })
        report[dataset] = {
            "folder": str(folder).replace("\\", "/"),
            "files_md5": {name: md5(folder / name) for name in
                          ("info.dat", "node.dat", "link.dat", "label.dat", "label.dat.test")},
            "every_link_row_has_4_columns": rows_ok,
            "node_types": {name: type_sizes[name] for name in nodes.data.node_types},
            "feature_dims": {name: (int(nodes.data[name].x.size(1)) if "x" in nodes.data[name] else 0)
                             for name in nodes.data.node_types},
            "relations": rows,
            "facts_before": layer.facts_before, "facts_after": layer.facts_after,
            "seconds": round(time.time() - t0, 1),
        }
        print(dataset, report[dataset]["facts_before"], "->", report[dataset]["facts_after"],
              f"({report[dataset]['seconds']} s)", flush=True)
    (OUT / "inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
