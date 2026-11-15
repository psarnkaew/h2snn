"""Part M3b B2: merge the per-dataset count records into docs/partM3b_evidence/h2gb_nc_counts.{json,md}.

    python scripts/data/h2gb_nc_counts_table.py --evidence ../docs/partM3b_evidence --jobs <json>

Compares each dataset with H2GB Table 1 (arXiv 2407.10916v2, "Statistics of H2GB datasets", read
2026-11-15 from the arXiv HTML page): #nodes (types), #edges (types), #feat, #classes. A dataset is
"verified" when nodes, node types, edges and edge types of the loaded graph (H2GB's `dataset[0]`) equal
Table 1; a difference in the number of classes or the feature dimension is reported, not counted: both are
read from H2GB's own processed file (the card's rule: a count that differs from the paper while equal to
H2GB's code or released files is reported, not a stop). `--jobs` maps key -> {job, node, peak_rss_gib,
elapsed} from sacct. Reads and writes small evidence files only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# H2GB Table 1 (arXiv 2407.10916v2): nodes, node types, edges, edge types, feature dim, classes
TABLE1: dict[str, tuple[int, int, int, int, int, int]] = {
    "ogbn-mag": (1_939_743, 4, 42_182_144, 7, 128, 349),
    "mag-year": (1_939_743, 4, 42_182_144, 7, 128, 5),
    "oag-cs": (1_112_691, 4, 27_537_448, 22, 768, 3_514),
    "oag-eng": (929_315, 4, 12_346_854, 22, 768, 3_956),
    "oag-chem": (1_918_881, 4, 38_098_014, 22, 768, 2_985),
    "ieee-cis-g": (153_880, 12, 2_873_472, 22, 4_823, 2),
    "h-pokec": (1_731_977, 16, 51_774_836, 31, 66, 2),
    "pdns": (1_173_558, 2, 76_797_104, 4, 10, 2),
}
KEYS = ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec")


def _target_feat(record: dict) -> int | None:
    dims = record["loaded"]["feature_dims"].get(record["target_node_type"])
    return dims[0] if dims else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--jobs", default="{}")
    args = parser.parse_args()
    evidence = Path(args.evidence)
    jobs = json.loads(args.jobs)
    records: dict = {}
    for key in KEYS:
        path = evidence / f"b2_counts_{key}.json"
        if path.is_file():
            records[key] = json.loads(path.read_text())[key]
    compare = evidence / "b2_counts_compare-mag.json"
    merged: dict = {"source_table": "H2GB Table 1, arXiv 2407.10916v2 (HTML), read 2026-11-15",
                    "datasets": records, "jobs": jobs, "comparison_with_table1": {}, "verified": []}
    if compare.is_file():
        merged["mag_vs_mag_year"] = json.loads(compare.read_text())["mag_vs_mag_year"]
    lines = ["# H2GB NC datasets as H2GB loads them (Part M3b B2)", "",
             "Loaded through `src/data/h2gb_nc.py` (H2GB's own `preformat_*`, `dataset[0]`), by "
             "`scripts/data/h2gb_nc_counts.py` in SLURM jobs; merged by `scripts/data/h2gb_nc_counts_table.py`. "
             "Table 1 = H2GB, arXiv 2407.10916v2, Table 1 (read from the arXiv HTML page, 2026-11-15). "
             "`stored` = the processed file; `loaded` = after H2GB's transform.", "",
             "| Dataset | nodes (types) loaded | Table 1 | edges (types) stored | edges (types) loaded | Table 1 | "
             "target feat | Table 1 | classes | Table 1 | train / val / test | load s | peak RSS GiB (sacct) | job, node |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, rec in records.items():
        nodes, ntypes, edges, etypes, feat, classes = TABLE1[key]
        loaded, stored, labels = rec["loaded"], rec["stored"], rec["labels"]
        got_classes = labels["num_classes_(max_y+1)"]
        checks = {
            "nodes": loaded["total_nodes"] == nodes, "node_types": loaded["num_node_types"] == ntypes,
            "edges": loaded["total_edges"] == edges, "edge_types": loaded["num_edge_types"] == etypes,
            "classes": got_classes == classes, "target_feature_dim": _target_feat(rec) == feat}
        merged["comparison_with_table1"][key] = checks
        if all(v for k, v in checks.items() if k not in ("target_feature_dim", "classes")):
            merged["verified"].append(key)
        split = " / ".join(str(labels["splits"][s]["size"]) for s in ("train", "val", "test"))
        job = jobs.get(key, {})
        lines.append(
            f"| {key} | {loaded['total_nodes']:,} ({loaded['num_node_types']}) | {nodes:,} ({ntypes}) | "
            f"{stored['total_edges']:,} ({stored['num_edge_types']}) | {loaded['total_edges']:,} "
            f"({loaded['num_edge_types']}) | {edges:,} ({etypes}) | {_target_feat(rec)} | {feat} | "
            f"{got_classes} | {classes} | {split} | {rec['load_seconds']} | {job.get('peak', '?')} | "
            f"{job.get('job', '?')}, {job.get('node', '?')} |")
    lines += ["", "Equal to Table 1 (nodes, node types, edges and edge types of the loaded graph): "
              + ", ".join(merged["verified"]) + ".",
              "Reported differences from the paper (equal to H2GB's processed files, not a stop): "
              + "; ".join(f"{k}: {', '.join(n for n, ok in c.items() if not ok)}"
                          for k, c in merged["comparison_with_table1"].items() if not all(c.values()))
              + ".", "",
              "## Per dataset", ""]
    for key, rec in records.items():
        lines.append(f"### {key}")
        lines.append(f"- H2GB: `{rec['h2gb_class']}` via `{rec['h2gb_loader']}` (name {rec['h2gb_name']!r}); "
                     f"transform `{rec['transform']}`; target `{rec['target_node_type']}`, label "
                     f"{rec['label']}; split: {rec['split_source']}.")
        lines.append(f"- processed `{rec['processed_file']}`: {rec['processed_bytes']:,} bytes, sha256 "
                     f"`{rec['processed_sha256']}`; folder: {rec.get('processed_dir_files')}.")
        lines.append(f"- nodes: {rec['loaded']['nodes']}")
        lines.append(f"- feature dims: {rec['loaded']['feature_dims']}")
        lines.append(f"- edges stored: {rec['stored']['edges']}")
        lines.append(f"- edges loaded: {rec['loaded']['edges']}")
        labels = rec["labels"]
        lines.append(f"- labels: y {labels['y_shape']} {labels['y_dtype']}, labelled {labels['labelled']:,}, "
                     f"unlabelled (y = -1) {labels['unlabelled_(y=-1)']:,}, classes (max y + 1) "
                     f"{labels['num_classes_(max_y+1)']}, present {labels['classes_present']}; split overlaps "
                     f"{labels['split_overlaps']}.")
        for split in ("train", "val", "test"):
            info = labels["splits"][split]
            lines.append(f"  - {split}: {info['size']:,} nodes, unlabelled in mask {info['unlabelled_in_mask']}, "
                         f"classes present {info['classes_present']}, class counts {info['class_counts']}")
        lines.append(f"- Table 1 comparison: {merged['comparison_with_table1'][key]}")
        lines.append("")
    if "mag_vs_mag_year" in merged:
        lines += ["## ogbn-mag vs mag-year (`--compare-mag`, stored graphs)", "",
                  "```", json.dumps(merged["mag_vs_mag_year"], indent=1), "```", ""]
    (evidence / "h2gb_nc_counts.json").write_text(json.dumps(merged, indent=1) + "\n")
    (evidence / "h2gb_nc_counts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
