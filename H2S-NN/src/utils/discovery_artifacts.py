from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.selector_reporting import stable_hash
from src.data.stage1_selector import SelectorReport


def persist_selector_artifacts(
    exp_dir: Path,
    report: SelectorReport | None,
) -> dict[str, Any] | None:
    if report is None:
        return None
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "selector_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
    manifest = {
        "protocol_version": report.protocol_version,
        "split_seed": report.split_seed,
        "sample_seed": report.sample_seed,
        "selector_unit_hash": report.selector_unit_hash,
        "selector_support_hash": report.selector_support_hash,
        "final_materialization_hash": report.materialized_face_hash,
        "candidate_identities": [
            item.identity for item in report.candidate_statistics
        ],
        "selected_identities": list(report.selected_identities),
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    with (exp_dir / "selector_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)
    (exp_dir / "selector_manifest_hash.txt").write_text(
        str(manifest["manifest_hash"]) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_selector_manifest(exp_dir: Path) -> dict[str, Any]:
    path = exp_dir / "selector_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    stored_hash = str(manifest.get("manifest_hash", ""))
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    computed_hash = stable_hash(payload)
    if not stored_hash or stored_hash != computed_hash:
        raise ValueError("AMPSS selector manifest hash is missing or invalid.")
    return manifest
