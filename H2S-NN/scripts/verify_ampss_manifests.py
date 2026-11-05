"""Verify that AMPSS ablations 3--5 share one frozen Stage-I manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ampss_ablation_protocol import assert_frozen_stage1_manifests


def main(items: list[tuple[int, Path]]) -> None:
    manifests = []
    for index, path in items:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["ablation_index"] = index
        manifests.append(payload)
    assert_frozen_stage1_manifests(manifests)
    print(f"Verified {len(manifests)} frozen AMPSS Stage-I manifests.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        nargs=2,
        action="append",
        metavar=("ABLATION_INDEX", "PATH"),
        required=True,
    )
    parsed = parser.parse_args()
    main([(int(index), Path(path)) for index, path in parsed.manifest])
