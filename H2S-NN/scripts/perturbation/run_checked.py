"""Card 7 amendment 3 (author 2026-11-19): every perturbation run checks the graph its loader actually built.

Runs a launcher in this process (`runpy`, its own `__main__`, its own argv) after wrapping H2S-NN's loader
`src.data.dataset.build_dataset` (the name `src.builders` calls for H2S-NN runs; `src.baselines.bundles.build_bundle`
imports it at call time for every baseline bundle). At every load:
1. the configured data root, resolved as the loader resolves it (relative to the working directory: the baselines'
   default `data` from the graph's folder), must be the graph's data root `<graph folder>/data`;
2. the SHA-256 of that root's `link.dat` must be the generator's `link_sha256` (the file the vendor loaders, e.g.
   HAN's, read under the same root);
3. the fingerprint of the loaded edge list (`fingerprint.from_prepared`) must be the expected one (the manifest's
   `fingerprint` column = the generator's `edge_fingerprint` in `graph.json`).
Each check is appended to `--log` (JSON lines). Check 1 runs before loading. A mismatch raises
`FingerprintMismatch`: the run fails (exit status 1) before it trains. A launcher that never loads through the
wrapped name also fails (zero checks).

    python scripts/perturbation/run_checked.py --graph_json <graph.json> --expected_fingerprint <sha256> \
        --log <jsonl> -- <launcher.py> <launcher args ...>
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fingerprint as fp  # noqa: E402


class FingerprintMismatch(RuntimeError):
    """The loaded graph is not the generator's graph."""


def _link_sha256(root: Path, dataset: str) -> str:
    from src.data.data_layout import hgb_nc_raw_dir

    return hashlib.sha256((Path(hgb_nc_raw_dir(root, dataset)) / "link.dat").read_bytes()).hexdigest()


@contextlib.contextmanager
def checked_loading(graph_json: Path, expected: str, log: Path) -> Iterator[list[dict[str, Any]]]:
    """Wraps the loader for the duration of the block; yields the list of checks made."""
    import src.builders as builders
    import src.data.dataset as dataset_module

    graph = json.loads(Path(graph_json).read_text(encoding="utf-8"))
    graph_root = (Path(graph_json).resolve().parent / "data").resolve()
    original = dataset_module.build_dataset
    checks: list[dict[str, Any]] = []
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)

    def record(entry: dict[str, Any]) -> None:
        checks.append(entry)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def checked_build_dataset(config):
        root = Path(config.data.root).resolve()
        entry: dict[str, Any] = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "cwd": os.getcwd(),
                                 "dataset": config.data.dataset_name, "seed": getattr(config, "seed", None),
                                 "data_root": str(root), "graph_root": str(graph_root), "expected": expected}
        if root != graph_root or config.data.dataset_name.lower() != graph["dataset"]:
            entry.update(match=False, failed="data_root")
            record(entry)
            raise FingerprintMismatch(f"loader reads {config.data.dataset_name} at {root}, the graph is "
                                      f"{graph['dataset']} at {graph_root}")
        entry["link_sha256"] = _link_sha256(root, graph["dataset"])
        if entry["link_sha256"] != graph["link_sha256"]:
            entry.update(match=False, failed="link_sha256")
            record(entry)
            raise FingerprintMismatch(f"{root}: link.dat differs from the generator's ({graph['link_sha256']})")
        data = original(config)
        loaded = fp.from_prepared(data)
        entry.update(fingerprint=loaded["sha256"], pairs=loaded["pairs"], match=loaded["sha256"] == expected)
        if not entry["match"]:
            entry["failed"] = "fingerprint"
        record(entry)
        if not entry["match"]:
            raise FingerprintMismatch(f"loaded edge list {loaded['sha256']} != expected {expected} ({root})")
        return data

    dataset_module.build_dataset = checked_build_dataset
    builders.build_dataset = checked_build_dataset
    try:
        yield checks
    finally:
        dataset_module.build_dataset = original
        builders.build_dataset = original


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: run_checked.py --graph_json G --expected_fingerprint F --log L -- launcher.py ...")
    split = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_json", type=Path, required=True)
    parser.add_argument("--expected_fingerprint", required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args(argv[:split])
    launcher, launcher_args = argv[split + 1], argv[split + 2:]
    with checked_loading(args.graph_json, args.expected_fingerprint, args.log) as checks:
        saved = sys.argv
        sys.argv = [launcher, *launcher_args]
        try:
            runpy.run_path(launcher, run_name="__main__")
        finally:
            sys.argv = saved
    if not checks:
        raise FingerprintMismatch(f"{launcher} never loaded a graph through build_dataset: nothing was checked")
    print(f"fingerprint checks: {len(checks)}, all match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
