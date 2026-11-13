"""Compare the installed packages with `requirements/linux-lock.txt` (cluster-setup CS2).

    python scripts/setup/check_env.py [--lock requirements/linux-lock.txt]

Prints one line per difference (missing package, different version) and exits 1 if there is
any. An installed version that equals the pin plus a local build tag the pin does not state
(`2.6.0+cu124` for `torch==2.6.0`) is printed as a note and is not a difference; a pin that
states a local tag (`dgl==2.5.0+cu124`) must match exactly. Nothing is installed or changed.
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import NamedTuple

DEFAULT_LOCK = Path(__file__).resolve().parents[2] / "requirements" / "linux-lock.txt"


class Finding(NamedTuple):
    name: str
    kind: str  # "missing" | "different" | "local-tag" (note only)
    locked: str
    installed: str | None

    @property
    def is_difference(self) -> bool:
        return self.kind != "local-tag"

    def describe(self) -> str:
        if self.kind == "missing":
            return f"MISSING    {self.name}: locked {self.locked}, not installed"
        if self.kind == "different":
            return f"DIFFERENT  {self.name}: locked {self.locked}, installed {self.installed}"
        return f"note       {self.name}: locked {self.locked}, installed {self.installed} (local build tag)"


def normalize_name(name: str) -> str:
    """PEP 503 normalisation: `torch_geometric`, `Torch-Geometric` -> `torch-geometric`."""

    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(text: str) -> dict[str, str]:
    """`name==version` pins of a requirements-style lock; options, comments and markers dropped."""

    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].split(";", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.\-]*)\s*==\s*(\S+)", line)
        if match is None:
            raise ValueError(f"lock line is not an exact `name==version` pin: {raw!r}")
        name = normalize_name(match.group(1))
        if name in pins:
            raise ValueError(f"package pinned twice in the lock: {name}")
        pins[name] = match.group(2)
    return pins


def compare(locked: dict[str, str], installed: dict[str, str]) -> list[Finding]:
    """Every locked package that is missing, different, or differs only by an extra local tag."""

    installed_norm = {normalize_name(k): v for k, v in installed.items()}
    findings: list[Finding] = []
    for name, want in sorted(locked.items()):
        have = installed_norm.get(name)
        if have is None:
            findings.append(Finding(name, "missing", want, None))
        elif have == want:
            continue
        elif "+" not in want and have.split("+", 1)[0] == want:
            findings.append(Finding(name, "local-tag", want, have))
        else:
            findings.append(Finding(name, "different", want, have))
    return findings


def installed_versions() -> dict[str, str]:
    return {normalize_name(d.metadata["Name"]): d.version for d in metadata.distributions()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args(argv)
    findings = compare(parse_lock(args.lock.read_text(encoding="utf-8")), installed_versions())
    for finding in findings:
        print(finding.describe())
    n_diff = sum(f.is_difference for f in findings)
    print(f"python {sys.version.split()[0]}; lock {args.lock}; differences: {n_diff}; "
          f"notes: {len(findings) - n_diff}")
    return 1 if n_diff else 0


if __name__ == "__main__":
    sys.exit(main())
