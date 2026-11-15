"""Part M3b B9 item 7: the provenance section of the H2GB datasets for `data/PROVENANCE.md`.

    python scripts/data/h2gb_provenance.py --root data --counts ../docs/partM3b_evidence/h2gb_nc_counts.json \
        --out ../docs/partM3b_evidence/h2gb_provenance_section.md

Reads file sizes and sha256 (processed files: from the B2 count record; raw files: computed here) and writes the
markdown section only (the caller appends it to `data/PROVENANCE.md`; nothing under `data/` is written here).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


RAW = {"IEEE-CIS-G": "NC/IEEE-CIS/raw", "H-Pokec": "NC/Pokec/raw", "PDNS": "NC/PDNS/raw"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    counts = json.loads(Path(args.counts).read_text())["datasets"]
    day = time.strftime("%Y-%m-%d")
    lines = [f"## H2GB node-classification datasets (Part M3b, {day})", "",
             "Source: the H2GB benchmark (Lin et al., arXiv 2407.10916; code `github.com/junhongmit/H2GB` at commit "
             "`d3b49db63ad66f8a3282dbe699d50ce10ed3062a`, cloned by `scripts/setup/clone_vendors.sh`). The author "
             "downloaded and processed them on the lab cluster with H2GB's own classes "
             "(`scripts/data/h2gb_use_dataset.py`, formerly `vendor/H2GB/examples/use_dataset.py`), with the vendor "
             "patch `vendor/patches/H2GB.patch` (IEEE-CIS test identity columns `id-NN` → `id_NN`; Pokec download "
             "without `extract_zip`). Loaded offline by `src/data/h2gb_nc.py` (H2GB's own `preformat_*`); counts "
             "and checks: `docs/partM3b_evidence/h2gb_nc_counts.md`, `oag_identity.md`, `leak_audit.md`, "
             "`pdns_protocol.md`. ogbn-mag/mag-year also need OGB's own `mag/ogbn_mag` (H2GB builds them from it).", ""]
    for key, rec in counts.items():
        lines.append(f"- `{key}` — H2GB `{rec['h2gb_class']}` via `{rec['h2gb_loader']}` (name "
                     f"{rec['h2gb_name']!r}); processed `{rec['processed_file']}`, {rec['processed_bytes']:,} bytes, "
                     f"sha256 `{rec['processed_sha256']}`.")
        if key == "oag-eng":
            lines.append("  Verified identity: the folder `OAG/chemistry` holds the **engineering** field "
                         "(H2GB's `oag_dataset.py:55-57` swaps the hub files; Table 1 counts match).")
        if key == "oag-chem":
            lines.append("  Verified identity: the folder `OAG/engineering` holds the **chemistry** field.")
    lines.append("- `OAG/raw` is empty (H2GB processed the hub files elsewhere; not needed offline).")
    for name, folder in RAW.items():
        lines.append(f"- {name} raw (`{folder}`):")
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                lines.append(f"  - `{path.relative_to(root / folder)}` {path.stat().st_size:,} bytes, sha256 "
                             f"`{_sha256(path)}`")
    lines.append("- PDNS has no processed file: H2GB's class reads `raw/` on every load (`pdns_protocol.md`).")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} lines")


if __name__ == "__main__":
    main()
