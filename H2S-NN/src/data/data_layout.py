"""Where each task's data lives under the data root (card Part M, M1).

The author's layout (2026-11-12): ``<root>/NC`` node classification (HGB releases as PyG's
``HGBDataset`` stores them: ``NC/<name>/raw/<NAME>/*.dat``), ``<root>/LP`` link prediction
(``LP/PubMed``, ``LP/LastFM``, ``LP/lp_derived/<task>``), ``<root>/GC`` graph classification
(PyG ``TUDataset`` folders: ``GC/<NAME>/raw``). ``config.data.root`` names ``<root>``; this module
is the only place that knows the three sub-folders. No compatibility path for the old layout.
"""

from __future__ import annotations

from pathlib import Path

# The repository's data root (`H2S-NN/data`), for scripts and adapters that run without a config.
REPO_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

NC_DIR = "NC"
LP_DIR = "LP"
GC_DIR = "GC"
# Derived link-prediction tasks (spec v1.7 §6): `LP/lp_derived/<task>`.
LP_DERIVED_DIR = "lp_derived"

# HGB node-classification releases by key (PyG's HGBDataset names).
HGB_NC_NAMES: dict[str, str] = {
    "acm": "ACM",
    "dblp": "DBLP",
    "imdb": "IMDB",
    "freebase": "Freebase",
}


# H2GB node-classification datasets (Part M3b B2): the root each H2GB dataset class receives, relative to
# `<root>/NC`, exactly as H2GB's `load_dataset_master` forms it (`H2GB/loader/master_loader.py:165-168, 228`:
# `PyG-ogbn-mag`/`PyG-mag-year` -> `mag`, `PyG-<id>` -> `<id>`, `IEEE-CIS` -> `IEEE-CIS`), and the processed
# file the class loads under it (`<class>.processed_dir`/`data.pt`). The OAG folder of each OAG key follows the
# identity verified in B3 (`docs/partM3b_evidence/oag_identity.md`); PDNS has no processed file (B5).
H2GB_NC_DIRS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ogbn-mag": ("mag", ("ogbn-mag", "processed", "data.pt")),
    "mag-year": ("mag", ("mag-year", "processed", "data.pt")),
    "oag-cs": ("OAG", ("cs", "processed", "data.pt")),
    # B3 (verified identity, docs/partM3b_evidence/oag_identity.md): H2GB's name `engineering` loads
    # OAG_Chemistry.pt and `chemistry` loads OAG_Engineering.pt (oag_dataset.py:55-57), so the folder
    # `engineering` holds the chemistry field (Table 1's oag-chem counts) and `chemistry` the engineering field.
    "oag-eng": ("OAG", ("chemistry", "processed", "data.pt")),
    "oag-chem": ("OAG", ("engineering", "processed", "data.pt")),
    "ieee-cis-g": ("IEEE-CIS", ("processed", "data.pt")),
    "h-pokec": ("Pokec", ("processed", "data.pt")),
}


def h2gb_nc_dataset_dir(root: Path, key: str) -> Path:
    """The root H2GB's dataset class of `key` receives (`dataset_dir` of `load_dataset_master`)."""
    return nc_root(root) / H2GB_NC_DIRS[key][0]


def h2gb_nc_processed_file(root: Path, key: str) -> Path:
    """The processed file H2GB's dataset class of `key` loads; it is never written by this project."""
    return h2gb_nc_dataset_dir(root, key).joinpath(*H2GB_NC_DIRS[key][1])


def nc_root(root: Path) -> Path:
    """Root passed to PyG's ``HGBDataset`` (it appends ``<name>/raw``)."""
    return Path(root) / NC_DIR


def lp_root(root: Path) -> Path:
    """Folder of the link-prediction tasks (HGB LP releases and ``lp_derived``)."""
    return Path(root) / LP_DIR


def lp_derived_root(root: Path) -> Path:
    return lp_root(root) / LP_DERIVED_DIR


def gc_root(root: Path) -> Path:
    """Root passed to PyG's ``TUDataset`` (it appends ``<NAME>/raw``)."""
    return Path(root) / GC_DIR


def hgb_nc_raw_dir(root: Path, key: str) -> Path:
    """Folder holding ``node.dat``/``link.dat``/``label.dat`` of one HGB NC release."""
    name = HGB_NC_NAMES[key]
    return nc_root(root) / key / "raw" / name


def tu_raw_dir(root: Path, tud_name: str) -> Path:
    return gc_root(root) / tud_name / "raw"


__all__ = ["GC_DIR", "H2GB_NC_DIRS", "HGB_NC_NAMES", "h2gb_nc_dataset_dir", "h2gb_nc_processed_file", "LP_DERIVED_DIR", "LP_DIR", "NC_DIR", "REPO_DATA_ROOT", "gc_root",
           "hgb_nc_raw_dir", "lp_derived_root", "lp_root", "nc_root", "tu_raw_dir"]
