"""A tiny release in HGB Freebase's raw format (Part P P3): `info.dat` with the TYPE and LINK text
sections PyG's `HGBDataset` parses for Freebase (`hgb_dataset.py:114-131`), `node.dat` without
attributes, `link.dat` over HGB Freebase's 36 relations (link ids and names as the real release,
`HGB_RELATIONS["freebase"]`), `label.dat` / `label.dat.test` on books (7 classes), and books left
unlabelled. Every listed meta-path of `LITERATURE_METAPATHS["freebase"]` has instances. No real data
are read.
"""

from __future__ import annotations

import random
from pathlib import Path

TYPES = ("book", "film", "music", "sports", "people", "location", "organization", "business")
COUNTS = {"book": 40, "film": 12, "music": 12, "sports": 6, "people": 12, "location": 8,
          "organization": 6, "business": 6}
NUM_CLASSES = 7


def write_freebase_release(root: Path, seed: int = 0, edges_per_relation: int = 30) -> Path:
    """Write the release under `<root>/NC/freebase/raw/Freebase` and return that directory."""
    from src.baselines.adapters._h2gformer_encoders import HGB_RELATIONS
    from src.data.data_layout import hgb_nc_raw_dir

    rng = random.Random(seed)
    folder = hgb_nc_raw_dir(root, "freebase")
    folder.mkdir(parents=True, exist_ok=True)
    ids: dict[str, list[int]] = {}
    nodes, next_id = [], 0
    for type_id, name in enumerate(TYPES):
        ids[name] = list(range(next_id, next_id + COUNTS[name]))
        for node in ids[name]:
            nodes.append(f"{node}\t{name}{node}\t{type_id}\n")
        next_id += COUNTS[name]
    rng.shuffle(nodes)  # HGB's node.dat is not sorted by type
    (folder / "node.dat").write_text("".join(nodes), encoding="utf-8")

    info = ["node.dat", "", "TYPE\tMEANING"]
    info += [f"{i}\t\t{name.upper()}" for i, name in enumerate(TYPES)]
    info += ["", "Attribute Dimension: 0", "", "", "link.dat", "", "LINK\tSTART\tEND\tMEANING"]
    links = []
    for link_id, relation in enumerate(HGB_RELATIONS["freebase"]):
        source, word, destination = relation.split("__")
        info.append(f"{link_id}\t\t{TYPES.index(source)}\t\t{TYPES.index(destination)}\t"
                    f"{source.upper()}-{word}-{destination.upper()}")
        pairs = set()
        while len(pairs) < edges_per_relation:
            u, v = rng.choice(ids[source]), rng.choice(ids[destination])
            if u != v:
                pairs.add((u, v))
        links += [f"{u}\t{v}\t{link_id}\t1.0\n" for u, v in sorted(pairs)]
    info += ["", "", "label.dat", ""]
    (folder / "info.dat").write_text("\n".join(info) + "\n", encoding="utf-8")
    (folder / "link.dat").write_text("".join(links), encoding="utf-8")

    books = list(ids["book"])
    rng.shuffle(books)
    labelled = books[: len(books) * 3 // 4]  # the rest stay unlabelled
    train, test = labelled[: len(labelled) * 3 // 10 + 4], labelled[len(labelled) * 3 // 10 + 4:]
    label = {book: i % NUM_CLASSES for i, book in enumerate(sorted(labelled))}
    (folder / "label.dat").write_text(
        "".join(f"{b}\tbook{b}\t0\t{label[b]}\n" for b in train), encoding="utf-8")
    (folder / "label.dat.test").write_text(
        "".join(f"{b}\tbook{b}\t0\t{label[b]}\n" for b in test), encoding="utf-8")
    return folder


__all__ = ["COUNTS", "NUM_CLASSES", "TYPES", "write_freebase_release"]
