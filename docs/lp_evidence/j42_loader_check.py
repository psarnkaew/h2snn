"""J4.2 (Part J): the loader with the fixed validation files, on the real tasks.

1. PubMed and LastFM: `build_link_prediction_dataset` with `link.dat.valid` equals the load from a
   scratch copy of the release without the file (today's carve), tensor for tensor.
2. Derived tasks: the evaluated split is now J1.2's 81/9/10 (validation = link.dat.valid), every
   existing leak check passes, and validation facts are absent from the message graph in both
   orientations.
Run from H2S-NN/: python ../docs/lp_evidence/j42_loader_check.py <scratch dir>"""
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from src.configs.config import TrainConfig  # noqa: E402
from src.data.link_prediction_dataset import build_link_prediction_dataset  # noqa: E402

FIELDS = ("train_pos_edge_index", "val_pos_edge_index", "val_neg_edge_index",
          "test_pos_edge_index", "test_neg_edge_index", "known_pos_edge_index", "edge_index",
          "edge_type")
DERIVED = {"acm_pa": ("paper", "rel_2", "author"), "dblp_ap": ("author", "rel_0", "paper"),
           "imdb_ma": ("movie", "rel_2", "actor")}


def config(root: Path, dataset: str, target=None) -> TrainConfig:
    data = {"dataset_name": dataset, "root": str(root), "fact_layer": "tied",
            "path_sampler": "walk", "face_sample_k": 4}
    if target is not None:
        data.update(task_type="link_prediction", target_relation=list(target))
    return TrainConfig.from_dict({"device": "cpu", "seed": 3, "data": data})


def main(scratch: Path) -> None:
    root = Path("data")
    for name, folder in (("pubmed", "PubMed"), ("lastfm", "LastFM")):
        t0 = time.time()
        copy = scratch / folder
        if copy.exists():
            shutil.rmtree(copy)
        copy.mkdir(parents=True)
        for f in ("node.dat", "link.dat", "link.dat.test"):
            shutil.copyfile(root / folder / f, copy / f)
        with_file = build_link_prediction_dataset(config(root, name))
        carve = build_link_prediction_dataset(config(scratch, name))
        same = {f: bool(torch.equal(getattr(with_file, f), getattr(carve, f))) for f in FIELDS}
        print(f"{name}: with link.dat.valid == without (carve): {all(same.values())} {same}; "
              f"train/val/test pos {with_file.train_pos_edge_index.size(1)}/"
              f"{with_file.val_pos_edge_index.size(1)}/{with_file.test_pos_edge_index.size(1)} "
              f"({time.time() - t0:.0f} s)", flush=True)
    for name, target in DERIVED.items():
        t0 = time.time()
        data = build_link_prediction_dataset(config(root, name, target))
        n = (data.train_pos_edge_index.size(1), data.val_pos_edge_index.size(1),
             data.test_pos_edge_index.size(1))
        total = sum(n)
        val = {tuple(p) for p in data.val_pos_edge_index.t().tolist()}
        message = {tuple(p) for p in data.edge_index.t().tolist()}
        print(f"{name}: loaded with every leak check passing; train/val/test {n[0]}/{n[1]}/{n[2]} "
              f"= {100 * n[0] / total:.2f}/{100 * n[1] / total:.2f}/{100 * n[2] / total:.2f} %; "
              f"validation facts in the message graph: {len(val & message)} as stored, "
              f"{len({(v, u) for u, v in val} & message)} reversed; message-graph edges "
              f"{data.edge_index.size(1)} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
