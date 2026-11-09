"""J1.3 (Part J): load smoke, leak checks and the seed question on the real derived tasks (plus
PubMed and LastFM for the seed question), and the three leak mutations on a copy of acm_pa.
Run from H2S-NN/: python ../docs/lp_evidence/j13_loader_check.py <scratch dir>"""
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from scripts.lp import build_derived_lp as bd  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402
from src.data.link_prediction_dataset import build_link_prediction_dataset  # noqa: E402

SPLITS = ("train_pos_edge_index", "val_pos_edge_index", "val_neg_edge_index",
          "test_pos_edge_index", "test_neg_edge_index")
DERIVED = {"acm_pa": ("paper", "rel_2", "author"), "dblp_ap": ("author", "rel_0", "paper"),
           "imdb_ma": ("movie", "rel_2", "actor")}


def config(root: Path, dataset: str, target, seed: int) -> TrainConfig:
    data = {"dataset_name": dataset, "root": str(root), "fact_layer": "tied",
            "path_sampler": "walk", "face_sample_k": 4}
    if target is not None:
        data.update(task_type="link_prediction", target_relation=list(target))
    return TrainConfig.from_dict({"device": "cpu", "seed": seed, "data": data})


def pairs(index: torch.Tensor, src0: int, dst0: int) -> set:
    return {(u - src0, v - dst0) for u, v in index.t().tolist()}


def main(scratch: Path) -> None:
    root = Path("data")
    for name, target in [*DERIVED.items(), ("lastfm", None), ("pubmed", None)]:
        t0 = time.time()
        a = build_link_prediction_dataset(config(root, name, target, seed=3))
        b = build_link_prediction_dataset(config(root, name, target, seed=11))
        same = {s: bool(getattr(a, s).shape == getattr(b, s).shape
                        and (getattr(a, s) == getattr(b, s)).all()) for s in SPLITS}
        line = (f"{name}: loaded with every leak check passing; training seed 3 vs 11 -> splits "
                f"identical: {all(same.values())} {same}; sizes train/val/test pos "
                f"{a.train_pos_edge_index.size(1)}/{a.val_pos_edge_index.size(1)}/"
                f"{a.test_pos_edge_index.size(1)}, val/test neg {a.val_neg_edge_index.size(1)}/"
                f"{a.test_neg_edge_index.size(1)}")
        if name in DERIVED:
            # The loader's validation vs the J1.2 file (local ids of the node.dat ranks).
            task_dir = root / "lp_derived" / name
            from src.data.link_prediction_dataset import _load_local_hgb_nodes
            nodes = _load_local_hgb_nodes(task_dir, {0: "a", 1: "b", 2: "c", 3: "d"})
            to_local = nodes.raw_to_local
            file_val = {(to_local[u], to_local[v])
                        for u, v in bd.read_split(task_dir, int(target[1][4:])).valid}
            loader_val = pairs(a.val_pos_edge_index, a.src_node_offset, a.dst_node_offset)
            line += (f"; link.dat.valid {len(file_val)} facts, loader validation "
                     f"{len(loader_val)}, in both {len(file_val & loader_val)}")
        print(line + f" ({time.time() - t0:.0f} s)", flush=True)

    # Mutations on a copy of acm_pa (route 1: both rows / r row; route 2: r' row only).
    for label, rows in (("both rows (as HGB stores a fact)", "both"), ("r row only", "r"),
                        ("r' row only", "rev")):
        work = scratch / "mut" / "lp_derived" / "acm_pa"
        shutil.rmtree(scratch / "mut", ignore_errors=True)
        shutil.copytree(root / "lp_derived" / "acm_pa", work)
        u, v = sorted(bd.read_split(work, 2).test)[0]
        add = {"both": [f"{u}\t{v}\t2\t1.0\n", f"{v}\t{u}\t3\t1.0\n"], "r": [f"{u}\t{v}\t2\t1.0\n"],
               "rev": [f"{v}\t{u}\t3\t1.0\n"]}[rows]
        with (work / "link.dat").open("ab") as handle:
            handle.write("".join(add).encode())
        try:
            data = build_link_prediction_dataset(config(scratch / "mut", "acm_pa", DERIVED["acm_pa"], 3))
            rho = data.fact_layer.summary()["rho"]
            print(f"mutation {label}: NOT caught by the loader; rho now {rho}; file-level guard "
                  f"test_fact_leaks = {bd.test_fact_leaks(work, 2, 3)}", flush=True)
        except (ValueError, AssertionError) as exc:
            print(f"mutation {label}: caught by the loader ({type(exc).__name__}: {exc})", flush=True)
    shutil.rmtree(scratch / "mut", ignore_errors=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
