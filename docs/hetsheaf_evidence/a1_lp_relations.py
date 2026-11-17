"""A1: relation tuples and feature widths of the local LP bundles (structure only; no metric)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "H2S-NN"))
from src.baselines.bundles import build_bundle
for ds in ["pubmed", "lastfm", "acm_pa", "dblp_ap", "imdb_ma", "freebase_bb"]:
    b = build_bundle("link_prediction", ds)
    d = b.data
    print(ds, "target", d.target_relation, "N", d.x.size(0), "E", d.edge_index.size(1))
    print("  types", list(d.node_type_names), "dims", dict(d.feature_dims), "slices", dict(d.node_slices))
    import torch
    counts = torch.bincount(d.edge_type, minlength=len(d.relation_tuples)).tolist()
    for r, c in zip(d.relation_tuples, counts):
        print("   ", r, c)
