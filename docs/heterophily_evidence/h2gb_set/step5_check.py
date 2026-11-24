"""Card 19 step 5 (read-only): the perturbation generator's meta-path list equals het.h2gb_metapaths on ACM and DBLP,
and card 7's original-graph h equals the card 19 MLH. Regenerates nothing; reads graph.json of the original graphs."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "H2S-NN"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "heterophily"))
import compute_heterophily as ref  # noqa: E402  (the generator's reference module, make_graphs.py:80)
from src.analysis import heterophily as het  # noqa: E402
from src.data.data_layout import REPO_DATA_ROOT  # noqa: E402

OUT = Path(__file__).resolve().parent
out = {}
for ds, name in (("acm", "ACM"), ("dblp", "DBLP")):
    data = ref.load_hgb(REPO_DATA_ROOT, ds)              # the generator's loader (make_graphs.py:170)
    target = ref.target_type(data)
    gen = [het.metapath_name([tuple(e) for e in p]) for p in ref.h2gb_metapaths(data, target)]
    sym, _ = het.symmetrise_facts(data)
    harness = [het.metapath_name(p) for p in het.h2gb_metapaths(sym.edge_types, target)]
    graph = json.loads((REPO_DATA_ROOT / "NC" / f"{name}_pert" / "original" / "graph.json").read_text())
    mlh = json.loads((OUT / "records" / f"nc__{ds}__mlh.json").read_text())["value"]
    out[ds] = {"generator_paths": gen, "harness_paths": harness, "paths_equal": gen == harness,
               "card7_original_h": graph["h"], "card7_original_h2_index": graph["h2_index"],
               "card19_mlh": mlh, "abs_diff": abs(graph["h"] - mlh)}
    print(ds, out[ds])
(OUT / "step5.json").write_text(json.dumps(out, indent=1))
