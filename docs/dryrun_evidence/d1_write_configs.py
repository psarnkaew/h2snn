"""Card 2 D1: write the missing configs/experiments/<dataset>_h2snn_v3b.yaml from the existing ones (same keys).

NC (freebase + 8 H2GB NC) from dblp_h2snn_v3b.yaml (no ACM-only keys); LP from acm_pa_h2snn_v3b.yaml with the
dataset's preset target relation; GC from mutag_h2snn_v3b.yaml. Only the dataset's own data keys change, plus
r and l_max named explicitly (spec v1.18: tuned; no calibration) and, on H2GB NC, data.batch_size. Never overwrites.
Run from H2S-NN/ (text only; no data read).
"""
from pathlib import Path

import yaml

EXP = Path("configs/experiments")
SOLVER_NOTE = "Solver settings are the fixed card values (never tuned)."
R_NOTE = ("l_max 4 and r 8 named explicitly (spec v1.18: l_max and r are tuned, no calibrated r; 8 is the lowest of "
          "the tuned range {8, 16, 32, 64}; the sweep overrides both)")
H2GB = ["ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns"]
LP = {"dblp_ap": ["author", "rel_0", "paper"], "imdb_ma": ["movie", "rel_2", "actor"],
      "freebase_bb": ["book", "rel_0", "book"], "pubmed": ["disease", "rel_2", "disease"],
      "lastfm": ["user", "rel_0", "artist"]}
GC = ["proteins", "enzymes", "nci1"]


def body(template: str) -> list[str]:
    """The template's lines without its leading comment block."""
    lines = (EXP / template).read_text().splitlines()
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    return lines


def write(name: str, header: list[str], lines: list[str]) -> None:
    path = EXP / f"{name}_h2snn_v3b.yaml"
    if path.exists():
        raise FileExistsError(path)
    text = "\n".join([f"# {h}" for h in header] + lines) + "\n"
    yaml.safe_load(text)
    path.write_text(text)
    print("wrote", path)


def replace_data(lines: list[str], data: list[str], name: str) -> list[str]:
    start = lines.index("data:")
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith(" "))
    out = lines[:start] + ["data:"] + [f"  {d}" for d in data] + lines[end:]
    return [f"exp_name: {name}_h2snn_v3b" if l.startswith("exp_name:") else l for l in out]


nc = body("dblp_h2snn_v3b.yaml")
write("freebase", [f"Card 2 D1 (dry-run): HGB Freebase NC (target book, featureless), v3-B anchored, arm `full`; built as",
                   f"dblp_h2snn_v3b.yaml. {R_NOTE}. {SOLVER_NOTE}"],
      replace_data(nc, ["dataset_name: freebase", "root: data", "face_sample_k: 50", "wisp_max_length: 4",
                        "wisp_walks_per_anchor: 8"], "freebase"))
for name in H2GB:
    write(name, [f"Card 2 D1 (dry-run): H2GB NC {name} (mini-batch, MB1), v3-B anchored, arm `full`; built as dblp_h2snn_v3b.yaml.",
                 f"{R_NOTE}. data.batch_size 32: H2GB's own batch size in its GNN configs (vendor/H2GB/configs/*/*-GCN.yaml",
                 "train.batch_size); in MINIBATCH_BATCH_SIZES; H2S-NN's sweep has no batch_size axis, so this value is",
                 f"the batch size of every trial (decision for the author, dry-run report). Epochs: the H2GB budget. {SOLVER_NOTE}"],
          replace_data(nc, [f"dataset_name: {name}", "root: data", "face_sample_k: 50", "batch_size: 32",
                            "wisp_max_length: 4", "wisp_walks_per_anchor: 8"], name))
lp = body("acm_pa_h2snn_v3b.yaml")
for name, rel in LP.items():
    write(name, [f"Card 2 D1 (dry-run): link prediction on {name} (target {rel[0]} -{rel[1]}-> {rel[2]}, the dataset's preset),",
                 f"v3-B anchored, arm `full`, WISP from both endpoint types; built as acm_pa_h2snn_v3b.yaml. {R_NOTE}.",
                 SOLVER_NOTE],
          replace_data(lp, [f"dataset_name: {name}", "task_type: link_prediction", "root: data",
                            f"target_relation: [{', '.join(rel)}]", "decoder_type: dot", "wisp_max_length: 4",
                            "wisp_walks_per_anchor: 8"], name))
gc = body("mutag_h2snn_v3b.yaml")
for name in GC:
    write(name, [f"Card 2 D1 (dry-run): graph classification on {name.upper()} (typed by label), v3-B anchored, arm `full`,",
                 f"WISP from every node of each graph; built as mutag_h2snn_v3b.yaml. {R_NOTE}. {SOLVER_NOTE}"],
          replace_data(gc, [f"dataset_name: {name}", "root: data", "wisp_max_length: 4", "wisp_walks_per_anchor: 8"], name))
