"""LR-4: parsed comparison of notebooks/baselines_graph_classification.ipynb against HEAD (run from H2S-NN/ on
the login node as an editor check; pure JSON, no computation). Same cell count, sources, metadata, cell types and
top-level keys; every code cell has execution_count null and outputs []; nothing else differs."""
import json, subprocess, sys

PATH = "notebooks/baselines_graph_classification.ipynb"
head = json.loads(subprocess.run(["git", "show", f"HEAD:H2S-NN/{PATH}"], capture_output=True, text=True,
                                 check=True).stdout)
new = json.loads(open(PATH, encoding="utf-8").read())
ok = len(head["cells"]) == len(new["cells"])
print("cells", len(head["cells"]), len(new["cells"]))
for key in head:
    if key != "cells" and head[key] != new.get(key):
        ok = False; print("top-level differs:", key)
if set(head) != set(new):
    ok = False; print("top-level keys differ")
changed = []
for i, (a, b) in enumerate(zip(head["cells"], new["cells"])):
    for key in set(a) | set(b):
        if key in ("execution_count", "outputs") and a["cell_type"] == "code":
            continue
        if a.get(key) != b.get(key):
            ok = False; print(f"cell {i}: {key} differs")
    if b["cell_type"] == "code":
        if b.get("execution_count") is not None or b.get("outputs") != []:
            ok = False; print(f"cell {i}: not clean")
        if a.get("execution_count") is not None or a.get("outputs"):
            changed.append(i)
print("cleared cells (had outputs or a count at HEAD):", changed)
print("parsed comparison:", "OK" if ok else "DIFFERS")
sys.exit(0 if ok else 1)
