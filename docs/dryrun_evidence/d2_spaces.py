"""Card 2 D2: the search space of every runnable baseline cell (adapter.search_space), for choosing the largest
memory point. No data read. Run from H2S-NN/."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.baselines.registry import iter_cells, get_adapter
out = {}
for c in iter_cells():
    if c["runnable"]:
        out[f"{c['model']}|{c['task']}|{c['dataset']}"] = get_adapter(c["model"]).search_space(c["task"], c["dataset"])
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
seen = {}
for k, v in out.items():
    m, t, d = k.split("|")
    key = (m, t, json.dumps(v, sort_keys=True, default=str))
    seen.setdefault(key, []).append(d)
for (m, t, sp), ds in seen.items():
    print(f"== {m} {t} {ds}\n   {sp}")
