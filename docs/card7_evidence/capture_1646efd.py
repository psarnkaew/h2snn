"""Card 5 T4, `baselines_1_cell_list` key only, captured at v3b 1646efd (ops card T7.4) with the same
`docs/card5_evidence/capture_base.py`. Run from the H2S-NN/ folder of a 1646efd tree (capture_base puts the working
directory on sys.path, so `scripts.tsubame.make_cell_list` is 1646efd's). `capture_base.records()` is called as it
is, with its two model fits (`_fit`, the model parts that stay compared with 894fd74) stubbed out; only the
`baselines_1_cell_list` key is written.

    python capture_1646efd.py <out.json>
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from tests.partMB_fixtures import unreachable_real_data  # noqa: E402

unreachable_real_data()
spec = importlib.util.spec_from_file_location(
    "capture_base", Path(__file__).resolve().parents[1] / "card5_evidence" / "capture_base.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
capture._fit = lambda raw: None
import scripts.tsubame.make_cell_list as mcl  # noqa: E402

assert Path(mcl.__file__).resolve().is_relative_to(Path.cwd().resolve()), mcl.__file__
out = {"baselines_1_cell_list": capture.records()["baselines_1_cell_list"]}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"make_cell_list: {mcl.__file__}; baselines_1_cell_list: {len(out['baselines_1_cell_list'])} chars")
