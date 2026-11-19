"""Card 7 amendments 9-10 evidence: card 5's T4 (`test_card5_protocol_hooks.HgbUnchangedTests`) compares
`make_cell_list.py --phase baselines-1` with its capture at 894fd74 (`docs/card5_evidence/base_capture_894fd74.json`).
This prints, for the scripts of 894fd74, v3b 1646efd (ops card T7.4: HetSheaf / CSNN on ACM / DBLP NC moved to
`baselines-early`) and card7 HEAD, whether the list equals that capture, and whether card7 equals 1646efd."""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path("/mnt/data/peeradon/h2snn-model/H2S-NN")
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scripts/tsubame"))
capture = json.loads((root.parent / "docs/card5_evidence/base_capture_894fd74.json").read_text())["baselines_1_cell_list"]
lists = {}
with tempfile.TemporaryDirectory() as tmp:
    for name, sha in (("894fd74", "894fd7421ce8581d38b967906bde5593e881518a"),
                      ("v3b 1646efd", "1646efde6274255a712a0a0c3a425e0dd31e8ca0"), ("card7 HEAD", "HEAD")):
        text = subprocess.run(["git", "show", f"{sha}:H2S-NN/scripts/tsubame/make_cell_list.py"], cwd=root,
                              capture_output=True, text=True, check=True).stdout
        path = Path(tmp) / f"mcl_{len(lists)}.py"
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        lists[name] = "\n".join(mod.tuning_list("baselines-1", Path("/OUT"))) + "\n"
for name, text in lists.items():
    early = sum(line.startswith("# in phase baselines-early") for line in text.splitlines())
    print(f"{name}: equals the card 5 capture: {text == capture}; '# in phase baselines-early' lines: {early}")
print(f"card7 HEAD equals v3b 1646efd: {lists['card7 HEAD'] == lists['v3b 1646efd']}")
