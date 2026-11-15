"""M3a-fix F1 (evidence): the dot and DistMult readouts are unchanged by the concat_mlp removal.

Loads `src/models/link_readout.py` as of v3b 310e5c7 (`git show`, into a temporary file) and the current
one; for dot and distmult, under the same seed: every parameter (names, values) and the logits of the
same input are torch.equal, also after one backward (gradients equal). Run from H2S-NN/.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from src.models.link_readout import LinkPredictionReadout as New  # noqa: E402

old_source = subprocess.run(["git", "show", "310e5c7:H2S-NN/src/models/link_readout.py"],
                            capture_output=True, text=True, check=True).stdout
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "link_readout_310e5c7.py"
    path.write_text(old_source)
    spec = importlib.util.spec_from_file_location("link_readout_310e5c7", path)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
Old = old_module.LinkPredictionReadout

failures = 0
for decoder in ("dot", "distmult"):
    for num_relations in (1, 2):
        results = []
        for cls in (Old, New):
            torch.manual_seed(0)
            readout = cls(d=8, decoder_type=decoder, hidden_dim=16, dropout=0.3,
                          num_relations=num_relations, input_dim=16)
            g = torch.Generator().manual_seed(1)
            x = torch.randn(10, 16, generator=g)
            edges = torch.randint(0, 10, (2, 12), generator=g)
            rel = torch.randint(0, num_relations, (12,), generator=g)
            logits = readout(x, edges, rel)
            logits.square().sum().backward()
            results.append((dict(readout.named_parameters()), logits.detach(),
                            {n: p.grad.clone() for n, p in readout.named_parameters()}))
        (p_old, l_old, g_old), (p_new, l_new, g_new) = results
        same = (list(p_old) == list(p_new)
                and all(torch.equal(p_old[n], p_new[n]) for n in p_old)
                and torch.equal(l_old, l_new)
                and all(torch.equal(g_old[n], g_new[n]) for n in g_old))
        failures += not same
        print(f"{decoder:8s} relations={num_relations}: parameters {list(p_new)}; "
              f"parameters, logits, gradients equal to 310e5c7: {same}")
print(f"readout_equal_failures={failures}")
sys.exit(1 if failures else 0)
