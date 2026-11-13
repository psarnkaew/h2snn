"""Part P-fix, F2 (`docs/H2S-NN_partPfix_card.md`; decision D-P6): H2G-former's width and the
MetaPath2Vec channel.

- `hidden_dim` in {64, 128, 256, 512}; with `metapath_pe` on, `dim_pe = min(128, hidden_dim // 2)`
  (64->32, 128->64, 256->128, 512->128); the raw channel keeps `hidden_dim - dim_pe > 0`;
- every (`hidden_dim`, `metapath_pe`) pair of the space builds the encoder on a tiny ACM-shaped graph
  and runs one forward (finite);
- the H2GB configs' (dim_hidden, dim_pe) pairs of the MetaPath2Vec channel are reachable (values read
  from the config files).
Card mutation: `dim_pe` fixed at 128 -> the 64/128 forward test fails.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.baselines.adapters._h2gformer_encoders import LABEL, METAPATH, RAW, metapath_dim_pe
from src.baselines.protocol import NODE_CLASSIFICATION

H2GB_CONFIGS = Path(__file__).resolve().parent.parent / "vendor" / "H2GB" / "configs"


def _grid() -> list[int]:
    from src.baselines.adapters.h2gformer import ADAPTER

    return list(ADAPTER.search_space(NODE_CLASSIFICATION, "acm")["hidden_dim"]["values"])


class WidthRuleTests(unittest.TestCase):
    def test_grid_and_dim_pe_per_width(self) -> None:
        self.assertEqual(_grid(), [64, 128, 256, 512])
        self.assertEqual({h: metapath_dim_pe(h) for h in _grid()},
                         {64: 32, 128: 64, 256: 128, 512: 128})
        for hidden in _grid():
            self.assertGreater(hidden - metapath_dim_pe(hidden), 0)

    def test_h2gb_config_pairs_are_reachable(self) -> None:
        pairs = []
        for path in sorted(H2GB_CONFIGS.glob("*/*H2Gformer*.yaml")):
            text = path.read_text(encoding="utf-8")
            if "Hetero_Metapath" not in re.search(r"node_encoder_name:\s*(\S+)", text).group(1):
                continue
            hidden = int(re.search(r"^\s*dim_hidden:\s*(\d+)", text, re.M).group(1))
            block = text.split("posenc_Hetero_Metapath:", 1)[1]
            pairs.append((hidden, int(re.search(r"dim_pe:\s*(\d+)", block).group(1))))
        self.assertEqual(sorted(set(pairs)), [(64, 32), (256, 128), (512, 128)])
        for hidden, dim_pe in pairs:
            with self.subTest(pair=(hidden, dim_pe)):
                self.assertIn(hidden, _grid())
                self.assertEqual(metapath_dim_pe(hidden), dim_pe)


class ForwardTests(unittest.TestCase):
    def test_every_pair_of_the_space_runs_one_forward(self) -> None:
        from src.baselines.adapters.h2gformer import build_node_classification_model
        from src.baselines.protocol import BaselineParams
        from tests.test_partP_h2gformer import _tiny

        for hidden in _grid():
            for metapath_pe in (False, True):
                with self.subTest(hidden_dim=hidden, metapath_pe=metapath_pe):
                    hetero, bundle, _ = _tiny()
                    params = BaselineParams({"hidden_dim": hidden, "gt_layers": 1,
                                             "attn_heads": 2, "metapath_pe": metapath_pe})
                    stack = (RAW, LABEL, METAPATH) if metapath_pe else (RAW, LABEL)
                    torch.manual_seed(0)
                    model = build_node_classification_model(
                        hetero, "paper", bundle, params, stack, device=torch.device("cpu"),
                        seed=0, metapath_epochs=1)
                    if metapath_pe:
                        width = int(hetero["paper"]["pestat_Hetero_Metapath"].size(1))
                        hand = {64: 32, 128: 64, 256: 128, 512: 128}[hidden]
                        self.assertEqual(width, hand)
                        self.assertGreater(hidden - width, 0)  # the raw channel
                    model.eval()
                    hetero.split = "val"
                    with torch.no_grad():
                        out = model(hetero)
                    out = out[0] if isinstance(out, tuple) else out
                    self.assertEqual(out.size(0), int(hetero["paper"].num_nodes))
                    self.assertTrue(bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    unittest.main()
