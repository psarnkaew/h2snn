"""Card 15 smoke: one epoch on acm_pp through `scripts/partn_smoke.py` (its checks unchanged: finite losses, no stop
condition, no collapse, parameters move), with its fixed two epochs set to one. H2S-NN uses partn_smoke's LP template
(acm_pa_h2snn_v3b.yaml, the task swapped: the same as configs/experiments/acm_pp_h2snn_v3b.yaml); GCN runs at its
default protocol through the shared LP loop.

    python scripts/diagnostics/card15/smoke_one_epoch.py --run h2snn:lp:acm_pp --out <json>
    python scripts/diagnostics/card15/smoke_one_epoch.py --run gcn:lp:acm_pp --out <json>
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import partn_smoke  # noqa: E402

if __name__ == "__main__":
    partn_smoke.EPOCHS = 1
    partn_smoke.main()
