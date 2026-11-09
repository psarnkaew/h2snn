"""J5.5 (spec v1.9 F16): HeaRT's random-fill share per task, from each task's heart/meta.json
(entries drawn uniformly because fewer than 250 candidates of a half-list score above zero).
Read only. Run from H2S-NN/: python ../docs/lp_evidence/j55_random_fill_share.py"""
import json
from pathlib import Path

FOLDERS = {"acm_pa": "lp_derived/acm_pa", "dblp_ap": "lp_derived/dblp_ap",
           "imdb_ma": "lp_derived/imdb_ma", "freebase_bb": "lp_derived/freebase_bb",
           "pubmed": "PubMed", "lastfm": "LastFM"}
for task, folder in FOLDERS.items():
    meta = json.loads((Path("data") / folder / "heart" / "meta.json").read_text(encoding="utf-8"))
    cells = []
    for split in ("valid", "test"):
        m = meta[split]
        share = m["random_fill_entries"] / (m["positives"] * meta["num_samples"])
        cells.append(f"{split} {100 * share:.1f} % ({m['lists_with_random_fill']} half-lists, "
                     f"min fill pool {m['min_fill_pool']}, min eligible {m['min_eligible_candidates']})")
    print(f"{task}: " + "; ".join(cells))
