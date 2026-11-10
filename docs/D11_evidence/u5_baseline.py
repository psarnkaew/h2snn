"""U5 reference baseline: HINormer with its published parameters, 3 seeds per dataset (CPU: DGL has no
working GPU build here). Datasets without published parameters are reported as not run (no tuning).
Run from H2S-NN/:  python ../docs/D11_evidence/u5_baseline.py [dataset ...]"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from src.baselines.registry import NODE_CLASSIFICATION, get_adapter  # registry before bundles: the
from src.baselines.bundles import build_bundle  # other order segfaults (DGL / PyG import order)

out_dir = Path.cwd().parent / "docs" / "D11_evidence" / "screen"
out_dir.mkdir(parents=True, exist_ok=True)
datasets = sys.argv[1:] or ["acm", "dblp", "imdb"]
# Bundles (PyG) are built BEFORE the adapter module is imported: the other order segfaults (DLL load
# order of DGL vs the PyG extensions), as observed in this environment.
bundles = {dataset: build_bundle(NODE_CLASSIFICATION, dataset) for dataset in datasets}
adapter = get_adapter("hinormer")
results = {}
for dataset in datasets:
    params = adapter.published_params(NODE_CLASSIFICATION, dataset)
    if params is None:
        results[dataset] = {"status": "not run: no published parameters (no tuning in the screen)"}
        print(dataset, results[dataset], flush=True)
        continue
    bundle = bundles[dataset]
    runs = []
    for seed in (42, 43, 44):
        start = time.time()
        metrics = adapter.fit_task(bundle, params, task=NODE_CLASSIFICATION, seed=seed,
                                   device=torch.device("cpu"))
        runs.append({"seed": seed, "minutes": (time.time() - start) / 60, **{
            k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
        print(dataset, seed, {k: round(v, 4) for k, v in runs[-1].items()}, flush=True)
    macro = [r["test_macro_f1"] for r in runs]
    micro = [r["test_micro_f1"] for r in runs]
    results[dataset] = {"status": "ran", "params_source": "published", "runs": runs,
                        "test_macro_f1_mean": statistics.fmean(macro),
                        "test_macro_f1_std": statistics.stdev(macro),
                        "test_micro_f1_mean": statistics.fmean(micro),
                        "test_micro_f1_std": statistics.stdev(micro)}
    (out_dir / "baseline_hinormer.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
(out_dir / "baseline_hinormer.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
