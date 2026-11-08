"""K3 mutations: one per rule; each must make its targeted test fail.
Run from H2S-NN/: python ../docs/WISP_evidence/k3_mutation_check.py"""
import subprocess
import sys
from pathlib import Path

G = "tests.test_wisp.WispGuardTests."
CFG = "src/configs/config.py"
MUTATIONS = {
    "r allowed in a search space": (CFG,
        '''                                           "wisp_proposal", "path_weight_context",
                                           "wisp_walks_per_anchor")''',
        '''                                           "wisp_proposal", "path_weight_context")''',
        G + "test_r_is_never_in_a_search_space"),
    "base config: l_max not checked": (CFG,
        'if int(data.get("wisp_max_length", 4)) != 4:',
        'if False and int(data.get("wisp_max_length", 4)) != 4:',
        G + "test_tuning_base_config_carries_the_pre_registered_wisp_settings"),
    "base config: adaptive proposal not checked": (CFG,
        'if str(data.get("wisp_proposal", "uniform")) != "uniform":',
        'if False and str(data.get("wisp_proposal", "uniform")) != "uniform":',
        G + "test_tuning_base_config_carries_the_pre_registered_wisp_settings"),
    "base config: r not checked": (CFG,
        "        if walks not in (0, calibrated):",
        "        if False and walks not in (0, calibrated):",
        G + "test_tuning_base_config_carries_the_pre_registered_wisp_settings"),
    "path_weight_context allowed off the ladder": (CFG,
        '''            or self.endpoint_edges or self.free_path_maps or self.coupling == "pair_attention"
''', "",
        G + "test_path_weight_context_is_the_ladder_appendix_pair_only"),
    "r table ignores l_max": (CFG,
        "            walks = table[self.wisp_max_length]",
        "            walks = table.get(4, table[self.wisp_max_length])",
        G + "test_r_resolves_from_the_table_for_its_l_max"),
    "uncalibrated l_max does not raise": (CFG,
        "            if self.wisp_max_length not in table:",
        "            if False and self.wisp_max_length not in table:",
        G + "test_r_resolves_from_the_table_for_its_l_max"),
    "build_model consumes an unresolved path source": ("src/models/h2snn_model.py",
        '''def build_model(config: TrainConfig, data: PreparedH2SData) -> H2SNN:
    if config.data.path_source not in ("schemas", "wisp"):''',
        '''def build_model(config: TrainConfig, data: PreparedH2SData) -> H2SNN:
    if False and config.data.path_source not in ("schemas", "wisp"):''',
        G + "test_an_unresolved_path_source_is_refused_by_its_consumers"),
    "the path builder consumes an unresolved path source": ("src/data/dataset.py",
        '''    if config.data.path_source not in ("schemas", "wisp"):''',
        '''    if False and config.data.path_source not in ("schemas", "wisp"):''',
        G + "test_an_unresolved_path_source_is_refused_by_its_consumers"),
}
caught = 0
for name, (path, old, new, test) in MUTATIONS.items():
    file = Path(path)
    original = file.read_text(encoding="utf-8")
    assert original.count(old) == 1, name
    file.write_text(original.replace(old, new), encoding="utf-8")
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", test], capture_output=True, text=True)
    finally:
        file.write_text(original, encoding="utf-8")
    ok = out.returncode != 0
    caught += ok
    print(f"{name}: {'CAUGHT' if ok else 'NOT CAUGHT'} ({out.stderr.strip().splitlines()[-1]})",
          flush=True)
clean = subprocess.run([sys.executable, "-m", "unittest", G[:-1]], capture_output=True, text=True)
print(f"caught {caught}/{len(MUTATIONS)}; clean after restore: {clean.stderr.strip().splitlines()[-1]}")
