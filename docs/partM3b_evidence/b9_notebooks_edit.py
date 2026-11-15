"""Part M3b B9 item 2 (editor script, run on the login node; no computation): point every
`H2S-NN/notebooks/h2snn_*_colab.ipynb` at an existing config of `configs/experiments/` (the five `*_h2snn_v3b.yaml`) or,
for a dataset without one, at its preset in `src/configs/config.py` (`{'data': {'dataset_name': DATASET}}`; for the tuning
scripts, which take a file, that dict is written at run time next to the experiment's outputs). No new file under
`configs/`, no other cell or setting changes; outputs and execution counts are cleared; each file keeps its JSON layout
(indent / separators / trailing newline as before). Every replacement must match exactly once.

    python ../docs/partM3b_evidence/b9_notebooks_edit.py      (from H2S-NN/)
"""

from __future__ import annotations

import json
from pathlib import Path

PRESET_NOTE = "  # None: no config file; start from the dataset's preset (src/configs/config.py)"

EDITS: dict[str, list[tuple[str, str]]] = {
    "h2snn_node_classification_colab.ipynb": [
        ("    'acm': 'configs/experiments/acm_h2snn.yaml',\n", "    'acm': 'configs/experiments/acm_h2snn_v3b.yaml',\n"),
        ("    'dblp': 'configs/experiments/dblp_h2snn.yaml',\n", "    'dblp': 'configs/experiments/dblp_h2snn_v3b.yaml',\n"),
        ("    'imdb': 'configs/experiments/imdb_h2snn.yaml',\n", "    'imdb': 'configs/experiments/imdb_h2snn_v3b.yaml',\n"),
    ],
    "h2snn_node_classification_wandb_tuning_colab.ipynb": [
        ("CONFIG_PATH = Path('configs/experiments/acm_h2snn.yaml')  # acm/dblp/imdb *_h2snn.yaml",
         "CONFIG_PATH = Path('configs/experiments/acm_h2snn_v3b.yaml')  # acm/dblp/imdb *_h2snn_v3b.yaml"),
    ],
    "h2snn_wandb_paper_tuning_colab.ipynb": [
        ("#   configs/experiments/acm_h2snn.yaml             # node classification",
         "#   configs/experiments/acm_h2snn_v3b.yaml         # node classification"),
        ("#   configs/experiments/dblp_h2snn.yaml", "#   configs/experiments/dblp_h2snn_v3b.yaml"),
        ("#   configs/experiments/imdb_h2snn.yaml", "#   configs/experiments/imdb_h2snn_v3b.yaml"),
        ("#   configs/experiments/lastfm_h2snn.yaml          # link prediction",
         "#   configs/experiments/acm_pa_h2snn_v3b.yaml      # link prediction"),
        ("#   configs/experiments/mutag_h2snn.yaml           # graph classification",
         "#   configs/experiments/mutag_h2snn_v3b.yaml       # graph classification"),
        ("CONFIG_PATH = Path('configs/experiments/acm_h2snn.yaml')",
         "CONFIG_PATH = Path('configs/experiments/acm_h2snn_v3b.yaml')"),
    ],
    "h2snn_graph_classification_wandb_tuning_colab.ipynb": [
        ("BASE_CONFIG_PATH = Path('configs/experiments/mutag_h2snn.yaml')",
         "BASE_CONFIG_PATH = Path('configs/experiments/mutag_h2snn_v3b.yaml')"),
    ],
    "h2snn_graph_classification_colab.ipynb": [
        ("    'mutag': 'configs/experiments/mutag_h2snn.yaml',\n"
         "    'proteins': 'configs/experiments/proteins_h2snn.yaml',\n"
         "    'enzymes': 'configs/experiments/enzymes_h2snn.yaml',\n"
         "    'nci1': 'configs/experiments/nci1_h2snn.yaml',\n",
         "    'mutag': 'configs/experiments/mutag_h2snn_v3b.yaml',\n"
         f"    'proteins': None,{PRESET_NOTE}\n"
         "    'enzymes': None,\n"
         "    'nci1': None,\n"),
        ("base_config = Path(CONFIG_BY_DATASET[DATASET])\n"
         "payload = yaml.safe_load(base_config.read_text(encoding='utf-8'))\n",
         "base_config = CONFIG_BY_DATASET[DATASET] and Path(CONFIG_BY_DATASET[DATASET])\n"
         "payload = (yaml.safe_load(base_config.read_text(encoding='utf-8')) if base_config\n"
         "           else {'data': {'dataset_name': DATASET}})\n"),
    ],
    "h2snn_link_prediction_colab.ipynb": [
        ("    'lastfm': 'configs/experiments/lastfm_h2snn.yaml',\n"
         "    'amazonbook': 'configs/experiments/amazonbook_h2snn.yaml',\n"
         "    'amazon': 'configs/experiments/amazon_h2snn.yaml',\n"
         "    'pubmed': 'configs/experiments/pubmed_h2snn.yaml',\n"
         "    'youtube': 'configs/experiments/youtube_h2snn.yaml',\n",
         f"    'lastfm': None,{PRESET_NOTE}\n"
         "    'amazonbook': None,\n"
         "    'amazon': None,\n"
         "    'pubmed': None,\n"
         "    'youtube': None,\n"),
        ("base_config = Path(CONFIG_BY_DATASET[DATASET])\n"
         "payload = yaml.safe_load(base_config.read_text(encoding='utf-8'))\n",
         "base_config = CONFIG_BY_DATASET[DATASET] and Path(CONFIG_BY_DATASET[DATASET])\n"
         "payload = (yaml.safe_load(base_config.read_text(encoding='utf-8')) if base_config\n"
         "           else {'data': {'dataset_name': DATASET}})\n"),
    ],
    "h2snn_graph_classification_repeated_cv_wandb_tuning_colab.ipynb": [
        ("BASE_CONFIG_PATH = Path(f'configs/experiments/{DATASET}_h2snn.yaml')\n",
         "# mutag has a config file; the others start from their preset (src/configs/config.py), written below\n"
         "BASE_CONFIG_PATH = Path('configs/experiments/mutag_h2snn_v3b.yaml') if DATASET == 'mutag' else None\n"),
        ("STATE_DIR = OUTPUT_DIR / EXP_NAME / 'tuning_state'\n",
         "STATE_DIR = OUTPUT_DIR / EXP_NAME / 'tuning_state'\n"
         "if BASE_CONFIG_PATH is None:\n"
         "    import yaml\n"
         "    BASE_CONFIG_PATH = OUTPUT_DIR / EXP_NAME / 'preset_base_config.yaml'\n"
         "    BASE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
         "    BASE_CONFIG_PATH.write_text(yaml.safe_dump({'data': {'dataset_name': DATASET}}), encoding='utf-8')\n"),
    ],
    "h2snn_link_prediction_wandb_tuning_colab.ipynb": [
        ("CONFIG_PATH = Path('configs/experiments/lastfm_h2snn.yaml')\n",
         "DATASET = 'lastfm'  # no config file: start from its preset (src/configs/config.py), written below\n"
         "CONFIG_PATH = None\n"),
        ("STATE_DIR = OUTPUT_DIR / EXP_NAME / 'tuning_state'\n",
         "STATE_DIR = OUTPUT_DIR / EXP_NAME / 'tuning_state'\n"
         "if CONFIG_PATH is None:\n"
         "    import yaml\n"
         "    CONFIG_PATH = OUTPUT_DIR / EXP_NAME / 'preset_base_config.yaml'\n"
         "    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
         "    CONFIG_PATH.write_text(yaml.safe_dump({'data': {'dataset_name': DATASET}}), encoding='utf-8')\n"),
    ],
}


def _layout(raw: str, notebook: dict) -> tuple[dict, bool]:
    for kwargs in ({"separators": (",", ":")}, {"indent": 1}, {"indent": 2}):
        for newline in (False, True):
            text = json.dumps(notebook, ensure_ascii=False, **kwargs) + ("\n" if newline else "")
            if text == raw:
                return kwargs, newline
    raise SystemExit("unknown JSON layout")


def main() -> None:
    folder = Path("notebooks")
    for name, edits in EDITS.items():
        path = folder / name
        raw = path.read_text(encoding="utf-8")
        notebook = json.loads(raw)
        kwargs, newline = _layout(raw, notebook)
        for old, new in edits:
            hits = [c for c in notebook["cells"] if old in "".join(c.get("source", []))]
            if len(hits) != 1 or "".join(hits[0]["source"]).count(old) != 1:
                raise SystemExit(f"{name}: replacement does not match exactly once: {old[:60]!r}")
            text = "".join(hits[0]["source"]).replace(old, new)
            hits[0]["source"] = text.splitlines(keepends=True)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
        path.write_text(json.dumps(notebook, ensure_ascii=False, **kwargs) + ("\n" if newline else ""),
                        encoding="utf-8")
        print("edited", name, len(edits))


if __name__ == "__main__":
    main()
