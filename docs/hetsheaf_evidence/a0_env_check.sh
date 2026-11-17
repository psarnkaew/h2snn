#!/bin/bash
# A0: which of HetSheaf's import-time packages exist in h2snn-b (no install).
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
cd /mnt/data/peeradon/h2snn-hetsheaf
echo "HEAD $(git rev-parse HEAD) host $(hostname) job $SLURM_JOB_ID"
python - <<'PY'
import importlib, importlib.metadata as md
for m, dist in [("strenum","StrEnum"),("lightning","lightning"),("pytorch_lightning","pytorch-lightning"),("omegaconf","omegaconf"),("hydra","hydra-core"),("torchmetrics","torchmetrics"),("torch_scatter","torch-scatter"),("torch_sparse","torch-sparse"),("torch_householder","torch-householder"),("torch_geometric","torch-geometric"),("torch","torch"),("dgl","dgl"),("numpy","numpy"),("scipy","scipy"),("wandb","wandb"),("lightning_utilities","lightning-utilities")]:
    try:
        v = md.version(dist)
    except Exception as e:
        v = "not-installed(dist)"
    try:
        importlib.import_module(m); ok = "import-ok"
    except Exception as e:
        ok = f"import-fail {type(e).__name__}: {str(e)[:120]}"
    print(f"{m:22s} {v:24s} {ok}")
PY
pip check 2>&1 | head
