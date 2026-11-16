#!/bin/bash
# H-1: install strenum at HetSheaf's uv.lock version (0.4.15), --no-deps; versions of torch/PyG/DGL before and after.
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
cd /mnt/data/peeradon/h2snn-hetsheaf
echo "HEAD $(git rev-parse HEAD) host $(hostname) job $SLURM_JOB_ID"
pip freeze > /tmp/freeze_before_$SLURM_JOB_ID.txt
echo "before:"; grep -i -E "^(torch|torch[-_]geometric|dgl|strenum|torch[-_]scatter|torch[-_]sparse|numpy)==" /tmp/freeze_before_$SLURM_JOB_ID.txt
pip install --no-deps "strenum==0.4.15" 2>&1 | tail -3
pip freeze > /tmp/freeze_after_$SLURM_JOB_ID.txt
echo "diff of pip freeze (before -> after):"; diff /tmp/freeze_before_$SLURM_JOB_ID.txt /tmp/freeze_after_$SLURM_JOB_ID.txt
echo "pip check:"; pip check
python -c "import strenum, torch, torch_geometric, dgl; print('strenum', strenum.__file__); print('torch', torch.__version__, 'pyg', torch_geometric.__version__, 'dgl', dgl.__version__)"
