#!/bin/bash
# M3a-fix F2: notebooks without the decoder switch and concat_mlp: the notebook tests, the card mutation
# (the switch restored in one notebook; applied and restored inside this job), then the gate. With E exported.
#   srun --jobid=<allocation> --overlap bash docs/partM3a_fix_evidence/f2_job.sh   (or sbatch)
#SBATCH --job-name=m3afix-f2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:40:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
export CUDA_VISIBLE_DEVICES=""
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
git status --short -- . | grep -v '^??' | sed 's/^/working tree: /'
TMP=${TMPDIR:-/tmp}/m3afix_f2_$$; mkdir -p ${TMP}
NB=tests.test_wandb_and_runs.WandbAndRunReportingTests

echo "===== notebook tests"
python -m unittest -v ${NB}.test_h2snn_notebooks_have_no_decoder_switch_or_concat_mlp \
    ${NB}.test_notebooks_reference_repeated_run_and_wandb_tuning_scripts \
    ${NB}.test_repeated_cv_notebook_is_clean_and_uses_fold_campaign 2>&1
echo "tests_exit=$?"
echo "===== the whole tests.test_wandb_and_runs module"
python -m unittest tests.test_wandb_and_runs 2>&1 | tail -n 3

F=notebooks/h2snn_graph_classification_colab.ipynb
cp ${F} ${TMP}/nb.orig
echo; echo "===== mutation: the switch restored in ${F} -> the new notebook test"
python - <<'EOF'
from pathlib import Path
p = Path("notebooks/h2snn_graph_classification_colab.ipynb"); t = p.read_text(encoding="utf-8")
old = '    "TUNE_METRIC = \'auroc\'\\n",\n'
assert t.count(old) == 1
p.write_text(t.replace(old, old + '    "INCLUDE_DECODER_TYPE_IN_TUNING = False\\n",\n'), encoding="utf-8")
EOF
python -m unittest ${NB}.test_h2snn_notebooks_have_no_decoder_switch_or_concat_mlp 2>&1 | grep -E "^AssertionError|^Ran|^(OK|FAILED)"
cp ${TMP}/nb.orig ${F}; cmp ${F} ${TMP}/nb.orig && echo "restored: identical"
python -m unittest ${NB}.test_h2snn_notebooks_have_no_decoder_switch_or_concat_mlp 2>&1 | tail -n 1

echo
bash ../docs/partM3a_fix_evidence/gate.sh
echo "end=$(date -Is)"
