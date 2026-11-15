#!/bin/bash
# M3a-A1: LP decoder in {dot, DistMult}: tests, the two card mutations (each against its target test only,
# applied and restored inside this job), then the gate. With E exported.
#   srun --jobid=<allocation> --overlap bash docs/partM3a_evidence/a1_job.sh   (or sbatch)
#SBATCH --job-name=m3a-a1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
export CUDA_VISIBLE_DEVICES=""
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
git status --short -- . | grep -v '^??' | sed 's/^/working tree: /'
TMP=${TMPDIR:-/tmp}/m3a_a1_$$; mkdir -p ${TMP}

echo "===== tests"
python -m unittest -v tests.test_partM3a_decoder tests.test_search_spaces tests.test_wandb_and_runs \
    tests.test_cooperative tests.test_task_heads tests.test_paths_and_facts 2>&1 | grep -vE "FutureWarning|autocast"
echo "tests_exit=${PIPESTATUS[0]}"

F=scripts/run_h2snn_wandb_tuning.py
cp ${F} ${TMP}/tuner.orig
echo; echo "===== mutation 1: the tuner omits decoder_type on LP -> test_lp_sweeps_tune_the_decoder"
python - <<'EOF'
from pathlib import Path
p = Path("scripts/run_h2snn_wandb_tuning.py"); t = p.read_text()
old = '    if task_type == "link_prediction":\n        parameters["decoder_type"]'
assert t.count(old) == 1
p.write_text(t.replace(old, '    if False:\n        parameters["decoder_type"]'))
EOF
python -m unittest tests.test_partM3a_decoder.LinkDecoderSweepTests.test_lp_sweeps_tune_the_decoder 2>&1 | grep -E "^AssertionError|Error:|^Ran|^(OK|FAILED)"
cp ${TMP}/tuner.orig ${F}; cmp ${F} ${TMP}/tuner.orig && echo "restored: identical"

echo; echo "===== mutation 2: concat_mlp accepted by the tuner -> test_concat_mlp_is_refused_by_the_tuner"
python - <<'EOF'
from pathlib import Path
p = Path("scripts/run_h2snn_wandb_tuning.py"); t = p.read_text()
old = 'SPACE_DECODER_TYPE: dict[str, Any] = {"values": ["dot", "distmult"]}'
assert t.count(old) == 1
p.write_text(t.replace(old, 'SPACE_DECODER_TYPE: dict[str, Any] = {"values": ["dot", "distmult", "concat_mlp"]}'))
EOF
python -m unittest tests.test_partM3a_decoder.ConcatMlpRefusalTests.test_concat_mlp_is_refused_by_the_tuner 2>&1 | grep -E "^AssertionError|Error:|^Ran|^(OK|FAILED)"
cp ${TMP}/tuner.orig ${F}; cmp ${F} ${TMP}/tuner.orig && echo "restored: identical"
python -m unittest tests.test_partM3a_decoder 2>&1 | tail -n 1

echo
bash ../docs/partM3a_evidence/gate.sh
echo "end=$(date -Is)"
