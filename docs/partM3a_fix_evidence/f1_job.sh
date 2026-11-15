#!/bin/bash
# M3a-fix F1: concat_mlp refused by the config and removed from the model: tests, the readout equality check
# (dot/distmult vs 310e5c7), the card mutation (config accepts concat_mlp again; applied and restored inside
# this job), then the gate. With E exported.
#   srun --jobid=<allocation> --overlap bash docs/partM3a_fix_evidence/f1_job.sh   (or sbatch)
#SBATCH --job-name=m3afix-f1
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
TMP=${TMPDIR:-/tmp}/m3afix_f1_$$; mkdir -p ${TMP}

echo "===== tests"
python -m unittest -v tests.test_partM3afix_concat_mlp tests.test_task_heads tests.test_partM3a_decoder \
    tests.test_search_spaces tests.test_v2_model tests.test_partMB_link 2>&1 | grep -vE "FutureWarning|autocast"
echo "tests_exit=${PIPESTATUS[0]}"

echo; echo "===== readout equality, dot / distmult vs 310e5c7: python ../docs/partM3a_fix_evidence/f1_readout_equal.py"
python ../docs/partM3a_fix_evidence/f1_readout_equal.py; echo "readout_equal_exit=$?"

F=src/configs/config.py
cp ${F} ${TMP}/config.orig
echo; echo "===== mutation: the config accepts concat_mlp again -> test_config_refuses_concat_mlp"
python - <<'EOF'
from pathlib import Path
p = Path("src/configs/config.py"); t = p.read_text()
old = '''            if self.decoder_type == "concat_mlp":
                raise ValueError(REMOVED_DECODER_MESSAGE.format(where="data.decoder_type"))
            if self.decoder_type not in {"dot", "distmult"}:'''
new = '''            if self.decoder_type not in {"dot", "distmult", "concat_mlp"}:'''
assert t.count(old) == 1
p.write_text(t.replace(old, new))
EOF
python -m unittest tests.test_partM3afix_concat_mlp.ConcatMlpConfigTests.test_config_refuses_concat_mlp 2>&1 \
    | grep -E "^AssertionError|Error:|^Ran|^(OK|FAILED)"
cp ${TMP}/config.orig ${F}; cmp ${F} ${TMP}/config.orig && echo "restored: identical"
python -m unittest tests.test_partM3afix_concat_mlp 2>&1 | tail -n 1

echo
bash ../docs/partM3a_fix_evidence/gate.sh
echo "end=$(date -Is)"
