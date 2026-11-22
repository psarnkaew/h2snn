#!/bin/bash
# Merge c12+c14 step 3: re-capture the v3b golden and card 5 T4 reference from the merged code, with the scripts,
# environment and settings that produced them (CPU only, pinned kernel environment E, h2snn-b; the scripts set
# 1 thread themselves). Then run the two tests against the new captures.
#   sbatch --nodelist=g2n01 --mem=1300M docs/merge_c12_c14_evidence/capture.sh <base sha>
#SBATCH --job-name=mc-capture
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=/mnt/data/peeradon/h2snn-merge/H2S-NN/logs/slurm-%j.out
set -uo pipefail
BASE="${1:?base sha}"
cd /mnt/data/peeradon/h2snn-merge/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
lscpu | grep 'Model name'
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability())'
F='GBWarning|expandable|allocation pattern|PYTORCH_CUDA|gb_warning|An experimental|See https|and set the|FutureWarning|autocast|_flatten\("", torch.load'
REF=../docs/legacy_evidence/v3b_golden_cluster.pt
sha256sum ${REF} | sed 's/^/before: /'
python ../docs/legacy_evidence/v3b_golden.py capture ${REF} 2>&1 | grep -vE "${F}"; echo "golden_capture_exit=${PIPESTATUS[0]}"
sha256sum ${REF} | sed 's/^/after: /'
OUT=../docs/card5_evidence/base_capture_${BASE}.json
python ../docs/card5_evidence/capture_base.py ${OUT} 2>&1 | grep -vE "${F}|\| INFO \|"; echo "t4_capture_exit=${PIPESTATUS[0]}"
sha256sum ${OUT}
echo "===== compare old T4 capture (894fd74) with the new one"
diff <(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k:d[k] for k in ("hgb_full_batch","acm_minibatch_flag")}, indent=1))' ../docs/card5_evidence/base_capture_894fd74.json) \
     <(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k:d[k] for k in ("hgb_full_batch","acm_minibatch_flag")}, indent=1))' ${OUT})
cmp <(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["baselines_1_cell_list"])' ../docs/card5_evidence/base_capture_894fd74.json) \
    <(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["baselines_1_cell_list"])' ${OUT}) && echo "baselines_1_cell_list: identical to 894fd74" || echo "baselines_1_cell_list: differs from 894fd74"
echo "end=$(date -Is)"
