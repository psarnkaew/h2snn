#!/bin/bash
# LR-3/LR-5 helper: list every unittest id (loader only, no test runs) of the working tree and of a given
# commit (extracted with git archive into the node's /tmp). Output: docs/legacy_evidence/test_ids_<label>.txt
#   srun --jobid=<allocation> --overlap bash docs/legacy_evidence/lr3_list_tests.sh <commit> <label>
set -uo pipefail
cd /mnt/data/peeradon/h2snn || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) git=$(git rev-parse HEAD)"
lscpu | grep 'Model name'
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads())'
LIST='import sys, unittest
def walk(s):
    for t in s:
        if isinstance(t, unittest.TestSuite): yield from walk(t)
        else: yield t.id()
ids = sorted(walk(unittest.TestLoader().discover("tests", top_level_dir=".")))
bad = [i for i in ids if "_FailedTest" in i or "ModuleImportFailure" in i]
print("\n".join(ids)); print(f"count={len(ids)} import_failures={bad}", file=sys.stderr)'
W="${TMPDIR:-/tmp}/lr_list.$$"; mkdir -p "$W"
git archive "$1" H2S-NN/src H2S-NN/tests H2S-NN/scripts H2S-NN/configs H2S-NN/notebooks H2S-NN/folds docs | tar -x -C "$W"
(cd "$W/H2S-NN" && python -c "$LIST") > docs/legacy_evidence/test_ids_$2.txt
(cd H2S-NN && python -c "$LIST") > docs/legacy_evidence/test_ids_worktree.txt
rm -rf "$W"
