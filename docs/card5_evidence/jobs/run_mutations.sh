#!/bin/bash
# Card 5 U5 mutations, in this worktree, CPU only, env E. Each mutation edits one committed file, runs the tests that
# must catch it, then restores the file from a copy and checks it equals HEAD (git diff --quiet).
#   M1 (B1a): training epochs score the whole validation split instead of the subset -> T1 must fail.
#   M2 (F3):  knockout ignores the node filter and removes every cell -> T6 must fail.
set -uo pipefail
cd /mnt/data/peeradon/h2snn-ablprep/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES=
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"
git diff --quiet -- src tests || { echo "worktree not clean"; exit 2; }
backup=$(mktemp -d)
mutate() {  # file old new
    cp "$1" "$backup/$(basename "$1")"
    python - "$1" "$2" "$3" <<'PY' || exit 2
import sys
path, old, new = sys.argv[1:]
text = open(path, encoding="utf-8").read()
assert text.count(old) == 1, f"mutation anchor not unique in {path}"
open(path, "w", encoding="utf-8").write(text.replace(old, new))
PY
    git diff --stat -- "$1"
}
restore() { cp "$backup/$(basename "$1")" "$1"; git diff --quiet -- "$1" && echo "restored $1 (= HEAD)" || { echo "RESTORE FAILED $1"; exit 3; }; }
T=tests.test_card5_protocol_hooks

echo "=== M1: subset_mask -> whole validation split during training"
F=src/trainers/minibatch_trainer.py
mutate $F "subset_metrics = self.evaluate_split(eval_graph, subset_mask)" \
          "subset_metrics = self.evaluate_split(eval_graph, eval_graph.target_val_mask)"
python -m unittest -v $T.B1aTests 2>&1 | grep -E "^test_|^(FAIL|ERROR):|^Ran|^OK|^FAILED|AssertionError"
restore $F

echo "=== M2: knockout removes every cell (node filter ignored)"
F=src/models/h2snn_model.py
mutate $F "keep_masks.append(~torch.isin(ids, knock).any(dim=1))" \
          "keep_masks.append(torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device))"
python -m unittest -v $T.KnockoutTests 2>&1 | grep -E "^test_|^(FAIL|ERROR):|^Ran|^OK|^FAILED|AssertionError"
restore $F

git diff --quiet -- src tests && echo "worktree clean after mutations"
echo "end=$(date -Is)"
