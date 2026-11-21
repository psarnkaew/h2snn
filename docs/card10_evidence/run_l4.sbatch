#!/bin/bash
# Card 10 L4: targeted tests in one CPU job (no --gres, 16G): every card 10 test, the TSUBAME script tests, and the
# tests that exercise the two launchers L2 changed; then every card 10 mutation; then the src/ check.
# The worktree is set up as a run checkout (guide section 3): vendor/ cloned by clone_vendors.sh, data -> lab data root.
#SBATCH --job-name=card10-l4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card10/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
# the pinned CPU kernel environment of scripts/slurm/run_tests.sbatch (bit-exact tests); W&B mode as there (unset)
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
unset WANDB_MODE
echo "head=$(git rev-parse HEAD) host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
git status --short --untracked-files=no
TESTS="tests.test_lab_slots tests.test_site_split tests.test_tsubame_types tests.test_failures_report tests.test_fit_check
tests.test_tsubame_slots tests.test_tsubame_cell_list tests.test_tsubame_collect tests.test_tsubame_collect_h2snn
tests.test_tsubame_final_seeds_gate tests.test_tsubame_guard tests.test_tsubame_h2snn_lines tests.test_tsubame_host_memory
tests.test_tsubame_phases tests.test_card7_tsubame tests.test_card8_phases
tests.test_baseline_final_seeds tests.test_b1a_validation_subset tests.test_baselines tests.test_card4_ablation_driver
tests.test_card4b_prefreeze tests.test_card5_protocol_hooks tests.test_card7_perturbation tests.test_dry_run_flag
tests.test_partM3a_decoder tests.test_partO_reload tests.test_search_spaces tests.test_sweep_batch_size
tests.test_paths_and_facts tests.test_wandb_logging tests.test_wandb_and_runs"
python -m unittest $TESTS; a=$?
m=0
for f in mutation_l1 mutation_l3 mutation_a21 mutation_a23 mutation_l2 mutation_a22; do
    python ../docs/card10_evidence/$f.py > /dev/null 2>&1; r=$?; echo "$f exit=$r (0 = caught)"; [ $r -eq 0 ] || m=1
done
echo "src diff vs 2d822a4: $(git diff --stat 2d822a4 -- src | wc -l) lines"
echo "tests=${a} mutations=${m} end=$(date -Is)"
