#!/bin/bash
# Card 2 D3: tuning launcher end to end (one dry-run trial per cell, W&B project h2snn-dryrun), seeds launchers on the
# trial's configuration, schema read-out and the evaluation-draw check on the ACM trial. Run from
# /mnt/data/peeradon/h2snn-dryrun/H2S-NN inside a SLURM allocation. Outputs (runs, checkpoints, logs with the trainers'
# console lines) stay in $D3 outside the repository; the summary lines this script prints carry no metric value.
set -euo pipefail
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
export WANDB_SILENT=true
D3=${D3:-/mnt/data/peeradon/dryrun_scratch/d3}
mkdir -p "$D3/logs"
echo "head=$(git rev-parse HEAD) host=$(hostname) job=${SLURM_JOB_ID} step=${SLURM_STEP_ID:-}"
git status --short -- scripts tests | sed 's/^/worktree: /'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
P="--wandb_project h2snn-dryrun"

step() {  # name, command...: run, keep the full log outside the repo, print the exit code only
  local name=$1; shift
  local start=$SECONDS
  if "$@" > "$D3/logs/$name.log" 2>&1; then rc=0; else rc=$?; fi
  echo "STEP $name exit=$rc seconds=$((SECONDS - start))"
}

# 1. one trial per cell through the launchers, exactly as the protocol (plain Bayes; the generated space is the
#    tested one), with --resume as run_sweep_agents.sbatch runs them
for ds in acm acm_pa mutag; do
  step "tune_h2snn_$ds" python scripts/run_h2snn_wandb_tuning.py --config "configs/experiments/${ds}_h2snn_v3b.yaml" \
    --n_trials 1 --dry_run --output_dir "$D3/h2snn" $P --resume
done
step tune_gcn_acm python scripts/run_baseline_tuning.py --task node_classification --models gcn --datasets acm \
  --num_runs 1 --n_trials 1 --dry_run --output_dir "$D3/baselines" $P --resume

# resume: the same command again finds the fingerprinted state (same sweep id, nothing left to run)
step resume_h2snn_acm python scripts/run_h2snn_wandb_tuning.py --config configs/experiments/acm_h2snn_v3b.yaml \
  --n_trials 1 --dry_run --output_dir "$D3/h2snn" $P --resume
step resume_gcn_acm_more python scripts/run_baseline_tuning.py --task node_classification --models gcn --datasets acm \
  --num_runs 1 --n_trials 2 --dry_run --output_dir "$D3/baselines" $P --resume

# 2. the trial's resolved configuration and best checkpoint -> seeds launchers (seed 0, one epoch) and the read-out
for ds in acm acm_pa mutag; do
  trial=$(ls -d "$D3/h2snn/${ds}_h2snn_v3b_tune_run0_"*/ | head -1)
  echo "TRIAL $ds $trial config=$(test -f "$trial/config.yaml" && echo yes) best=$(test -f "$trial/checkpoints/best.pt" && echo yes)"
  step "runs_$ds" python scripts/run_h2snn_runs.py --config "$trial/config.yaml" --num_runs 1 --base_seed 0 --dry_run \
    --exp_name "${ds}_seed0" --output_dir "$D3/runs" --use_wandb $P
  step "final_seeds_$ds" python scripts/run_h2snn_wandb_tuning.py --config "configs/experiments/${ds}_h2snn_v3b.yaml" \
    --final_seeds 0 --dry_run --output_dir "$D3/h2snn" $P
done
acm_trial=$(ls -d "$D3/h2snn/acm_h2snn_v3b_tune_run0_"*/ | head -1)
step schema_readout_acm python scripts/schema_readout.py --runs "$acm_trial" --out_dir "$D3/readout" --device cuda \
  --permutations 10

# 3. evaluation draw reproduces on the ACM trial
step eval_draw_acm python ../docs/dryrun_evidence/d3_eval_draw_check.py "$acm_trial" "$D3/eval_draw"
grep -h "^EVAL_DRAW" "$D3/logs/eval_draw_acm.log" || true
# file inventory (names only)
find "$D3" -maxdepth 3 \( -name 'tuning_state_*.json' -o -name 'best_tuned_*.json' -o -name 'config.yaml' -o -name 'best.pt' \
  -o -name 'final_seed*.json' -o -name 'schema_readout.md' \) | sed "s|$D3/||" | sort
