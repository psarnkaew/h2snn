#!/bin/bash
# Card 2 D3.4: an agent killed mid-sweep leaves the sweep usable. The launcher (and its trial worker) run in their own
# process group, which is SIGKILLed after K seconds (as SLURM kills a job); the same command with --resume then
# finishes the sweep. Run from H2S-NN/ in a SLURM allocation.
set -u
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
export WANDB_SILENT=true
D=${D:-/mnt/data/peeradon/dryrun_scratch/d3c}; K=${K:-25}
S=$D/baselines/tuning_state/gcn_node_classification_acm/tuning_state_run0.json
echo "head=$(git rev-parse HEAD) host=$(hostname) job=$SLURM_JOB_ID step=${SLURM_STEP_ID:-}"
st() { python -c "import json; s=json.load(open('$S')); print('state: sweep', s['sweep_id'], 'attempted', s['attempted_trials'], 'ok', s['successful_trials'], 'failed', s.get('failed_trials'))"; }
CMD=(python scripts/run_baseline_tuning.py --task node_classification --models gcn --datasets acm --num_runs 1
     --n_trials 5 --dry_run --output_dir "$D/baselines" --wandb_project h2snn-dryrun --resume)
st
setsid "${CMD[@]}" > "$D/gcn_killed.log" 2>&1 &
PID=$!
sleep "$K"
kill -9 -- "-$PID" 2>/dev/null && echo "killed process group $PID after ${K}s"
wait "$PID" 2>/dev/null; echo "killed agent exit=$?"
st
"${CMD[@]}" > "$D/gcn_resumed.log" 2>&1; echo "resumed agent exit=$?"
grep -h "^W&B run" "$D/gcn_resumed.log"
st
