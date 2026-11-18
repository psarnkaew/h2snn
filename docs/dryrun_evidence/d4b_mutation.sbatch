#!/bin/bash
# Card 2 D4b: the mutation check (d4b_mutation.py) on the CPU. Submit from H2S-NN/.
#SBATCH --job-name=dryrun-d4b-mutation
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/dryrun_scratch/logs/%x_%j.out
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "head=$(git rev-parse HEAD) host=$(hostname) job=$SLURM_JOB_ID"
python ../docs/dryrun_evidence/d4b_mutation.py 2>&1 | grep -vE "Warning|warn|^  "
echo "exit=$?"
