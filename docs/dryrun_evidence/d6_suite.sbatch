#!/bin/bash
#SBATCH --job-name=dryrun-d6-suite
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/dryrun_scratch/logs/dryrun-d6-suite_%j.out
# Card 2 D6.9: the full H2S-NN suite once on one node with the pinned kernel environment E (set by d5_suite.sh).
# Submit from /mnt/data/peeradon/h2snn-dryrun/H2S-NN; unittest's result lines go to d6_suite.txt. 16G host: the rule
# "recorded need + 30 %, at least 16G" (the suite's synthetic tests); no A6000.
cd /mnt/data/peeradon/h2snn-dryrun/H2S-NN
bash ../docs/dryrun_evidence/d5_suite.sh > ../docs/dryrun_evidence/d6_suite.txt 2>&1
