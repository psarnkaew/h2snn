# Short interactive work on the lab GPU cluster

The login node `login01` is for git, editing, submitting and monitoring only: never run tests,
training, `pip`/`conda install` or preprocessing there. For short checks, open a shell on a
compute node:

```bash
srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00 --pty bash
cd /mnt/data/peeradon/h2snn/H2S-NN
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn
python -m unittest -v tests.test_v2_model
```

- Every job sets `--mem` and `--time` (at most 72 h; longer work needs checkpoints).
- A GPU type can be requested: `--gres=gpu:2080ti:1` (11 GB), `titan_rtx` (24 GB), `3090`
  (24 GB), `a6000` (48 GB).
- Use the author's Miniforge (`~/miniforge3`), never `/opt/conda`, and never mix the two in one shell.
- Work from the repository (`/mnt/data/peeradon/h2snn/H2S-NN`), never from `$HOME`. `/tmp` on a
  compute node is fast scratch that is deleted when the job ends.
- For the full suite use `sbatch scripts/slurm/run_tests.sbatch` (log in `logs/slurm-<jobid>.out`).
