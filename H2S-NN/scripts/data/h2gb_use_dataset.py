"""Download / preprocess all H2GB datasets (except RCDD) into H2S-NN/data.

Run from H2S-NN:
    python scripts/data/h2gb_use_dataset.py                 # all datasets
    python scripts/data/h2gb_use_dataset.py oag-cs Pokec    # only some

(Part M3b B1: the author's vendor/H2GB/examples/use_dataset.py, moved here; the only changes are the
usage lines above and H2GB_ROOT/DATA_DIR below, which the move requires.)

Each dataset is loaded in its own subprocess (H2GB's global cfg is not
reusable within one process). A failure in one dataset does not stop the rest.
"""
import os
import subprocess
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))           # H2S-NN
H2GB_ROOT = os.path.join(REPO, 'vendor', 'H2GB')        # .../vendor/H2GB
DATA_DIR = os.path.join(REPO, 'data', 'NC')             # H2S-NN/data/NC (as before the move)

# dataset -> plain config (no rand_split / posenc / data_test dir), so all
# models share the same official split.
DATASETS = {
    #'ogbn-mag': 'ogbn-mag/ogbn-mag-MLP.yaml',
    #'mag-year': 'mag-year/mag-year-MLP.yaml',
    'oag-cs': 'oag-cs/oag-cs-MLP.yaml',
    'oag-chem': 'oag-chem/oag-chem-MLP.yaml',
    'oag-eng': 'oag-eng/oag-eng-MLP.yaml',
    'IEEE-CIS': 'IEEE-CIS/IEEE-CIS-MLP.yaml',
    'PDNS': 'PDNS/PDNS-MLP.yaml',            
    'Pokec': 'Pokec/Pokec-MLP.yaml',
}


def load_one(cfg_file):
    # H2GB is not pip-installed; running the file directly only puts examples/ on
    # sys.path, so add the H2GB repo root (the folder that contains the H2GB package).
    if H2GB_ROOT not in sys.path:
        sys.path.insert(0, H2GB_ROOT)
    import torch
    from H2GB.graphgym.config import cfg, set_cfg, load_cfg
    from H2GB.graphgym.loader import create_dataset
    args = argparse.Namespace(
        cfg_file=cfg_file,
        opts=['wandb.use', 'False', 'dataset.dir', DATA_DIR,
              'out_dir', os.path.join(DATA_DIR, '_results')])
    set_cfg(cfg)
    load_cfg(cfg, args)
    if cfg.device == 'auto':
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(cfg.num_threads)

    dataset = create_dataset()
    print(dataset)
    print(dataset[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='*', help='subset of: ' + ', '.join(DATASETS))
    parser.add_argument('--cfg', dest='cfg_file', default=None,
                        help='(internal) load a single config in this process')
    a = parser.parse_args()

    if a.cfg_file:
        load_one(a.cfg_file)
        return

    names = a.names or list(DATASETS)
    os.makedirs(DATA_DIR, exist_ok=True)
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')  # torch>=2.6 workaround
    status = {}
    for name in names:
        cfg_file = os.path.join(H2GB_ROOT, 'configs', DATASETS[name])
        print(f'\n===== {name}  ({DATASETS[name]}) -> {DATA_DIR} =====', flush=True)
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), '--cfg', cfg_file],
                             env=env, cwd=H2GB_ROOT)
        status[name] = 'OK' if rc == 0 else f'FAILED (exit {rc})'

    print('\n===== summary =====')
    for k, v in status.items():
        print(f'{k:10s} {v}')


if __name__ == '__main__':
    main()
