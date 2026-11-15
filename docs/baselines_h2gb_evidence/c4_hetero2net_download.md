# Hetero²Net on ogbn-mag / mag-year: the MetaPath2Vec embedding file (author download)

Author decision 2 (2026-11-16): Hetero²Net's MAG branch uses PyG's pretrained MetaPath2Vec embeddings for author,
institution and field (`vendor/Hetero2Net/hetero2net/dataset.py:54-57`: `OGB_MAG(root, preprocess='metapath2vec')`).
Read in PyG 2.8.0 of `h2snn-b` (`torch_geometric/datasets/ogb_mag.py`), VERIFIED-CODE:

| Item | Value |
|---|---|
| URL | `https://data.pyg.org/datasets/mag_metapath2vec_emb.zip` (`ogb_mag.py:51-53`) |
| File to download | `mag_metapath2vec_emb.zip` |
| What PyG reads from it | `mag_metapath2vec_emb.pt` (raw file name `mag_{preprocess}_emb.pt`, `ogb_mag.py:94`), a dict node type → tensor; every key except `paper` becomes that type's `x` (`ogb_mag.py:150-153`) |
| Where to place it | `/mnt/data/peeradon/h2snn/H2S-NN/data/NC/mag/hetero2net/mag_metapath2vec_emb.zip` (new folder `hetero2net` beside H2GB's `ogbn-mag/`, `mag-year/` and OGB's `ogbn_mag/`; the zip as downloaded, not extracted) |

Suggested command (the author, on the login node): 
`mkdir -p /mnt/data/peeradon/h2snn/H2S-NN/data/NC/mag/hetero2net && wget -O /mnt/data/peeradon/h2snn/H2S-NN/data/NC/mag/hetero2net/mag_metapath2vec_emb.zip https://data.pyg.org/datasets/mag_metapath2vec_emb.zip`

The Hetero²Net H2GB adapter was not wired in this card (C4 report); when it is, its loader reads
`mag_metapath2vec_emb.pt` from inside the zip (no extraction under `data/`), records the zip's sha256 at first use and
refuses a different file. Until the file is there the two cells say they wait for it. Not checked here (UNVERIFIED):
that PyG's author/institution/field row order equals OGB's node ids as H2GB's graph uses them (paper features are
bit-identical, Part M3b B7); the loader's first use should compare the row counts with H2GB's types (1,134,649 /
8,740 / 59,965).
