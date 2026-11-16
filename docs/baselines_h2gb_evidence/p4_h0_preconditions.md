# Part 4, H0 — preconditions (card `H2S-NN_baselines_h2gb_part4_card.md`)

Branch: `origin/v3b` = `72d5768c94e690e0090776d2ebdde05c5a23ac9e`; `origin/design-docs` contains `29250017dfcf0a32e716461a6787450f493bf161`;
`baselines-h2gb-4` created from `72d5768`, `git merge --no-ff 2925001` clean. Read: card 1d, the decisions-log row
"Card 1c review" (D1-b, D2), the part 3 report.

## The author's embedding file (VERIFIED-RUN, job 16741 step 178, `p4_h0_zip_sha256.txt`)
`H2S-NN/data/NC/mag/hetero2net/mag_metapath2vec_emb.zip`: 920,582,404 bytes (= the card), sha256
`00e34ce933d7bdc12bb7d35e6a23c1057e79053f0c679f1dc2e3295be3185dbf`. Card 1c recorded no sha256 (the file was absent; no
`.sha256` beside it), so there is nothing to compare; this is the first record. The loader writes this value beside the zip
at first use and refuses a different file (`_hetero2net_h2gb.read_mag_embeddings`).

## `sparse_tools` (commit `0596f00b9f86b8f46abe9280946e8a8f999f5628`; VERIFIED-CODE)
- `sparse_tools/__init__.py:9-11`: loads `_spspmm_diag_cpu` and `_spspmm_diag_unsym_cpu` from the package folder with
  `torch.ops.load_library`; `:13-14` import the diagonal functions; **`:24` `from .utils import SparseAdjList`**, the only
  route to `utils.py` (which imports `sortednp` at `utils.py:5`).
- `spspmm_diag.py:17-26` `spspmm_diag_sym_ABA(src)`: row weight 1/rowcount, column weight 1/colcount of `src`; kernel
  `csrc/cpu/spspmm_diag_cpu.cpp:48-85`: `value[r] = row_weight[r] * Σ_{e in row r} col_weight[col[e]]`, i.e.
  **diag((D_row⁻¹ A)(D_col⁻¹ Aᵀ))** for the 0/1 pattern A of `src` (counts per stored entry).
- `spspmm_diag.py:6-13` `spspmm_diag_sym_AAA(src)` (square, symmetric): weight 1/rowcount; kernel `:20-45`:
  `value[r] = w[r] * Σ_{e in row r} w[col[e]]`, i.e. **diag((D⁻¹ A)(D⁻¹ A))**.
- Both kernels compute in float32 only (`#define scalar_t float`, `spspmm_diag_cpu.cpp:8`). The H1 diagonal test therefore
  uses a fixture whose degrees are powers of two, so every weight and diagonal is exactly representable and the comparison
  with the dense float64 formula is exact (no tolerance).

## PSHGCN ogbn-mag label part (VERIFIED-CODE, PSHGCN `f87406c`)
- Diagonals (`processing.py:134-150`, repeated `:164-180`): `PFFP` = `spspmm_diag_sym_ABA(PF)` with `PF = adjs[3]` (P→F);
  `PPPP` = `spspmm_diag_sym_AAA(PP)` with `PP = adjs[2]` (P–P, symmetrised and diagonal-free, `:112-113`); `PAAP` =
  `spspmm_diag_sym_ABA(PA)` with `PA = adjs[1].t()` (P×A). Only `spspmm_diag_sym_ABA/AAA` are called; nothing from
  `sparse_tools.utils`.
- Labels (`main.py:96-101`): at stage 0 `label_onehot` is zeros with the **training rows** set to the one-hot labels
  (`main.py:101`); stage > 0 would use predictions (enhance stage; their default `--stage 1` runs stage 0 only).
- Propagation of `label_onehot` with `propagate(g, K)` (`main.py:103-108`), keys except `P` kept; for `PPPP`, `PAAP`, `PFFP`
  the self-effect `diag * label_onehot` is subtracted (`main.py:110-115`).
- Model: `label_keys = ['PPPP', 'PAAP', 'PP', 'PFFP']` (`main.py:126`).

## HGB Hetero²Net validation call (D2)
`src/baselines/adapters/hetero2net.py:255-261`: `eval_data` = copy with `eval_y_emb` (test labels hidden only);
`:312-323`: one forward on `eval_data`, validation and test metrics from it. Their code: validation on `train_data`
(validation + test hidden), test on `test_data` (`Hetero2Net/main.py:139-143, 217-219`).
