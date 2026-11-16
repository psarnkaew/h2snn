# Part 4, H1 — stopped before any change (card error: a required file is outside the card's list)

Decision D1-b makes `sparse_tools` a vendor through a `vendor/VENDORS.tsv` row of kind `clone`. The existing test
`tests/test_vendor_manifest.py:77-81` (`VendorIgnoreTests`) requires every `kind=clone` row's folder to be git-ignored
(`git check-ignore --no-index vendor/<name>/ignore_probe.txt`), and `H2S-NN/.gitignore:90-106` lists each clone folder by
name; `vendor/sparse_tools/` is not covered (`git check-ignore -v H2S-NN/vendor/sparse_tools/x` returns nothing). Adding the
row therefore needs one more line, `/vendor/sparse_tools/`, in `H2S-NN/.gitignore`, which is not in card 1d's file list
("Nothing else"). Per `docs/working_preferences.md` (a card that is wrong makes the agent stop and report), H1 is not started:
no VENDORS.tsv row, no clone, no build, no adapter change; `h2snn-b` untouched.

What H1 would do once the line is allowed (prepared from H0, nothing written): the clone row (`https://github.com/Yangxc13/sparse_tools`,
`0596f00b9f86b8f46abe9280946e8a8f999f5628`, kind `clone`), `scripts/setup/build_sparse_tools.sh` (in-place
`build_ext --inplace` with `h2snn-b`'s torch, pip freeze before/after), the stub-`sortednp` import, the label part of
their ogbn-mag code, and the four tests of the card (the diagonal test exact on a power-of-two-degree fixture, since the
kernels are float32 only: `spspmm_diag_cpu.cpp:8`).
