# Card 25: NSD and CSNN, `orth: euler` only with `d` ≤ 3 (spec 1.34 AH3)

The model session wrote this card on 2026-11-29, at the author's request. Ops found the failures.

- **Base:** `v3b` `fe25277`. **Branch:** `card25`.
- **Report:** `docs/H2S-NN_card25_report.md`.
- **Allowed files:**
  - `src/baselines/adapters/nsd.py`
  - `src/baselines/adapters/csnn.py`
  - one new test file
  - the report
- `vendor/` is unchanged.

## Why
- **The vendor code.** Both vendors build the restriction maps with `Orthogonal(d, orthogonal_map)`. Its Euler branch exists only for d = 2, 3:
  - NSD: `vendor/neural-sheaf-diffusion/models/orthogonal.py:70`, built in `laplacian_builders.py:177`, which is BundleSheaf only.
  - CSNN: `vendor/csnn/models/orthogonal.py:74`, built in `CSNNconv.py:30` and `DenseCSNN.py:28`.
  - The branch is `assert 2 <= self.d <= 3`. It fires at the first forward.
- **Our search spaces sample the two axes independently:**
  - NSD: `d` ∈ {2, 3, 4, 5} × `orth` ∈ {householder, cayley, euler} (`nsd.py:253-256`).
  - CSNN: `d` ∈ {2, 3, 4} × the same `orth` values (`csnn.py:202-205`).
- **So every Euler draw with d > 3 is a failed trial that uses up one of the 40.** Seen so far:
  - CSNN oag-cs (attempt 1: `euler`, d = 4);
  - NSD LP H-Pokec U–U (trial 1).
  - Ops record: `docs/ops_evidence/plan_change_2026-11-24.md` §13.

## Fix
1. In each adapter, at the single point where the sampled parameters become the vendor's `args` (NSD `_vendor_args`; the CSNN equivalent near `csnn.py:107`):
   - if `orth == "euler"` and `d > 3`, pass `orth = "householder"` (the vendor default) to the vendor;
   - in every other case, pass the sampled value unchanged.
2. Add `orth_effective` to the trial and final-seed records, next to the sampled `orth`.
3. Every path that builds vendor args must go through that one point: NSD NC (HGB and H2GB), LP (including large LP) and GC; CSNN likewise. Check there is no second copy of the args dict.
4. **Do not change `search_space()`.** Running W&B sweeps must resume with the same sweep config and fingerprint.
5. Final seeds re-apply the same mapping to the stored `sweep_params`, so a selected trial and its seeds agree.
6. **Required property:** every configuration that ran before this card is bit-identical, because the changed configurations always crashed. Say in the report how this was checked (reading the code paths is enough).

## Tests (only these; minimal)
One new file, `tests/test_baselines_sheaf_euler_d.py`:
1. **Mapping test, for both adapters.** `euler` with `d = 4` gives `householder` in the vendor args and `orth_effective = "householder"`. `euler` with `d = 3` stays `euler`.
2. **NSD forward test.** One NSD BundleSheaf forward on a tiny graph with `euler`, `d = 4` runs without an AssertionError.

Run the full suite once on the lab before reporting (`--mem` and `--time` set).

## Report
- the diff summary;
- where the mapping sits, and the list of paths that reach it;
- the bit-identity argument;
- the test output;
- the full-suite count.

## Ops after merge (not part of this card)
- The author fast-forwards `v3b`.
- Ops then switches the code checkout of every **unfinished** NSD and CSNN tuning cell to the new `v3b` at its next leg. This includes the H-Pokec reruns in `cells_tuning_b2_hpokec_r1128.txt`, which run on `b0f13fb`.
  - This is allowed mid-sweep, because only configurations that used to crash change.
- Trials that already failed this way stay failed and are reported with a note. They are not rerun.
