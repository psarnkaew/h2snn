# LR-2 stop: `map_family: dense` is kept by the card but exists only in the v2 model

Branch `legacy-removal`, pushed head `9e9087e` (LR-1 committed). LR-2 is **not committed**: the working tree holds the LR-2 default
change (`src/configs/config.py`: `propagation` default `anchored` in the field, in `from_dict` and in `reject_never_tuned_settings`;
`ablation.coupling` default `symmetric`) and the new `tests/test_legacy_removal.py`. Probe: the full suite with these defaults, 16741 step
on g2n01 → `docs/legacy_evidence/lr2_probe_suite.txt` (`Ran 713 tests — FAILED (failures=13, errors=90)`).

## The conflict (card stop: "a test of kept code that would have to be deleted"; the card is wrong on this point)

- The card's Rules list `map_family` factorized **and dense** among the code that "any v3-B configuration reaches" (kept).
- The anchored solve needs the factorized parameterisation: `_encode_anchored` passes `maps.src.reflectors`, `maps.src.rotations`,
  `maps.tgt.reflectors`, `maps.tgt.rotations` to the implicit solve (`src/models/h2snn_model.py:1339`, the `params` tuple) and rebuilds
  `FactorizedMapOperator`s from them inside `build`; `DenseMapOperator` has no `reflectors` →
  `AttributeError: 'DenseMapOperator' object has no attribute 'reflectors'`. `TrainConfig` does not refuse `anchored` + `dense`; the model
  fails at the first forward. So `map_family: dense` runs only in the v2 propagation (as `identity_maps`, B1).
- LR-1 left the dense configuration out for this reason (`lr1_capture_g2n01.txt`: `SKIPPED nc_map_family_dense`).
- Kept-code tests of the dense family that cannot move to an anchored configuration with unchanged assertions (probe, LR-2 defaults):
  - `tests/test_v2_model.py::V2ModelTests::test_T13_dense_and_factorized_families_match_on_materialized_maps` (whole-model forward with
    materialised dense maps vs factorized: equal outputs);
  - `tests/test_v2_model.py::V2ModelTests::test_T8_no_dense_map_tensor_in_v2_forward_or_backward` (its control builds the dense family
    and runs forward/backward);
  - the dense subcase of `tests/test_v2_model.py::AblationTests::test_every_ablation_trains_one_step`.
  The dense *operator* tests (`DenseOperatorTests`: closed-form inverse, path transport accepts dense operators) do not build a model and
  pass.

## Options for the author (design session)

| Option | Effect |
|---|---|
| (a) delete `map_family: dense` with the v2 propagation (as B1 = (c) for `identity_maps`): `DenseConformalNodeToEdgeSheaf` / the model branch; the value raises "removed by the legacy-removal card"; T13, T8's control and the dense subcase deleted; `DenseMapOperator` itself stays if kept tests of `map_operators` use it | consistent with A1; the dense family was D1's comparison family under v2 |
| (b) give the anchored solve a dense form (pass the dense maps' tensors through the implicit solve) | a new v3-B code path (no existing v3-B output changes); spec/card |
| (c) keep the v2 propagation for the dense family | most of A1's removal does not happen |

## Everything else in the probe is as LR-0b classified it

Deleted-code tests (gates, filter bank, pair attention, identity maps, the v2 operator, `delta_loss`, the v2 golden) fail on the v3-B
default, as expected; they get the explicit v2 values (card LR-2) or are deleted in LR-3. Kept-code tests that fail only through v2-only
quantities or the v2 default (adapt under B3): `test_task_heads` LP/GC heads (`filter_norm_sum`), `test_filter_bank` context tests (gates),
`test_v2_model` T10/T11/defaults/context/trainer logs, `test_v3b_model` T14 (legacy default), `test_wisp` TH6/TH9/round trip,
`test_partO_literature_row::test_builder_refuses_other_inputs` and `test_search_spaces::test_the_anchored_model_can_be_tuned` (their
"non-anchored" input relied on the default: it must name `propagation: chebyshev`, which raises after LR-3).
