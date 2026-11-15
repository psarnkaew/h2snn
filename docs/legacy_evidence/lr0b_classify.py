"""LR-0b: per-test-function class (delete / adapt / keep) for the inventory, from the scan TSV plus the
decisions below (read from the test bodies on login01). Prints a markdown table. Run from docs/legacy_evidence/:
    python lr0b_classify.py lr0b_test_scan.tsv > lr0b_classes.md
Rules for rows without an explicit decision: a test with legacy names in a file listed in DELETE_FILES is
delete; a test that builds a config/model without `anchored` is "adapt (default)": LR-2 switches the default
to the anchored v3-B model and the test runs on it with unchanged assertions (verified by the LR-2 run); every
other test is keep.
"""

from __future__ import annotations

import csv
import sys

D, A, K = "delete", "adapt", "keep"
# file -> default class of its tests (whole file tests deleted code); explicit entries override it
DELETE_FILES = {"test_cooperative.py", "test_pair_attention.py"}
EXPLICIT: dict[tuple[str, str], tuple[str, str]] = {
    # test_cooperative.py: D10 gates, pair attention, v2 operator theory, analyze_kernel, v2 golden
    # (whole file delete by DELETE_FILES)
    # test_filter_bank.py
    ("test_filter_bank.py", "test_context_off_builds_the_d11_parameter_shapes"): (A, "geometry_context false is a v3-B config; gate-shape assertions -> v3-B counterpart"),
    ("test_filter_bank.py", "test_TG7_neighbour_reaches_maps_and_gates_only_with_context"): (A, "context reaching the maps is kept; the gate half -> v3-B counterpart (g / maps)"),
    ("test_filter_bank.py", "test_TG1_spectrum_of_A_is_real_and_in_0_2_max_gb"): (D, "spectrum of the gated v2 operator A"),
    ("test_filter_bank.py", "test_TG2_gain_in_metric_M_is_at_most_one_with_control"): (D, "filter bank"),
    ("test_filter_bank.py", "test_TG3_cooperation_through_the_filter_exact_zero_blocks"): (D, "filter bank + gates"),
    ("test_filter_bank.py", "test_TG5_lemma_N_diagonal_blocks_are_identity"): (D, "v2 operator lemma"),
    ("test_filter_bank.py", "test_TG6_matrix_free_filter_equals_the_dense_reference"): (D, "filter bank"),
    ("test_filter_bank.py", "test_gain_diagnostic_matches_the_exact_M_gain"): (D, "filter bank diagnostic"),
    ("test_filter_bank.py", "test_TG4_d11_initialization_equals_first_order"): (D, "filter bank vs first_order"),
    ("test_filter_bank.py", "test_TG8_gradients_reach_every_V_k_gates_gamma_and_map_mlps"): (D, "filter bank, gates, gamma"),
    ("test_filter_bank.py", "test_eta_only_under_first_order_and_V_in_the_normal_group"): (D, "eta / filter bank"),
    ("test_filter_bank.py", "test_operator_diagnostics_under_the_filter_bank"): (D, "filter-bank diagnostics"),
    ("test_filter_bank.py", "test_norm_sum_diagnostic_casts_the_blocks_to_float64"): (D, "filter-bank diagnostics"),
    ("test_filter_bank.py", "test_switches_are_never_tuned"): (A, "geometry_context/encoder never tuned: encoder subcase removed (deleted key), geometry_context stays"),
    # test_gap_ladder.py: identity maps deleted (B1); feature normalization kept (G0b)
    ("test_gap_ladder.py", "test_identity_maps_give_the_normalized_graph_laplacian"): (D, "identity_maps (B1)"),
    ("test_gap_ladder.py", "test_control_learned_maps_are_not_the_graph_laplacian"): (D, "control of the identity-maps test (VertexPathSheafLaplacian)"),
    ("test_gap_ladder.py", "test_state_dict_unchanged_and_default_off"): (D, "identity_maps (B1)"),
    ("test_gap_ladder.py", "test_sheaf_free_logits_ignore_the_map_module"): (D, "identity_maps (B1)"),
    ("test_gap_ladder.py", "test_sheaf_free_still_propagates"): (D, "identity_maps (B1)"),
    ("test_gap_ladder.py", "test_config_rejects_invalid_combinations"): (D, "identity_maps combinations (B1)"),
    ("test_gap_ladder.py", "test_identity_maps_rejected_in_tuning_base_config"): (D, "identity_maps (B1); fail-closed test replaces it"),
    ("test_gap_ladder.py", "test_diagnostics_rejected_in_search_spaces"): (A, "identity_maps subcase removed (deleted key); kept diagnostics stay"),
    # test_task_heads.py
    ("test_task_heads.py", "test_outputs_and_gradients_are_bit_identical_to_the_pre_removal_capture"): (D, "v2 golden (captures the v2 model); replaced by tests/test_v3b_golden.py"),
    ("test_task_heads.py", "test_each_decoder_trains_one_step_and_reaches_the_encoder"): (A, "anchored config; eta/filter_norm_sum/gamma assertions -> tau / solver counterparts"),
    ("test_task_heads.py", "test_one_training_step_is_finite"): (A, "anchored config; filter_norm_sum assertion -> v3-B counterpart"),
    # test_v3b_model.py
    ("test_v3b_model.py", "test_T14_every_arm_trains_and_gradients_reach_the_right_groups"): (A, "asserts filter_bank / theta_eta / gates are None under anchored: attributes deleted -> asserted absent"),
    ("test_v3b_model.py", "test_T14_config_keys_are_validated_and_never_tuned"): (A, "legacy-default and legacy-value subcases -> fail-closed"),
    # test_v2_model.py
    ("test_v2_model.py", "test_T12a_per_side_ablation_keeps_two_scales"): (D, "per_side_scale"),
    ("test_v2_model.py", "test_T12b_negative_control_per_side_scales_break_orthogonality"): (D, "per_side_scale"),
    ("test_v2_model.py", "test_T14_node_rescaled_per_side_scales_give_the_same_normalized_delta0"): (D, "per_side_scale + VertexPathSheafLaplacian"),
    ("test_v2_model.py", "test_T2_path_operator_equals_dense_kron_reduced_reference"): (D, "VertexPathSheafLaplacian"),
    ("test_v2_model.py", "test_T3_normalized_operator_is_self_adjoint_and_psd"): (D, "VertexPathSheafLaplacian"),
    ("test_v2_model.py", "test_D_is_the_per_incidence_degree_of_the_operators_own_terms"): (D, "VertexPathSheafLaplacian"),
    ("test_v2_model.py", "test_T5_normalized_spectrum_is_bounded_by_two"): (D, "VertexPathSheafLaplacian"),
    ("test_v2_model.py", "test_T7_cell_state_is_the_least_squares_minimizer"): (K, "PathCellTerm (kept)"),
    ("test_v2_model.py", "test_T4_sections_are_preserved_and_the_kernel_is_unchanged"): (D, "VertexPathSheafLaplacian"),
    ("test_v2_model.py", "test_T4_negative_control_free_maps_shrink_the_kernel"): (D, "FreeIncidenceOperator"),
    ("test_v2_model.py", "test_T6_path_term_reaches_the_far_endpoint_and_the_intermediate_node"): (D, "v2 operator / first_order"),
    ("test_v2_model.py", "test_T9_deleted_components_are_absent"): (A, "asserts v1 components absent; reads build_operator -> v3-B forward"),
    ("test_v2_model.py", "test_T10_gradients_reach_every_learned_component"): (A, "anchored; theta_eta/theta_gamma/V_k -> theta_tau"),
    ("test_v2_model.py", "test_gate_initialization_is_near_one_half"): (A, "exp weight init (kept); sigmoid_gamma subcase removed"),
    ("test_v2_model.py", "test_T13_dense_and_factorized_families_match_on_materialized_maps"): (A, "per_side_scale subcase removed; map families kept"),
    ("test_v2_model.py", "test_trainer_logs_eta_gamma_uses_scalar_group_and_checkpoints_reload"): (A, "anchored; logged eta/gamma -> tau; scalar group and reload kept"),
    ("test_v2_model.py", "test_T11_no_path_cells_makes_logits_independent_of_path_data"): (A, "ablation.no_path_cells (B1) -> arm no_path_cells, same assertion"),
    ("test_v2_model.py", "test_T11_uniform_gate_gives_w_identically_one"): (A, "path_weight_input uniform (kept) via build_operator -> v3-B weights"),
    ("test_v2_model.py", "test_T11_endpoint_edges_ignore_intermediate_nodes_in_the_path_term"): (D, "endpoint_edges"),
    ("test_v2_model.py", "test_T11_holonomy_edges_equal_the_dense_closure_edge"): (D, "holonomy_edges switch (v2); closure_only arm covered by sheaf_energy tests"),
    ("test_v2_model.py", "test_T11_free_maps_are_orthogonal_and_per_incidence"): (D, "free_path_maps / FreeIncidenceOperator"),
    ("test_v2_model.py", "test_T11_per_side_scales_reach_transports_and_degree"): (D, "per_side_scale"),
    ("test_v2_model.py", "test_T11_h0_only_readout_ignores_the_graph"): (A, "h0_only_readout (kept); anchored config"),
    ("test_v2_model.py", "test_every_ablation_trains_one_step"): (A, "kept ablations stay; deleted ones removed"),
    ("test_v2_model.py", "test_defaults_are_the_full_model"): (A, "defaults become v3-B (LR-2)"),
    ("test_v2_model.py", "test_unknown_keys_fail_loudly"): (A, "no_path_cells example -> another unknown key"),
    ("test_v2_model.py", "test_invalid_values_and_conflicting_ablations_are_rejected"): (A, "deleted-setting subcases -> fail-closed tests"),
    ("test_v2_model.py", "test_a_endpoint_edge_maps_read_the_endpoints_context_not_intermediates"): (D, "endpoint_edges"),
    ("test_v2_model.py", "test_b_g_reads_h0_only_unless_the_appendix_switch"): (A, "path_weight_context (kept); endpoint_edges/free_path_maps subcases removed"),
    ("test_v2_model.py", "test_c_free_maps_read_every_instance_node_context_through_the_latent"): (D, "free_path_maps"),
    ("test_v2_model.py", "test_d_pair_attention_cell_scores_read_the_context"): (D, "pair_attention"),
    ("test_v2_model.py", "test_context_off_keeps_one_encoder_per_arm"): (A, "endpoint_edges/free_path_maps subcases removed; kept arms stay"),
    # test_wisp.py
    ("test_wisp.py", "test_TH9_evaluation_draw_ignores_model_and_ablation_settings"): (A, "deleted settings in the variation list -> kept ones"),
    ("test_wisp.py", "test_wisp_config_rejections"): (A, "uniform_instance subcase -> fail-closed"),
    ("test_wisp.py", "test_TH5_props_2_3_5_with_exp_weights"): (D, "VertexPathSheafLaplacian (v2 operator properties)"),
    ("test_wisp.py", "test_TH5_kernel_preserved_with_exp_weights"): (D, "VertexPathSheafLaplacian"),
    ("test_wisp.py", "test_TH6_exp_has_no_gamma_and_b_is_a_scalar_parameter"): (A, "exp weight (kept); gamma/sigmoid_gamma subcases removed"),
    ("test_wisp.py", "test_saved_configs_round_trip_under_both_path_weights"): (A, "sigmoid_gamma subcase removed"),
    ("test_wisp.py", "test_TH6_exp_operator_has_no_gamma_multiplier"): (D, "VertexPathSheafLaplacian"),
    ("test_wisp.py", "test_TH11_partial_pair_form_matches_its_kept_pair_energy"): (D, "delta_loss (v2 operator; spec §6 not reported)"),
    ("test_wisp.py", "test_TH11_delta_loss_of_an_absent_schema_is_zero"): (D, "delta_loss"),
    ("test_wisp.py", "test_direct_construction_is_validated"): (A, "sigmoid_gamma/gamma_init subcases removed"),
    ("test_wisp.py", "test_every_never_tuned_wisp_key_is_rejected"): (A, "path_weight subcase (sigmoid_gamma) -> fail-closed"),
    ("test_wisp.py", "test_path_weight_context_is_an_appendix_arm_never_tuned"): (A, "no_path_cells subcase -> arm"),
    ("test_wisp.py", "test_path_weight_context_is_the_ladder_appendix_pair_only"): (A, "deleted-arm subcases removed"),
    # test_paths_and_facts.py
    ("test_paths_and_facts.py", "test_T13_lambda_max_at_most_two"): (D, "VertexPathSheafLaplacian lambda_max"),
    ("test_paths_and_facts.py", "test_T16_uniform_instance_is_uniform_over_instances"): (D, "uniform_instance"),
    ("test_paths_and_facts.py", "test_T16_uniform_instance_refuses_long_schemas"): (D, "uniform_instance"),
    ("test_paths_and_facts.py", "test_config_rules"): (A, "uniform_instance subcase removed"),
    ("test_paths_and_facts.py", "test_T11_config"): (A, "uniform_instance subcase removed"),
    # test_search_spaces.py, test_wandb_and_runs.py
    ("test_search_spaces.py", "test_K_layer_dropout_and_legacy_keys_are_refused"): (K, "refusal of legacy keys stays (they raise even earlier now)"),
    ("test_wandb_and_runs.py", "test_wandb_sweep_config_and_tuning_selection"): (K, "asserts legacy keys absent from the sweep"),
    # false positives of the scan (local helper named build_operator / propagate)
    ("test_anchored_solve.py", "test_T9_pcg_matches_the_dense_solve"): (K, "local helper build_operator (v3-B)"),
    ("test_v3b_theory.py", "test_U1_unrolled_gradient_equals_the_implicit_gradient"): (K, "local helper (v3-B)"),
    ("test_baselines.py", "test_link_decoders_score_and_backpropagate"): (K, "baseline head; 'propagate' is unrelated"),
}


def main() -> None:
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
    print("| File | Test | Class | Note |\n|---|---|---|---|")
    counts: dict[str, int] = {}
    for r in rows:
        key = (r["file"], r["test"])
        if key in EXPLICIT:
            cls, note = EXPLICIT[key]
        elif r["file"] in DELETE_FILES:
            cls, note = D, f"whole file tests deleted code ({r['legacy_names'] or 'v2 operator / gates'})"
        elif r["legacy_names"] and r["legacy_names"] in ('"K"', "'K'"):
            cls, note = A, "sets model.K (deleted key) on a model test: K dropped, anchored config"
        elif r["builds"] == "1" and r["anchored_in_body"] == "0":
            cls, note = A, "builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run)"
        elif r["legacy_names"]:
            cls, note = A, f"names {r['legacy_names']}: checked in LR-2/LR-3"
        else:
            continue  # keep: not listed (every unlisted test is keep)
        counts[cls] = counts.get(cls, 0) + 1
        print(f"| {r['file']} | {r['class']}.{r['test']} | {cls} | {note} |")
    print(f"\nListed: {counts}; every other test of `tests/` ({len(rows) - sum(counts.values())}) is keep.")


if __name__ == "__main__":
    main()
