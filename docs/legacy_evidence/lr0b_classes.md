| File | Test | Class | Note |
|---|---|---|---|
| test_anchored_solve.py | PCGTests.test_T9_pcg_matches_the_dense_solve | keep | local helper build_operator (v3-B) |
| test_baselines.py | CampaignSweepTests.test_every_runnable_cell_builds_a_sweep | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_baselines.py | CampaignSweepTests.test_no_hyperband_option_remains | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_baselines.py | HeadTests.test_link_decoders_score_and_backpropagate | keep | baseline head; 'propagate' is unrelated |
| test_baselines.py | EpochReportingTests.test_report_key_matches_the_sweep_metric | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_baselines.py | SelectionMetricTests.test_sweeps_optimise_the_same_metric_training_selects_on | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_cooperative.py | SymmetricIsThePreviousModelTests.test_T1_symmetric_reproduces_the_golden_outputs_bit_for_bit | delete | whole file tests deleted code (v2_golden) |
| test_cooperative.py | SymmetricIsThePreviousModelTests.test_T1_control_the_default_model_is_not_the_golden_one | delete | whole file tests deleted code (broadcast_gate,cooperative,listen_gate,v2_golden) |
| test_cooperative.py | SymmetricIsThePreviousModelTests.test_modes_instantiate_the_right_heads | delete | whole file tests deleted code (K,broadcast_gate,cooperative,coupling_gate,diffuse,filter_bank,first_order,layer_dropout,listen_gate,listen_only) |
| test_cooperative.py | SymmetricIsThePreviousModelTests.test_gates_start_at_the_configured_init | delete | whole file tests deleted code (K,coupling_gate) |
| test_cooperative.py | GatedOperatorTests.test_T2_gated_step_equals_the_dense_reference | delete | whole file tests deleted code (K,cooperative,diffuse,filter_bank,first_order,layer_dropout,listen_only) |
| test_cooperative.py | GatedOperatorTests.test_T2_the_fixtures_mix_lengths_and_signs | delete | whole file tests deleted code (v2 operator / gates) |
| test_cooperative.py | SpectrumTests.test_T3_real_spectrum_and_bound_with_gates | delete | whole file tests deleted code (K,broadcast_gate,cooperative,diffuse,filter_bank,first_order,lambda_max,layer_dropout,listen_gate) |
| test_cooperative.py | SpectrumTests.test_T3_negative_control_csnn_composed_operator_is_complex | delete | whole file tests deleted code (cooperative) |
| test_cooperative.py | CooperativeBehaviourTests.test_T4_a_node_with_zero_listen_gate_is_unchanged_exactly | delete | whole file tests deleted code (K,cooperative,diffuse,filter_bank,first_order,layer_dropout,listen_gate) |
| test_cooperative.py | CooperativeBehaviourTests.test_T4_a_node_with_zero_broadcast_gate_influences_nobody | delete | whole file tests deleted code (broadcast_gate) |
| test_cooperative.py | CooperativeBehaviourTests.test_T4_unequal_gates_make_the_jacobian_blocks_asymmetric | delete | whole file tests deleted code (v2 operator / gates) |
| test_cooperative.py | FixedPointTests.test_T5_fixed_points_are_the_sections_seen_through_the_broadcast_gates | delete | whole file tests deleted code (K,broadcast_gate,cooperative,diffuse,filter_bank,first_order,layer_dropout) |
| test_cooperative.py | GateGradientTests.test_T6_gradients_reach_both_gate_heads | delete | whole file tests deleted code (K,broadcast_gate,cooperative,coupling_stats,listen_gate,listen_only) |
| test_cooperative.py | LearningFixTests.test_TF1_the_scalar_group_moves_eta_and_both_gate_heads | delete | whole file tests deleted code (K,.eta(),coupling_gate,first_order,layer_dropout) |
| test_cooperative.py | LearningFixTests.test_TF1_control_old_setting_does_not_move | delete | whole file tests deleted code (K,.eta(),coupling_gate,first_order,layer_dropout) |
| test_cooperative.py | LearningFixTests.test_TF1_group_contents_and_defaults | delete | whole file tests deleted code (K,broadcast_gate,cooperative,coupling_gate,first_order,listen_gate,sigmoid_gamma,theta_eta,theta_gamma) |
| test_cooperative.py | MetricGradientFlowTests.test_TF2_metric_gradient_identity | delete | whole file tests deleted code (K,cooperative,diffuse,filter_bank,first_order,layer_dropout) |
| test_cooperative.py | MetricGradientFlowTests.test_TF3_energy_never_increases_for_eta_up_to_one_and_does_for_eta_three | delete | whole file tests deleted code (v2 operator / gates) |
| test_cooperative.py | MetricGradientFlowTests.test_TF3b_prop4_the_d11_operator_factorizes_and_a_metric_exists | delete | whole file tests deleted code (v2 operator / gates) |
| test_cooperative.py | MetricGradientFlowTests.test_TF3b_control_free_per_direction_weights_have_no_metric | delete | whole file tests deleted code (v2 operator / gates) |
| test_cooperative.py | DiagnosticsTests.test_energy_parts_equal_the_dense_rayleigh_quotients | delete | whole file tests deleted code (v2 operator / gates) |
| test_cooperative.py | DiagnosticsTests.test_energy_trace_has_every_step_and_does_not_change_the_model_state | delete | whole file tests deleted code (K,energy_trace,first_order) |
| test_cooperative.py | DiagnosticsTests.test_the_trainer_logs_the_trace_and_the_gate_statistics | delete | whole file tests deleted code (K,eta_lambda_max,first_order,sigmoid_gamma) |
| test_cooperative.py | KernelScriptTests.test_lobpcg_eigenvalues_match_the_dense_spectrum | delete | whole file tests deleted code (K,analyze_kernel) |
| test_data.py | DataPipelineTests.test_repeated_kfold_config_validates_indices | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_factorized_maps.py | NumericalGuardTests.test_T9_zero_reflector_mlp_keeps_model_finite | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_factorized_maps.py | ModelIntegrationTests.test_cuda_float32_matches_cpu_float64_reference | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_failed_trials.py | NonFiniteGTests.test_non_finite_g_raises_before_exp | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_filter_bank.py | ContextModelTests.test_TG7_neighbour_reaches_maps_and_gates_only_with_context | adapt | context reaching the maps is kept; the gate half -> v3-B counterpart (g / maps) |
| test_filter_bank.py | ContextModelTests.test_context_off_builds_the_d11_parameter_shapes | adapt | geometry_context false is a v3-B config; gate-shape assertions -> v3-B counterpart |
| test_filter_bank.py | FilterBankOperatorTests.test_TG1_spectrum_of_A_is_real_and_in_0_2_max_gb | delete | spectrum of the gated v2 operator A |
| test_filter_bank.py | FilterBankOperatorTests.test_TG2_gain_in_metric_M_is_at_most_one_with_control | delete | filter bank |
| test_filter_bank.py | FilterBankOperatorTests.test_TG3_cooperation_through_the_filter_exact_zero_blocks | delete | filter bank + gates |
| test_filter_bank.py | FilterBankOperatorTests.test_TG5_lemma_N_diagonal_blocks_are_identity | delete | v2 operator lemma |
| test_filter_bank.py | FilterBankOperatorTests.test_TG6_matrix_free_filter_equals_the_dense_reference | delete | filter bank |
| test_filter_bank.py | FilterBankOperatorTests.test_gain_diagnostic_matches_the_exact_M_gain | delete | filter bank diagnostic |
| test_filter_bank.py | FilterBankModelTests.test_TG4_d11_initialization_equals_first_order | delete | filter bank vs first_order |
| test_filter_bank.py | FilterBankModelTests.test_TG8_gradients_reach_every_V_k_gates_gamma_and_map_mlps | delete | filter bank, gates, gamma |
| test_filter_bank.py | FilterBankModelTests.test_eta_only_under_first_order_and_V_in_the_normal_group | delete | eta / filter bank |
| test_filter_bank.py | FilterBankModelTests.test_operator_diagnostics_under_the_filter_bank | delete | filter-bank diagnostics |
| test_filter_bank.py | FilterBankModelTests.test_norm_sum_diagnostic_casts_the_blocks_to_float64 | delete | filter-bank diagnostics |
| test_filter_bank.py | FilterBankModelTests.test_switches_are_never_tuned | adapt | geometry_context/encoder never tuned: encoder subcase removed (deleted key), geometry_context stays |
| test_full_graph_early_stopping.py | FullGraphEarlyStoppingTests.test_config_rejects_nonpositive_patience | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_full_graph_early_stopping.py | FullGraphEarlyStoppingTests.test_node_trainer_honors_patience | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_full_graph_early_stopping.py | FullGraphEarlyStoppingTests.test_link_trainer_honors_patience | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_gap_ladder.py | IdentityMapsOperatorTests.test_identity_maps_give_the_normalized_graph_laplacian | delete | identity_maps (B1) |
| test_gap_ladder.py | IdentityMapsOperatorTests.test_control_learned_maps_are_not_the_graph_laplacian | delete | control of the identity-maps test (VertexPathSheafLaplacian) |
| test_gap_ladder.py | IdentityMapsModelTests.test_state_dict_unchanged_and_default_off | delete | identity_maps (B1) |
| test_gap_ladder.py | IdentityMapsModelTests.test_sheaf_free_logits_ignore_the_map_module | delete | identity_maps (B1) |
| test_gap_ladder.py | IdentityMapsModelTests.test_sheaf_free_still_propagates | delete | identity_maps (B1) |
| test_gap_ladder.py | IdentityMapsModelTests.test_config_rejects_invalid_combinations | delete | identity_maps combinations (B1) |
| test_gap_ladder.py | NeverTunedTests.test_identity_maps_rejected_in_tuning_base_config | delete | identity_maps (B1); fail-closed test replaces it |
| test_gap_ladder.py | NeverTunedTests.test_diagnostics_rejected_in_search_spaces | adapt | identity_maps subcase removed (deleted key); kept diagnostics stay |
| test_gap_ladder.py | FeatureNormalizationTests.test_config_values | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_gc_protocol.py | PoolingTests.test_config_pooling_values | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_gc_protocol.py | EarlyStoppingOnValidationLossTests.test_stops_on_validation_loss_and_reports_final_weights | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_graph_classification_trainer.py | GraphClassificationTrainerTests.test_early_stopping_on_validation_loss_reports_final_weights | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_hgb_lp_negatives.py | LoaderNegativeTests.test_validation_negatives_are_hgb_get_valid_neg | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_hgb_lp_negatives.py | LinkBatchTests.test_one_optimizer_step_and_one_solve_per_batch | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_hgb_oracle.py | FixedValidationFileTests.test_loader_reads_the_fixed_validation_file_exactly | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_lp_derived.py | FreebasePairSplitTests.test_loader_leak_checks_pass_and_positives_exclude_self_loops | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_lp_valid_files.py | FileWrittenFromTheCarveTests.test_written_file_reproduces_the_loaders_carve | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_lp_valid_files.py | DerivedTaskValidFileTests.test_non_target_relations_absent_from_the_file_stay_whole | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_lp_valid_files.py | TargetOnlyFileTests.test_target_only_file_keeps_the_target_split_and_completes_the_rest | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_lp_valid_files.py | RealValidFileTests.test_derived_tasks_evaluate_81_9_10 | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_pair_attention.py | AttentionWeightTests.test_TF4_every_row_is_positive_and_sums_to_one | delete | whole file tests deleted code (v2 operator / gates) |
| test_pair_attention.py | AttentionWeightTests.test_TF4_alpha_matches_an_independent_loop_based_softmax | delete | whole file tests deleted code (v2 operator / gates) |
| test_pair_attention.py | AttentionWeightTests.test_TF4_the_fixtures_have_loops_shared_roles_and_signed_hops | delete | whole file tests deleted code (v2 operator / gates) |
| test_pair_attention.py | SectionTests.test_TF5_a_global_section_is_a_fixed_point | delete | whole file tests deleted code (propagate() |
| test_pair_attention.py | NonExpansionTests.test_TF6_max_node_norm_is_nonincreasing_for_eta_up_to_one | delete | whole file tests deleted code (v2 operator / gates) |
| test_pair_attention.py | NonExpansionTests.test_TF6_control_unnormalized_exp_scores_expand | delete | whole file tests deleted code (v2 operator / gates) |
| test_pair_attention.py | DenseReferenceTests.test_TF7_operator_equals_the_explicit_matrix | delete | whole file tests deleted code (propagate() |
| test_pair_attention.py | DenseReferenceTests.test_TF7_state_dict_differs_from_d11_only_by_attention_added_and_gates_removed | delete | whole file tests deleted code (PairAttention,broadcast_gate,cooperative,listen_gate,pair_attention) |
| test_pair_attention.py | Prop4ControlTests.test_TF3b_control_pair_attention_has_no_metric | delete | whole file tests deleted code (cooperative,pair_attention) |
| test_pair_attention.py | MergedWeightTests.test_a_merged_cell_of_weight_two_equals_two_duplicate_unit_cells | delete | whole file tests deleted code (propagate() |
| test_pair_attention.py | MergedWeightTests.test_pair_attention_is_never_a_default_and_never_tuned | delete | whole file tests deleted code (cooperative,endpoint_edges,free_path_maps,holonomy_edges,no_path_cells,pair_attention,per_side_scale) |
| test_pair_attention.py | MergedWeightTests.test_attention_parameters_share_the_fast_group_and_gate_heads_are_absent | delete | whole file tests deleted code (K,cooperative,first_order,listen_gate,pair_attention,sigmoid_gamma,theta_eta,theta_gamma) |
| test_partM3a_decoder.py | LinkDecoderSweepTests.test_lp_sweeps_tune_the_decoder | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partM3a_decoder.py | LinkDecoderSweepTests.test_nc_and_gc_sweeps_have_no_decoder | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partM3a_decoder.py | ConcatMlpRefusalTests.test_concat_mlp_is_refused_by_the_tuner | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partM3afix_concat_mlp.py | ConcatMlpConfigTests.test_config_refuses_concat_mlp | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partM3afix_concat_mlp.py | ConcatMlpConfigTests.test_yaml_and_run_scripts_refuse_concat_mlp | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partMB_model.py | BatchTests.test_a_batch_equals_scopes_alone | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partMB_model.py | BatchTests.test_c_logits_do_not_depend_on_the_batching | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partMB_model.py | BatchTests.test_d_whole_graph_harness | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partMB_trainer.py | EpochTests.test_steps_per_epoch_follow_the_batch_size | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partMB_trainer.py | EpochTests.test_training_draw_seeds | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partMB_trainer.py | EvaluationTests.test_whole_split_and_metric_on_the_concatenation | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partMB_trainer.py | EvaluationTests.test_predictions_do_not_depend_on_the_evaluation_batch_size | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partMB_trainer.py | EvaluationTests.test_evaluation_graph_is_used_for_its_scopes | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partMB_trainer.py | StopAndConfigTests.test_mode_per_dataset_and_tuner_refusals | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partMBfix_flag.py | FlagConfigTests.test_tuner_refusal_and_h2gb_without_b | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partMBfix_flag.py | FlagTrainerTests.test_one_step_per_epoch_over_every_training_target | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partMBfix_flag.py | FlagTrainerTests.test_evaluation_every_epoch | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partO_literature_row.py | BuilderTests.test_row_configuration_rules | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partO_literature_row.py | DrawTests.test_sampler_asks_for_the_split | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partO_literature_row.py | DrawTests.test_every_cell_is_a_listed_meta_path | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partO_literature_row.py | DrawTests.test_evaluation_draw_is_deterministic_per_base_seed | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partO_literature_row.py | AnchoredForwardTests.test_forward_is_finite_on_an_acm_sized_graph | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partO_literature_row.py | RealListTests.test_lists_resolve_and_have_instances_on_the_real_graphs | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partO_readout.py | ModelWeightTests.test_weights_equal_the_models_path_weights | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_partO_readout.py | ModelWeightTests.test_constant_g_of_a_model_gives_ratio_one | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | FactLayerDefinitionTests.test_T4_real_hgb_counts | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | FactLayerDefinitionTests.test_T4_untied_is_rejected_by_the_tuning_check | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | SharedRoleTests.test_T16_model_shares_roles_per_fact_layer_and_ablation | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | SchemaTests.test_T10_untied_rejects_minus_hops | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | SchemaTests.test_T10_every_default_schema_resolves_on_real_data | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | StabilityTests.test_T13_lambda_max_at_most_two | delete | VertexPathSheafLaplacian lambda_max |
| test_paths_and_facts.py | UniformInstanceTests.test_T16_uniform_instance_is_uniform_over_instances | delete | uniform_instance |
| test_paths_and_facts.py | UniformInstanceTests.test_T16_uniform_instance_refuses_long_schemas | delete | uniform_instance |
| test_paths_and_facts.py | SamplerConfigAndFreeWalkTests.test_config_rules | adapt | uniform_instance subcase removed |
| test_paths_and_facts.py | ProtocolTests.test_T12_node_classification | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | ProtocolTests.test_T12_graph_classification | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | ProtocolTests.test_T12_config_rules | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | LinkPredictionFactTests.test_T11_split_is_by_fact | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | LinkPredictionFactTests.test_T12_link_prediction_resampling | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | LinkPredictionFactTests.test_T11_config | adapt | uniform_instance subcase removed |
| test_paths_and_facts.py | BaselineBundleIdentityTests.test_defaults_are_the_model | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | BaselineBundleIdentityTests.test_node_bundle_equals_explicit_untied_dfs | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | BaselineBundleIdentityTests.test_graph_bundle_equals_explicit_untied | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_paths_and_facts.py | BaselineBundleIdentityTests.test_pins_cannot_be_overridden | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_search_spaces.py | SweepConfigTests.test_node_classification_space_is_the_table | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_search_spaces.py | SweepConfigTests.test_link_prediction_spaces | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_search_spaces.py | SweepConfigTests.test_graph_classification_space | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_search_spaces.py | NeverTunedTests.test_K_layer_dropout_and_legacy_keys_are_refused | keep | refusal of legacy keys stays (they raise even earlier now) |
| test_search_spaces.py | SeedProtocolTests.test_final_seeds_train_the_best_seed0_configuration | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_search_spaces.py | BaselineCampaignTests.test_baseline_sweeps_are_plain_bayes | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_shared_protocol.py | OptimizerTests.test_h2snn_default_optimizer_is_adamw_with_the_scalar_group | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_shared_protocol.py | SchedulerAndClippingTests.test_no_scheduler_object_is_created | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_shared_protocol.py | SchedulerAndClippingTests.test_no_gradient_clipping_by_default | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_shared_protocol.py | BudgetTableTests.test_h2snn_defaults_follow_the_protocol | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_sheaf_energy.py | FunctorialityOnRealDrawTests.test_T1_fact_to_cell_maps_are_functorial_on_imdb | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_stop_conditions.py | StopConditionTests.test_normal_forward_does_not_stop | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_stop_conditions.py | StopConditionTests.test_tau_above_50_stops | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_stop_conditions.py | StopConditionTests.test_log_weight_range_above_30_stops | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_stop_conditions.py | StopConditionTests.test_limits_are_fixed_and_never_tuned | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_task_heads.py | NodeClassificationGoldenTests.test_outputs_and_gradients_are_bit_identical_to_the_pre_removal_capture | delete | v2 golden (captures the v2 model); replaced by tests/test_v3b_golden.py |
| test_task_heads.py | LinkPredictionHeadTests.test_each_decoder_trains_one_step_and_reaches_the_encoder | adapt | anchored config; eta/filter_norm_sum/gamma assertions -> tau / solver counterparts |
| test_task_heads.py | GraphClassificationHeadTests.test_one_training_step_is_finite | adapt | anchored config; filter_norm_sum assertion -> v3-B counterpart |
| test_trainer.py | TrainerSmokeTests.test_trainer_writes_expected_artifacts | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_v2_model.py | PerEdgeScaleTests.test_T12a_per_side_ablation_keeps_two_scales | delete | per_side_scale |
| test_v2_model.py | PerEdgeScaleTests.test_T12b_negative_control_per_side_scales_break_orthogonality | delete | per_side_scale |
| test_v2_model.py | PathLaplacianTests.test_T2_path_operator_equals_dense_kron_reduced_reference | delete | VertexPathSheafLaplacian |
| test_v2_model.py | PathLaplacianTests.test_T3_normalized_operator_is_self_adjoint_and_psd | delete | VertexPathSheafLaplacian |
| test_v2_model.py | PathLaplacianTests.test_D_is_the_per_incidence_degree_of_the_operators_own_terms | delete | VertexPathSheafLaplacian |
| test_v2_model.py | PathLaplacianTests.test_T5_normalized_spectrum_is_bounded_by_two | delete | VertexPathSheafLaplacian |
| test_v2_model.py | PathLaplacianTests.test_T7_cell_state_is_the_least_squares_minimizer | keep | PathCellTerm (kept) |
| test_v2_model.py | SectionPreservationTests.test_T4_sections_are_preserved_and_the_kernel_is_unchanged | delete | VertexPathSheafLaplacian |
| test_v2_model.py | SectionPreservationTests.test_T4_negative_control_free_maps_shrink_the_kernel | delete | FreeIncidenceOperator |
| test_v2_model.py | ScaleGaugeTests.test_T14_node_rescaled_per_side_scales_give_the_same_normalized_delta0 | delete | per_side_scale + VertexPathSheafLaplacian |
| test_v2_model.py | V2ModelTests.test_build_model_builds_the_node_classification_head | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_v2_model.py | V2ModelTests.test_T6_path_term_reaches_the_far_endpoint_and_the_intermediate_node | delete | v2 operator / first_order |
| test_v2_model.py | V2ModelTests.test_T8_no_dense_map_tensor_in_v2_forward_or_backward | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_v2_model.py | V2ModelTests.test_T9_deleted_components_are_absent | adapt | asserts v1 components absent; reads build_operator -> v3-B forward |
| test_v2_model.py | V2ModelTests.test_T10_gradients_reach_every_learned_component | adapt | anchored; theta_eta/theta_gamma/V_k -> theta_tau |
| test_v2_model.py | V2ModelTests.test_gate_initialization_is_near_one_half | adapt | exp weight init (kept); sigmoid_gamma subcase removed |
| test_v2_model.py | V2ModelTests.test_T13_dense_and_factorized_families_match_on_materialized_maps | adapt | per_side_scale subcase removed; map families kept |
| test_v2_model.py | V2ModelTests.test_checkpointing_switch_does_not_change_outputs_or_gradients | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_v2_model.py | V2ModelTests.test_trainer_logs_eta_gamma_uses_scalar_group_and_checkpoints_reload | adapt | anchored; logged eta/gamma -> tau; scalar group and reload kept |
| test_v2_model.py | AblationTests.test_T11_no_path_cells_makes_logits_independent_of_path_data | adapt | ablation.no_path_cells (B1) -> arm no_path_cells, same assertion |
| test_v2_model.py | AblationTests.test_T11_uniform_gate_gives_w_identically_one | adapt | path_weight_input uniform (kept) via build_operator -> v3-B weights |
| test_v2_model.py | AblationTests.test_T11_endpoint_edges_ignore_intermediate_nodes_in_the_path_term | delete | endpoint_edges |
| test_v2_model.py | AblationTests.test_T11_holonomy_edges_equal_the_dense_closure_edge | delete | holonomy_edges switch (v2); closure_only arm covered by sheaf_energy tests |
| test_v2_model.py | AblationTests.test_T11_free_maps_are_orthogonal_and_per_incidence | delete | free_path_maps / FreeIncidenceOperator |
| test_v2_model.py | AblationTests.test_T11_per_side_scales_reach_transports_and_degree | delete | per_side_scale |
| test_v2_model.py | AblationTests.test_T11_h0_only_readout_ignores_the_graph | adapt | h0_only_readout (kept); anchored config |
| test_v2_model.py | AblationTests.test_every_ablation_trains_one_step | adapt | kept ablations stay; deleted ones removed |
| test_v2_model.py | ConfigTests.test_defaults_are_the_full_model | adapt | defaults become v3-B (LR-2) |
| test_v2_model.py | ConfigTests.test_unknown_keys_fail_loudly | adapt | no_path_cells example -> another unknown key |
| test_v2_model.py | ConfigTests.test_invalid_values_and_conflicting_ablations_are_rejected | adapt | deleted-setting subcases -> fail-closed tests |
| test_v2_model.py | TwoLatentContextTests.test_a_endpoint_edge_maps_read_the_endpoints_context_not_intermediates | delete | endpoint_edges |
| test_v2_model.py | TwoLatentContextTests.test_b_g_reads_h0_only_unless_the_appendix_switch | adapt | path_weight_context (kept); endpoint_edges/free_path_maps subcases removed |
| test_v2_model.py | TwoLatentContextTests.test_c_free_maps_read_every_instance_node_context_through_the_latent | delete | free_path_maps |
| test_v2_model.py | TwoLatentContextTests.test_d_pair_attention_cell_scores_read_the_context | delete | pair_attention |
| test_v2_model.py | TwoLatentContextTests.test_context_off_keeps_one_encoder_per_arm | adapt | endpoint_edges/free_path_maps subcases removed; kept arms stay |
| test_v3b_model.py | ArmTests.test_T14_every_arm_trains_and_gradients_reach_the_right_groups | adapt | asserts filter_bank / theta_eta / gates are None under anchored: attributes deleted -> asserted absent |
| test_v3b_model.py | ArmTests.test_T14_config_keys_are_validated_and_never_tuned | adapt | legacy-default and legacy-value subcases -> fail-closed |
| test_v3b_theory.py | UnrollTests.test_U1_unrolled_gradient_equals_the_implicit_gradient | keep | local helper (v3-B) |
| test_wandb_and_runs.py | WandbAndRunReportingTests.test_run_summary_uses_task_metrics_and_mean_std | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wandb_and_runs.py | WandbAndRunReportingTests.test_wandb_sweep_config_and_tuning_selection | keep | asserts legacy keys absent from the sweep |
| test_wandb_and_runs.py | WandbAndRunReportingTests.test_repeated_cv_unit_mapping_config_and_population_summary | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wandb_and_runs.py | WandbAndRunReportingTests.test_fresh_validation_refuses_to_overwrite_existing_state | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wandb_and_runs.py | WandbAndRunReportingTests.test_failed_worker_does_not_prevent_next_isolated_attempt | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispSamplerTests.test_TH9_evaluation_draw_ignores_model_and_ablation_settings | adapt | deleted settings in the variation list -> kept ones |
| test_wisp.py | WispSamplerTests.test_wisp_config_rejections | adapt | uniform_instance subcase -> fail-closed |
| test_wisp.py | WispWeightOperatorTests.test_TH5_props_2_3_5_with_exp_weights | delete | VertexPathSheafLaplacian (v2 operator properties) |
| test_wisp.py | WispWeightOperatorTests.test_TH5_kernel_preserved_with_exp_weights | delete | VertexPathSheafLaplacian |
| test_wisp.py | WispLadderTests.test_TH6_exp_has_no_gamma_and_b_is_a_scalar_parameter | adapt | exp weight (kept); gamma/sigmoid_gamma subcases removed |
| test_wisp.py | WispLadderTests.test_saved_configs_round_trip_under_both_path_weights | adapt | sigmoid_gamma subcase removed |
| test_wisp.py | WispLadderTests.test_TH6_exp_operator_has_no_gamma_multiplier | delete | VertexPathSheafLaplacian |
| test_wisp.py | WispLadderTests.test_TH7_intermediate_and_its_neighbourhood_under_the_default_context | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispLadderTests.test_TH7_a_node_outside_the_instance_reaches_no_arm_but_the_appendix | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispLadderTests.test_TH7_positions_l_minus_1_and_l_separately | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispLadderTests.test_TH7_endpoint_perturbation_and_schema_equality | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispLadderTests.test_TH7_intermediate_reaches_endpoints_only_under_the_appendix_switch | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispLadderTests.test_TH7_capacity_rule_endpoints_vs_full_within_5_percent | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispLadderTests.test_path_weight_context_is_an_appendix_arm_never_tuned | adapt | no_path_cells subcase -> arm |
| test_wisp.py | WispLadderTests.test_TH8_length_neutral_initialisation | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispDefaultTests.test_default_path_source_and_r_per_dataset | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispDefaultTests.test_explicit_values_are_kept | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispDefaultTests.test_resolved_configs_round_trip | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispProposalTests.test_TH9_evaluation_draw_identical_across_proposals_and_ladder_arms | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispReadoutTests.test_TH11_partial_pair_form_matches_its_kept_pair_energy | delete | delta_loss (v2 operator; spec §6 not reported) |
| test_wisp.py | WispReadoutTests.test_TH11_delta_loss_of_an_absent_schema_is_zero | delete | delta_loss |
| test_wisp.py | WispConfigValidationTests.test_direct_construction_is_validated | adapt | sigmoid_gamma/gamma_init subcases removed |
| test_wisp.py | WispConfigValidationTests.test_wisp_keys_are_rejected_under_schemas_through_from_dict | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispConfigValidationTests.test_every_never_tuned_wisp_key_is_rejected | adapt | path_weight subcase (sigmoid_gamma) -> fail-closed |
| test_wisp.py | WispOverlapSamplingTests.test_overlap_switch_is_a_boolean_and_rejected_where_it_has_no_effect | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispGuardTests.test_path_weight_context_is_the_ladder_appendix_pair_only | adapt | deleted-arm subcases removed |
| test_wisp.py | WispGuardTests.test_r_resolves_from_the_table_for_its_l_max | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp.py | WispGuardTests.test_an_unresolved_path_source_is_refused_by_its_consumers | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp_tasks.py | LinkPredictionWispTests.test_config_resolves_wisp_for_link_prediction | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp_tasks.py | LinkPredictionWispTests.test_r_must_be_set_to_draw_cells | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp_tasks.py | LinkPredictionWispTests.test_no_held_out_fact_in_any_cell_of_any_draw | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp_tasks.py | LinkPredictionWispTests.test_evaluation_draw_fixed_training_draws_resampled | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |
| test_wisp_tasks.py | GraphClassificationWispTests.test_evaluation_draw_is_fixed_and_training_draws_change | adapt | builds through the default: LR-2 switches it to the anchored v3-B model (verified by the LR-2 run) |

Listed: {'keep': 6, 'adapt': 136, 'delete': 80}; every other test of `tests/` (467) is keep.
