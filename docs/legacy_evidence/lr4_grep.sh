#!/bin/bash
# LR-4 grep record (card LR-4): every removed name over src/, tests/, scripts/, configs/ of H2S-NN. Text search only
# (login node, no computation). Output: docs/legacy_evidence/lr4_grep.txt. Classification: docs/H2S-NN_legacy_removal_report.md.
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
P='chebyshev|filter_bank|FilterBank|first_order|pair_attention|PairAttention|listen_only|cooperative|sigmoid_gamma|endpoint_edges|holonomy_edges|free_path_maps|uniform_instance|coupling_gate|eta_init|gamma_init|per_side_scale|layer_dropout|identity_maps|IdentityMap|memory_guard|DenseConformal|VertexPathSheafLaplacian|coboundary_ops|delta_loss|analyze_kernel|theta_eta|theta_gamma|map_family.{0,6}dense|"dense"|"K"|"encoder"|ablation\.no_path_cells|"no_path_cells": True'
echo "git=$(git rev-parse HEAD) (working tree; LR-4 changes uncommitted at the time of the grep) date=$(date -Is)"
echo "pattern: ${P}"
echo "files searched: src tests scripts configs (*.py *.yaml *.sh *.sbatch)"
grep -rnE "${P}" src tests scripts configs --include=*.py --include=*.yaml --include=*.sh --include=*.sbatch
echo "hits per file:"
grep -rnE "${P}" src tests scripts configs --include=*.py --include=*.yaml --include=*.sh --include=*.sbatch | cut -d: -f1 | sort | uniq -c
