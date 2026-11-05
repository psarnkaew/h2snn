from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Callable, Literal, Sequence

import torch
from torch.nn import functional as F

from src.configs.config import AMPSS_PROTOCOL_VERSION


ProbeTask = Literal["single_label", "multi_label", "binary"]
SelectorProtocol = Literal["inner_holdout", "no_split_fallback"]


def derive_seed(run_seed: int, namespace: str, *parts: object) -> int:
    """Derive a stable 31-bit seed without depending on Python's salted hash."""
    payload = "|".join([str(run_seed), namespace, *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFF


@dataclass(frozen=True, slots=True)
class SelectorSplit:
    fit_indices: torch.Tensor
    selection_indices: torch.Tensor
    protocol: SelectorProtocol
    reason: str


@dataclass(frozen=True, slots=True)
class SelectionStep:
    step: int
    candidate_index: int
    candidate_identity: str
    task_gain: float
    selected_loss: float
    selected_coverage: float
    selected_incidence: float
    infeasible_candidates: int


@dataclass(frozen=True, slots=True)
class CandidateStatistics:
    candidate_index: int
    identity: str
    path_length: int
    viable: bool
    rejection_reason: str
    sampling_attempts: int
    sampling_successes: int
    sampling_unique_paths: int
    pre_quota_instances: int
    pre_quota_incidences: int
    probe_feature_dim: int
    coverage_units: int
    coverage_fraction: float
    selected: bool
    selection_step: int | None


@dataclass(frozen=True, slots=True)
class SelectedSchemaStatistics:
    schema_index: int
    identity: str
    materialized_faces: int
    path_incidences: int
    closure_incidences: int
    total_incidences: int
    coverage_units: int
    coverage_fraction: float
    face_set_hash: str


@dataclass(frozen=True, slots=True)
class CoverageDiagnostic:
    definition: str
    num_units: int
    candidate_identities: tuple[str, ...]
    candidate_coverage_fraction: tuple[float, ...]
    pairwise_jaccard: tuple[tuple[float, ...], ...]
    selected_identities: tuple[str, ...]
    selected_pairwise_jaccard: tuple[tuple[float, ...], ...]
    selected_union_coverage_fraction: float


@dataclass(frozen=True, slots=True)
class SelectorReport:
    protocol: SelectorProtocol
    split_seed: int
    sample_seed: int
    candidate_identity_count: int
    viable_candidate_count: int
    probe_feature_dims: tuple[int, ...]
    null_loss: float
    final_loss: float
    selected_indices: tuple[int, ...]
    selected_identities: tuple[str, ...]
    steps: tuple[SelectionStep, ...]
    stop_reason: str
    fit_size: int
    selection_size: int
    final_realized_incidence: float
    sampling_attempts: int
    sampling_successes: int
    sampling_unique_paths: int
    protocol_version: str = AMPSS_PROTOCOL_VERSION
    selector_unit_hash: str = ""
    selector_support_hash: str = ""
    materialized_face_hash: str = ""
    candidate_statistics: tuple[CandidateStatistics, ...] = ()
    selected_schema_statistics: tuple[SelectedSchemaStatistics, ...] = ()
    coverage_diagnostics: tuple[CoverageDiagnostic, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProbeSelectionResult:
    selected_indices: tuple[int, ...]
    report: SelectorReport
    final_support: "TrialSupport | None" = None


@dataclass(frozen=True, slots=True)
class TrialSupport:
    """The exact post-quota support used to score one selector subset."""

    candidate_indices: tuple[int, ...]
    quotas: tuple[int, ...]
    feature_blocks: tuple[torch.Tensor, ...]
    coverage_sets: tuple[frozenset[int], ...]
    realized_incidence: float
    support_hash: str = ""

    def __post_init__(self) -> None:
        size = len(self.candidate_indices)
        if not (
            len(self.quotas)
            == len(self.feature_blocks)
            == len(self.coverage_sets)
            == size
        ):
            raise ValueError("TrialSupport fields must align with candidate_indices.")
        if any(quota <= 0 for quota in self.quotas):
            raise ValueError("TrialSupport quotas must be positive.")
        if self.realized_incidence < 0.0:
            raise ValueError("TrialSupport realized_incidence must be non-negative.")


def _single_label_split(
    labels: torch.Tensor,
    holdout_ratio: float,
    seed: int,
    min_examples_per_label: int,
) -> SelectorSplit | None:
    generator = torch.Generator().manual_seed(seed)
    fit_parts: list[torch.Tensor] = []
    selection_parts: list[torch.Tensor] = []
    for label in sorted(int(value) for value in labels.unique().tolist()):
        indices = (labels == label).nonzero(as_tuple=False).view(-1)
        if indices.numel() < 2 * min_examples_per_label:
            return None
        order = indices[torch.randperm(indices.numel(), generator=generator)]
        num_selection = int(round(indices.numel() * holdout_ratio))
        num_selection = max(min_examples_per_label, num_selection)
        num_selection = min(num_selection, indices.numel() - min_examples_per_label)
        selection_parts.append(order[:num_selection])
        fit_parts.append(order[num_selection:])
    return SelectorSplit(
        fit_indices=torch.cat(fit_parts).sort().values,
        selection_indices=torch.cat(selection_parts).sort().values,
        protocol="inner_holdout",
        reason="stratified_single_label",
    )


def _multilabel_split(
    labels: torch.Tensor,
    holdout_ratio: float,
    seed: int,
    min_examples_per_label: int,
) -> SelectorSplit | None:
    """Choose the best of deterministic random trials under multilabel constraints."""
    num_examples = int(labels.size(0))
    num_selection = max(1, int(round(num_examples * holdout_ratio)))
    if num_selection >= num_examples:
        return None
    positives = labels.float().sum(dim=0)
    required = positives >= 2 * min_examples_per_label
    if bool((positives[required] < 2 * min_examples_per_label).any()):
        return None

    target = positives * holdout_ratio
    best: tuple[float, torch.Tensor] | None = None
    for trial in range(128):
        generator = torch.Generator().manual_seed(derive_seed(seed, "multilabel_trial", trial))
        order = torch.randperm(num_examples, generator=generator)
        selection = order[:num_selection]
        fit = order[num_selection:]
        selection_pos = labels[selection].float().sum(dim=0)
        fit_pos = labels[fit].float().sum(dim=0)
        if bool((selection_pos[required] < min_examples_per_label).any()):
            continue
        if bool((fit_pos[required] < min_examples_per_label).any()):
            continue
        scale = positives.clamp_min(1.0)
        score = float(((selection_pos - target).abs() / scale).mean().item())
        if best is None or score < best[0]:
            best = (score, selection)
    if best is None:
        return None
    selection = best[1].sort().values
    mask = torch.ones(num_examples, dtype=torch.bool)
    mask[selection] = False
    fit = mask.nonzero(as_tuple=False).view(-1)
    return SelectorSplit(
        fit_indices=fit,
        selection_indices=selection,
        protocol="inner_holdout",
        reason="multilabel_frequency_preserving",
    )


def make_selector_split(
    labels: torch.Tensor,
    task: ProbeTask,
    holdout_ratio: float,
    seed: int,
    min_examples_per_label: int,
) -> SelectorSplit:
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("selector holdout ratio must be in (0, 1).")
    if labels.size(0) < 2:
        indices = torch.arange(labels.size(0))
        return SelectorSplit(indices, indices, "no_split_fallback", "fewer_than_two_examples")

    split = (
        _multilabel_split(labels, holdout_ratio, seed, min_examples_per_label)
        if task == "multi_label"
        else _single_label_split(labels.view(-1), holdout_ratio, seed, min_examples_per_label)
    )
    if split is not None:
        return split
    indices = torch.arange(labels.size(0))
    return SelectorSplit(
        fit_indices=indices,
        selection_indices=indices,
        protocol="no_split_fallback",
        reason="stratified_holdout_infeasible",
    )


def _standardize(
    fit_x: torch.Tensor,
    eval_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = fit_x.mean(dim=0, keepdim=True)
    scale = fit_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (fit_x - mean) / scale, (eval_x - mean) / scale


def _class_prior_logits(labels: torch.Tensor, task: ProbeTask) -> torch.Tensor:
    if task == "single_label":
        num_classes = int(labels.max().item()) + 1
        counts = torch.bincount(labels.long(), minlength=num_classes).float()
        probs = (counts + 1.0) / (counts.sum() + num_classes)
        return probs.log()
    values = labels.float()
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    positives = values.sum(dim=0)
    probs = (positives + 1.0) / (values.size(0) + 2.0)
    probs = probs.clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(probs)


def _loss_weights(labels: torch.Tensor, task: ProbeTask) -> torch.Tensor | None:
    if task == "single_label":
        num_classes = int(labels.max().item()) + 1
        counts = torch.bincount(labels.long(), minlength=num_classes).float()
        if bool((counts == 0).any()):
            return None
        return labels.numel() / (num_classes * counts)
    values = labels.float()
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    positives = values.sum(dim=0)
    negatives = values.size(0) - positives
    return negatives / positives.clamp_min(1.0)


def _prediction_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task: ProbeTask,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    if task == "single_label":
        return F.cross_entropy(logits, labels.long(), weight=weights)
    targets = labels.float()
    if targets.dim() == 1:
        targets = targets.unsqueeze(-1)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights)


def null_probe_loss(
    fit_labels: torch.Tensor,
    eval_labels: torch.Tensor,
    task: ProbeTask,
) -> float:
    bias = _class_prior_logits(fit_labels, task)
    weights = _loss_weights(fit_labels, task)
    logits = bias.unsqueeze(0).expand(eval_labels.size(0), -1)
    return float(_prediction_loss(logits, eval_labels, task, weights).item())


def fit_linear_probe(
    fit_x: torch.Tensor,
    fit_labels: torch.Tensor,
    eval_x: torch.Tensor,
    eval_labels: torch.Tensor,
    task: ProbeTask,
    l2_strength: float,
    max_iter: int,
    include_regularization_in_score: bool,
) -> float:
    fit_x = fit_x.detach().float().cpu()
    eval_x = eval_x.detach().float().cpu()
    fit_labels = fit_labels.detach().cpu()
    eval_labels = eval_labels.detach().cpu()
    fit_x, eval_x = _standardize(fit_x, eval_x)

    output_dim = (
        int(fit_labels.max().item()) + 1
        if task == "single_label"
        else (int(fit_labels.size(1)) if fit_labels.dim() == 2 else 1)
    )
    weight = torch.zeros((fit_x.size(1), output_dim), requires_grad=True)
    bias = _class_prior_logits(fit_labels, task).detach().clone().requires_grad_(True)
    class_weights = _loss_weights(fit_labels, task)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = fit_x @ weight + bias
        loss = _prediction_loss(logits, fit_labels, task, class_weights)
        objective = loss + 0.5 * l2_strength * weight.pow(2).sum()
        objective.backward()
        return objective

    optimizer.step(closure)
    with torch.no_grad():
        eval_logits = eval_x @ weight + bias
        score = _prediction_loss(eval_logits, eval_labels, task, class_weights)
        if include_regularization_in_score:
            score = score + 0.5 * l2_strength * weight.pow(2).sum()
    return float(score.item())


def select_schema_features(
    *,
    candidate_identities: Sequence[str],
    feature_blocks: Sequence[torch.Tensor],
    coverage_sets: Sequence[frozenset[int]],
    incidence_masses: Sequence[float],
    labels: torch.Tensor,
    split: SelectorSplit,
    task: ProbeTask,
    max_schemas: int,
    incidence_budget: float | None,
    stop_tolerance: float,
    tie_tolerance: float,
    probe_l2: float,
    probe_max_iter: int,
    split_seed: int,
    sample_seed: int,
    candidate_identity_count: int,
    coverage_denominator: int,
    sampling_attempts: int,
    sampling_successes: int,
    sampling_unique_paths: int,
    subset_incidence_fn: Callable[[Sequence[int]], float | None] | None = None,
    subset_support_fn: Callable[[Sequence[int]], TrialSupport | None] | None = None,
) -> ProbeSelectionResult:
    if not (
        len(candidate_identities)
        == len(feature_blocks)
        == len(coverage_sets)
        == len(incidence_masses)
    ):
        raise ValueError(
            "candidate identities, features, coverage, and incidence masses must align."
        )
    if not candidate_identities:
        report = SelectorReport(
            protocol=split.protocol,
            split_seed=split_seed,
            sample_seed=sample_seed,
            candidate_identity_count=candidate_identity_count,
            viable_candidate_count=0,
            probe_feature_dims=tuple(),
            null_loss=math.nan,
            final_loss=math.nan,
            selected_indices=tuple(),
            selected_identities=tuple(),
            steps=tuple(),
            stop_reason="no_viable_candidates",
            fit_size=int(split.fit_indices.numel()),
            selection_size=int(split.selection_indices.numel()),
            final_realized_incidence=0.0,
            sampling_attempts=sampling_attempts,
            sampling_successes=sampling_successes,
            sampling_unique_paths=sampling_unique_paths,
        )
        return ProbeSelectionResult(tuple(), report, None)

    fit_labels = labels[split.fit_indices]
    eval_labels = labels[split.selection_indices]
    current_loss = null_probe_loss(fit_labels, eval_labels, task)
    null_loss = current_loss
    selected: list[int] = []
    selected_coverage: set[int] = set()
    selected_support: TrialSupport | None = None
    selected_incidence = 0.0
    steps: list[SelectionStep] = []
    stop_reason = "max_schemas_reached"

    while len(selected) < min(max_schemas, len(candidate_identities)):
        best: tuple[
            float,
            str,
            int,
            float,
            int,
            float,
            TrialSupport | None,
        ] | None = None
        infeasible = 0
        for candidate_index, identity in enumerate(candidate_identities):
            if candidate_index in selected:
                continue
            trial_indices = [*selected, candidate_index]
            trial_support = (
                subset_support_fn(trial_indices)
                if subset_support_fn is not None
                else None
            )
            subset_cost = (
                trial_support.realized_incidence
                if trial_support is not None
                else (
                    subset_incidence_fn(trial_indices)
                    if subset_incidence_fn is not None
                    else selected_incidence + float(incidence_masses[candidate_index])
                )
            )
            if subset_cost is None:
                infeasible += 1
                continue
            candidate_cost = float(subset_cost)
            if incidence_budget is not None and candidate_cost > incidence_budget:
                infeasible += 1
                continue
            if subset_support_fn is not None and trial_support is None:
                infeasible += 1
                continue
            trial_feature_blocks = (
                trial_support.feature_blocks
                if trial_support is not None
                else tuple(feature_blocks[index] for index in trial_indices)
            )
            features = torch.cat(trial_feature_blocks, dim=1)
            trial_loss = fit_linear_probe(
                fit_x=features[split.fit_indices],
                fit_labels=fit_labels,
                eval_x=features[split.selection_indices],
                eval_labels=eval_labels,
                task=task,
                l2_strength=probe_l2,
                max_iter=probe_max_iter,
                include_regularization_in_score=split.protocol == "no_split_fallback",
            )
            task_gain = current_loss - trial_loss
            if trial_support is not None:
                trial_coverage: set[int] = set()
                for covered in trial_support.coverage_sets:
                    trial_coverage.update(covered)
            else:
                trial_coverage = selected_coverage | set(coverage_sets[candidate_index])
            candidate_tuple = (
                task_gain,
                identity,
                candidate_index,
                trial_loss,
                len(trial_coverage),
                candidate_cost,
                trial_support,
            )
            if best is None:
                best = candidate_tuple
                continue
            if task_gain > best[0] + tie_tolerance:
                best = candidate_tuple
            elif abs(task_gain - best[0]) <= tie_tolerance and identity < best[1]:
                best = candidate_tuple

        if best is None:
            stop_reason = "no_feasible_candidates"
            break
        (
            task_gain,
            identity,
            candidate_index,
            trial_loss,
            coverage_size,
            candidate_cost,
            best_support,
        ) = best
        if task_gain <= stop_tolerance:
            stop_reason = "nonpositive_task_gain"
            break
        selected.append(candidate_index)
        selected_support = best_support
        if selected_support is not None:
            selected_coverage = set()
            for covered in selected_support.coverage_sets:
                selected_coverage.update(covered)
        else:
            selected_coverage.update(coverage_sets[candidate_index])
        selected_incidence = candidate_cost
        current_loss = trial_loss
        steps.append(
            SelectionStep(
                step=len(selected),
                candidate_index=candidate_index,
                candidate_identity=identity,
                task_gain=task_gain,
                selected_loss=current_loss,
                selected_coverage=coverage_size / max(coverage_denominator, 1),
                selected_incidence=selected_incidence,
                infeasible_candidates=infeasible,
            )
        )

    report = SelectorReport(
        protocol=split.protocol,
        split_seed=split_seed,
        sample_seed=sample_seed,
        candidate_identity_count=candidate_identity_count,
        viable_candidate_count=len(candidate_identities),
        probe_feature_dims=tuple(int(block.size(1)) for block in feature_blocks),
        null_loss=null_loss,
        final_loss=current_loss,
        selected_indices=tuple(selected),
        selected_identities=tuple(candidate_identities[index] for index in selected),
        steps=tuple(steps),
        stop_reason=stop_reason,
        fit_size=int(split.fit_indices.numel()),
        selection_size=int(split.selection_indices.numel()),
        final_realized_incidence=selected_incidence,
        sampling_attempts=sampling_attempts,
        sampling_successes=sampling_successes,
        sampling_unique_paths=sampling_unique_paths,
    )
    return ProbeSelectionResult(tuple(selected), report, selected_support)
