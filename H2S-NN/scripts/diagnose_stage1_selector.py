"""Statistical diagnostics for the AMPSS Stage-I probe selector.

Runs read-only analyses against the same selector code path used in training, without
touching `src/` or any training pipeline:

  baseline    reproduce the shipped selector run (sanity check against selector_report.json)
  seeds       vary `selector_seed` only -> how stable is the selected schema set?
  permute     shuffle training labels -> what does `task_gain` look like under the null?
  lp_only     probe on label propagation alone -> how much of the gain is not meta-paths?
  random_k    random schema subsets -> does greedy beat chance at the probe level?

Path sampling depends only on `materialization_seed`, so it is done once and cached.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import TrainConfig
from src.data.dataset import _load_hetero_data, _prepare_target_labels
from src.data.hetero_flatten import flatten_edges, stack_node_features
from src.data.node_stage1 import _anchors_from_slices, _node_schema_feature_block
from src.data.preprocessing import label_propagation
from src.data.schema_discovery import allocate_face_quotas, generate_schema_candidates
from src.data.schema_sampling import (
    SchemaPathSample,
    canonical_schema_identity,
    sample_schema_paths,
    sampled_support_hash,
    truncate_schema_path_sample,
)
from src.data.stage1_selector import (
    ProbeSelectionResult,
    SelectorSplit,
    TrialSupport,
    derive_seed,
    fit_linear_probe,
    make_selector_split,
    select_schema_features,
)


class SelectorHarness:
    """Holds everything that does not depend on the label split or the label values."""

    def __init__(self, config: TrainConfig, cache_dir: Path) -> None:
        self.config = config
        data = _load_hetero_data(config)
        raw_x, _, _, node_slices, _ = stack_node_features(data)
        edge_index, edge_type, _, relation_tuples = flatten_edges(data, node_slices)
        (
            _,
            _,
            _,
            _,
            target_indices,
            target_y,
            target_train_mask,
            num_classes,
            is_multilabel,
        ) = _prepare_target_labels(
            data=data,
            node_slices=node_slices,
            target_node_type=config.data.target_node_type,
            val_ratio=config.data.acm_val_ratio,
            split_seed=config.data.acm_split_seed,
        )

        self.node_slices = node_slices
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_nodes = int(raw_x.size(0))
        self.target_indices = target_indices
        self.target_begin = int(target_indices[0].item())
        self.train_target_nodes = target_indices[target_train_mask]
        self.true_train_labels = target_y[target_train_mask]
        self.num_classes = num_classes
        self.task = "multi_label" if is_multilabel else "single_label"

        self.sample_seed = derive_seed(
            config.discovery.materialization_seed,
            "ampss_path_sample",
            config.data.dataset_name,
        )
        self.candidates = generate_schema_candidates(
            relation_tuples=relation_tuples,
            target_node_type=config.data.target_node_type,
            min_length=config.discovery.min_schema_length,
            max_length=config.discovery.max_schema_length,
            require_target_start=config.discovery.require_target_start,
            require_target_end=config.discovery.require_target_end,
            allow_repeated_node_types=config.discovery.allow_repeated_node_types,
            max_identities=config.discovery.max_schema_identities,
        )
        self.viable_samples = self._sampled_paths(cache_dir)
        self.identities = tuple(
            canonical_schema_identity(sample.schema) for sample in self.viable_samples
        )
        self.standalone_quotas = [
            allocate_face_quotas(
                [sample.unique_paths],
                [sample.schema.length],
                config.discovery.max_total_instances,
                config.discovery.max_total_incidences,
                [canonical_schema_identity(sample.schema)],
            )[0]
            for sample in self.viable_samples
        ]

    def _sampled_paths(self, cache_dir: Path) -> tuple[SchemaPathSample, ...]:
        cfg = self.config
        key = "_".join(
            str(part)
            for part in (
                cfg.data.dataset_name,
                self.sample_seed,
                cfg.discovery.min_schema_length,
                cfg.discovery.max_schema_length,
                cfg.discovery.selector_num_attempts,
                cfg.discovery.max_instances_per_schema,
                cfg.discovery.max_total_incidences,
            )
        )
        cache_path = cache_dir / f"paths_{key}.pkl"
        if cache_path.exists():
            print(f"[cache] reusing sampled paths from {cache_path.name}")
            with open(cache_path, "rb") as handle:
                return pickle.load(handle)

        print("[sample] enumerating schema path instances (one time, then cached) ...")
        started = time.time()
        samples = sample_schema_paths(
            candidates=self.candidates,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            anchor_nodes_by_type=_anchors_from_slices(self.node_slices),
            target_nodes=frozenset(int(v) for v in self.target_indices.tolist()),
            attempts_per_anchor=cfg.discovery.selector_num_attempts,
            max_instances_per_schema=cfg.discovery.max_instances_per_schema,
            sample_seed=self.sample_seed,
            max_viable_samples=cfg.discovery.max_schema_candidates,
            max_total_incidences=cfg.discovery.max_total_incidences,
        )
        budget = cfg.discovery.max_total_incidences
        viable = tuple(
            sample
            for sample in samples
            if sample.unique_paths > 0
            and (budget is None or sample.schema.length + 1 <= budget)
        )
        print(f"[sample] {len(viable)} viable of {len(self.candidates)} candidates "
              f"in {time.time() - started:.1f}s")
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump(viable, handle)
        return viable

    def make_split(self, train_labels: torch.Tensor, selector_seed: int) -> SelectorSplit:
        cfg = self.config
        return make_selector_split(
            labels=train_labels,
            task=self.task,
            holdout_ratio=cfg.discovery.selector_holdout_ratio,
            seed=derive_seed(selector_seed, "ampss_selector_split", cfg.data.dataset_name),
            min_examples_per_label=cfg.discovery.selector_min_examples_per_label,
        )

    def propagate(self, train_labels: torch.Tensor, split: SelectorSplit) -> torch.Tensor:
        return label_propagation(
            edge_index=self.edge_index,
            train_indices=self.train_target_nodes[split.fit_indices],
            train_labels=train_labels[split.fit_indices],
            num_nodes=self.num_nodes,
            num_classes=self.num_classes,
            num_steps=self.config.data.num_label_prop_steps,
        )

    def _support_builders(self, propagated: torch.Tensor):
        """Mirror the prefix/subset support closures in src/data/node_stage1.py."""
        cfg = self.config
        cache: dict[tuple[int, int], tuple[SchemaPathSample, torch.Tensor, frozenset[int]]] = {}

        def prefix_support(candidate_index: int, quota: int):
            key = (candidate_index, quota)
            cached = cache.get(key)
            if cached is not None:
                return cached
            truncated = truncate_schema_path_sample(self.viable_samples[candidate_index], quota)
            block = _node_schema_feature_block(
                sample=truncated,
                target_nodes=self.train_target_nodes,
                propagated_fit_labels=propagated,
                num_classes=self.num_classes,
                attempts_per_anchor=cfg.discovery.selector_num_attempts,
            )
            covered = frozenset(n - self.target_begin for n in truncated.target_participants)
            cache[key] = (truncated, block, covered)
            return cache[key]

        def subset_support(indices: Sequence[int]) -> TrialSupport | None:
            canonical = tuple(
                sorted(
                    (int(i) for i in indices),
                    key=lambda i: canonical_schema_identity(self.viable_samples[i].schema),
                )
            )
            subset = [self.viable_samples[i] for i in canonical]
            quotas = allocate_face_quotas(
                [s.unique_paths for s in subset],
                [s.schema.length for s in subset],
                cfg.discovery.max_total_instances,
                cfg.discovery.max_total_incidences,
                [canonical_schema_identity(s.schema) for s in subset],
            )
            if any(q <= 0 for q in quotas):
                return None
            supports = [prefix_support(i, q) for i, q in zip(canonical, quotas, strict=True)]
            return TrialSupport(
                candidate_indices=canonical,
                quotas=tuple(quotas),
                feature_blocks=tuple(v[1] for v in supports),
                coverage_sets=tuple(v[2] for v in supports),
                realized_incidence=float(
                    sum(
                        q * (s.schema.length + 1)
                        for q, s in zip(quotas, subset, strict=True)
                    )
                ),
                support_hash=sampled_support_hash(tuple(v[0] for v in supports)),
            )

        return prefix_support, subset_support

    def run(self, train_labels: torch.Tensor, selector_seed: int) -> ProbeSelectionResult:
        cfg = self.config
        split = self.make_split(train_labels, selector_seed)
        propagated = self.propagate(train_labels, split)
        prefix_support, subset_support = self._support_builders(propagated)
        feature_blocks = tuple(
            prefix_support(i, q)[1] for i, q in enumerate(self.standalone_quotas)
        )
        coverage_sets = tuple(
            prefix_support(i, q)[2] for i, q in enumerate(self.standalone_quotas)
        )
        return select_schema_features(
            candidate_identities=self.identities,
            feature_blocks=feature_blocks,
            coverage_sets=coverage_sets,
            incidence_masses=tuple(s.incidence_cost for s in self.viable_samples),
            labels=train_labels,
            split=split,
            task=self.task,
            max_schemas=cfg.discovery.selected_schema_k,
            incidence_budget=cfg.discovery.max_total_incidences,
            stop_tolerance=cfg.discovery.selector_stop_tolerance,
            tie_tolerance=cfg.discovery.selector_tie_tolerance,
            probe_l2=cfg.discovery.selector_probe_l2,
            probe_max_iter=cfg.discovery.selector_probe_max_iter,
            split_seed=0,
            sample_seed=self.sample_seed,
            candidate_identity_count=len(self.candidates),
            coverage_denominator=int(self.target_indices.numel()),
            sampling_attempts=0,
            sampling_successes=0,
            sampling_unique_paths=0,
            subset_support_fn=subset_support,
        )


def short(identity: str) -> str:
    """'nodes=[paper,author,paper]|relations=[2,3]' -> 'PAP[2,3]'."""
    nodes_part, relations_part = identity.split("|")
    nodes = nodes_part.split("[", 1)[1].rstrip("]").split(",")
    relations = relations_part.split("[", 1)[1].rstrip("]")
    return "".join(n[0].upper() for n in nodes) + f"[{relations}]"


def experiment_baseline(harness: SelectorHarness) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print("BASELINE — reproduce the shipped selector run")
    print("=" * 78)
    started = time.time()
    result = harness.run(harness.true_train_labels, harness.config.discovery.selector_seed)
    r = result.report
    print(f"  protocol       : {r.protocol}")
    print(f"  fit/selection  : {r.fit_size} / {r.selection_size}")
    print(f"  candidates     : {r.viable_candidate_count} viable of {r.candidate_identity_count}")
    print(f"  null -> final  : {r.null_loss:.5f} -> {r.final_loss:.5f}")
    print(f"  stop_reason    : {r.stop_reason}")
    for s in r.steps:
        print(f"    step {s.step}: gain={s.task_gain:+.5f}  {short(s.candidate_identity)}")
    print(f"  [{time.time() - started:.1f}s per selector run]")
    return {
        "selected": [short(i) for i in r.selected_identities],
        "gains": [s.task_gain for s in r.steps],
        "null_loss": r.null_loss,
        "final_loss": r.final_loss,
        "stop_reason": r.stop_reason,
        "seconds": time.time() - started,
    }


def experiment_seeds(harness: SelectorHarness, num_seeds: int) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print(f"A. SEED STABILITY — {num_seeds} values of selector_seed (labels untouched)")
    print("=" * 78)
    rows, per_rank = [], {}
    for seed in range(num_seeds):
        r = harness.run(harness.true_train_labels, seed).report
        picked = [short(i) for i in r.selected_identities]
        rows.append({
            "seed": seed,
            "selected": picked,
            "indices": list(r.selected_indices),
            "gains": [s.task_gain for s in r.steps],
            "null_loss": r.null_loss,
            "final_loss": r.final_loss,
        })
        print(f"  seed {seed:>2}: k={len(picked)}  null={r.null_loss:.4f} "
              f"final={r.final_loss:.4f}  {picked}")
        for rank, name in enumerate(picked):
            per_rank.setdefault(rank, []).append(name)

    print("\n  stability by selection rank:")
    for rank in sorted(per_rank):
        names = per_rank[rank]
        top = max(set(names), key=names.count)
        print(f"    step {rank + 1}: {top:22s} chosen {names.count(top)}/{len(names)} times "
              f"({len(set(names))} distinct)")

    counts: dict[str, int] = {}
    for row in rows:
        for name in row["selected"]:
            counts[name] = counts.get(name, 0) + 1
    print("\n  how often each schema appears anywhere in the selected set:")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {name:22s} {n}/{num_seeds}")
    return {"runs": rows, "appearance_counts": counts}


def experiment_cross_eval(
    harness: SelectorHarness,
    seed_rows: list[dict[str, Any]],
    reference_loss: float,
) -> dict[str, Any]:
    """Are seed-to-seed differences in the selected set actually different models?

    Two views, because each answers a different question:
      own-split  each set scored on the split that chose it (no contamination, but
                 each row uses a different 72-example selection set)
      common     every set re-scored on the shipped seed-42 split (directly comparable,
                 but sets chosen on other splits saw part of this selection set during
                 their own selection, so their numbers are mildly optimistic)
    """
    print("\n" + "=" * 78)
    print("E. ARE THE UNSTABLE PICKS INTERCHANGEABLE?")
    print("=" * 78)
    cfg = harness.config
    labels = harness.true_train_labels
    split = harness.make_split(labels, cfg.discovery.selector_seed)
    propagated = harness.propagate(labels, split)
    _, subset_support = harness._support_builders(propagated)

    print(f"  {'seed':>5} {'k':>3} {'own-split final':>16} {'on seed-42 split':>18}")
    own, common = [], []
    for row in seed_rows:
        support = subset_support(row["indices"])
        if support is None:
            continue
        features = torch.cat(support.feature_blocks, dim=1)
        loss = fit_linear_probe(
            fit_x=features[split.fit_indices],
            fit_labels=labels[split.fit_indices],
            eval_x=features[split.selection_indices],
            eval_labels=labels[split.selection_indices],
            task=harness.task,
            l2_strength=cfg.discovery.selector_probe_l2,
            max_iter=cfg.discovery.selector_probe_max_iter,
            include_regularization_in_score=False,
        )
        own.append(row["final_loss"])
        common.append(loss)
        print(f"  {row['seed']:>5} {len(row['indices']):>3} {row['final_loss']:>16.5f} {loss:>18.5f}")

    print(f"\n  shipped seed-42 set on its own split: {reference_loss:.5f}")
    print(f"  own-split final : min {min(own):.5f} | median {sorted(own)[len(own)//2]:.5f} "
          f"| max {max(own):.5f}")
    print(f"  common split    : min {min(common):.5f} | median {sorted(common)[len(common)//2]:.5f} "
          f"| max {max(common):.5f}")
    return {"own_split": own, "common_split": common, "reference": reference_loss}


def experiment_permute(harness: SelectorHarness, num_permutations: int) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print(f"B. PERMUTATION NULL — {num_permutations} label shuffles (graph untouched)")
    print("=" * 78)
    gains_by_step: dict[int, list[float]] = {}
    selected_counts, all_rows = [], []
    for rep in range(num_permutations):
        generator = torch.Generator().manual_seed(derive_seed(9_000, "permutation", rep))
        order = torch.randperm(harness.true_train_labels.size(0), generator=generator)
        permuted = harness.true_train_labels[order].contiguous()
        r = harness.run(permuted, harness.config.discovery.selector_seed).report
        gains = [s.task_gain for s in r.steps]
        selected_counts.append(len(r.selected_indices))
        all_rows.append({"rep": rep, "k": len(r.selected_indices), "gains": gains,
                         "stop_reason": r.stop_reason})
        for i, g in enumerate(gains):
            gains_by_step.setdefault(i, []).append(g)
        print(f"  rep {rep:>2}: k={len(r.selected_indices)}  "
              f"gains={[round(g, 4) for g in gains]}  ({r.stop_reason})")

    print(f"\n  schemas selected under the null: mean {sum(selected_counts)/len(selected_counts):.2f}"
          f"  max {max(selected_counts)}  (ideal = 0)")
    print("\n  NOISE FLOOR — distribution of task_gain when labels carry no information:")
    print(f"    {'step':>5} {'n':>4} {'mean':>9} {'max':>9} {'p95':>9}")
    floor = {}
    for step in sorted(gains_by_step):
        values = sorted(gains_by_step[step])
        p95 = values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]
        floor[step + 1] = {"n": len(values), "mean": sum(values) / len(values),
                           "max": values[-1], "p95": p95}
        print(f"    {step + 1:>5} {len(values):>4} {sum(values)/len(values):>9.5f} "
              f"{values[-1]:>9.5f} {p95:>9.5f}")
    return {"runs": all_rows, "noise_floor": floor,
            "mean_selected": sum(selected_counts) / len(selected_counts)}


def experiment_lp_only(harness: SelectorHarness) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print("C. LABEL-PROPAGATION-ONLY NULL — how much gain is not from meta-paths?")
    print("=" * 78)
    cfg = harness.config
    labels = harness.true_train_labels
    split = harness.make_split(labels, cfg.discovery.selector_seed)
    propagated = harness.propagate(labels, split)
    lp_features = propagated[harness.train_target_nodes].float()
    loss = fit_linear_probe(
        fit_x=lp_features[split.fit_indices],
        fit_labels=labels[split.fit_indices],
        eval_x=lp_features[split.selection_indices],
        eval_labels=labels[split.selection_indices],
        task=harness.task,
        l2_strength=cfg.discovery.selector_probe_l2,
        max_iter=cfg.discovery.selector_probe_max_iter,
        include_regularization_in_score=False,
    )
    print(f"  probe on label propagation alone (no meta-path features): {loss:.5f}")
    return {"lp_only_loss": loss}


def experiment_random_k(harness: SelectorHarness, num_subsets: int, k: int) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print(f"D. RANDOM-K AT THE PROBE LEVEL — {num_subsets} random subsets of size {k}")
    print("=" * 78)
    cfg = harness.config
    labels = harness.true_train_labels
    split = harness.make_split(labels, cfg.discovery.selector_seed)
    propagated = harness.propagate(labels, split)
    _, subset_support = harness._support_builders(propagated)

    losses = []
    for rep in range(num_subsets):
        generator = torch.Generator().manual_seed(derive_seed(7_000, "random_k", rep))
        order = torch.randperm(len(harness.viable_samples), generator=generator)
        support = subset_support([int(i) for i in order[:k].tolist()])
        if support is None:
            continue
        features = torch.cat(support.feature_blocks, dim=1)
        losses.append(
            fit_linear_probe(
                fit_x=features[split.fit_indices],
                fit_labels=labels[split.fit_indices],
                eval_x=features[split.selection_indices],
                eval_labels=labels[split.selection_indices],
                task=harness.task,
                l2_strength=cfg.discovery.selector_probe_l2,
                max_iter=cfg.discovery.selector_probe_max_iter,
                include_regularization_in_score=False,
            )
        )
    losses.sort()
    best, worst = losses[0], losses[-1]
    median = losses[len(losses) // 2]
    print(f"  random-{k} probe loss over {len(losses)} draws (lower is better):")
    print(f"    best {best:.5f} | median {median:.5f} | worst {worst:.5f}")
    return {"losses": losses, "best": best, "median": median, "worst": worst}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--num_permutations", type=int, default=20)
    parser.add_argument("--num_random_subsets", type=int, default=20)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config) as handle:
        config = TrainConfig.from_dict(yaml.safe_load(handle))

    harness = SelectorHarness(config, args.cache_dir)
    results: dict[str, Any] = {"dataset": config.data.dataset_name}
    results["baseline"] = experiment_baseline(harness)
    results["lp_only"] = experiment_lp_only(harness)
    if args.num_random_subsets > 0:
        results["random_k"] = experiment_random_k(
            harness, args.num_random_subsets, len(results["baseline"]["selected"])
        )
    if args.num_seeds > 0:
        results["seeds"] = experiment_seeds(harness, args.num_seeds)
        results["cross_eval"] = experiment_cross_eval(
            harness, results["seeds"]["runs"], results["baseline"]["final_loss"]
        )
    if args.num_permutations > 0:
        results["permute"] = experiment_permute(harness, args.num_permutations)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
