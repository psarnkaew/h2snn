from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRIC_NAMES = ("acc", "macro_f1", "micro_f1", "loss")
RELATION_METRIC_COLUMNS = (
    "label_signed_agreement",
    "structural_signed_agreement",
    "feature_signed_agreement",
    "label_nmi",
)
@dataclass(frozen=True)
class VariantSpec:
    name: str
    patches: dict[str, Any]
    note: str
    requires_discovery: bool = False


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def _save_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _set_nested(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _get_nested(payload: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _apply_patches(payload: dict[str, Any], patches: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(payload)
    for key, value in patches.items():
        _set_nested(updated, key, value)
    return updated


def _variant_specs(suite: str) -> list[VariantSpec]:
    core = [
        VariantSpec(
            name="full",
            patches={},
            note="Full H2SNN configuration.",
        ),
        VariantSpec(
            name="no_faces",
            patches={
                "model.ablate_no_faces": True,
            },
            note="Removes face caches, delta1, discovery gates, and closure edges inside the model.",
        ),
        VariantSpec(
            name="no_delta1",
            patches={
                "model.ablate_no_delta1": True,
            },
            note="Keeps graph diffusion but removes the delta1/up-Hodge face term (and with it the closure 1-cells).",
        ),
        VariantSpec(
            name="identity_maps",
            patches={"model.ablate_identity_maps": True},
            note="Replaces learned node-to-edge sheaf maps with identity maps during the forward pass.",
        ),
        VariantSpec(
            name="weak_closure",
            patches={"data.closure_scale": 0.1},
            note=(
                "Attenuates the closure 1-cell's contribution to L0. It cannot be set to 0: "
                "the closure edge is what makes the face boundary a cycle, and its face map "
                "carries a compensating 1/closure_scale, so delta1.delta0=0 needs a positive "
                "scale. Use no_delta1 to remove faces and closure edges together."
            ),
        ),
        VariantSpec(
            name="uniform_instance_weights",
            patches={"discovery.instance_weight_mode": "uniform"},
            note="Uses the Stage-I schema pool with uniform instance weights.",
            requires_discovery=True,
        ),
    ]
    depth = [
        VariantSpec(
            name=f"depth_k{k}",
            patches={"model.K": k},
            note=f"Diffusion-depth sensitivity with K={k}.",
        )
        for k in (1, 2, 4, 6, 8)
    ]
    face_count = [
        VariantSpec(
            name=f"face_k{k}",
            patches={"data.face_sample_k": k},
            note=f"Manual-schema face sampling sensitivity with face_sample_k={k}.",
        )
        for k in (10, 25, 50, 100)
    ]
    if suite == "core":
        return core
    if suite == "sweeps":
        return depth + face_count
    if suite == "all":
        return core + depth + face_count
    raise ValueError(f"Unknown suite {suite!r}.")


def _base_config_for_dataset(dataset: str, mode: str) -> Path:
    suffix = "_h2snn_ampss.yaml" if mode == "auto" else "_h2snn.yaml"
    return PROJECT_ROOT / "configs" / "experiments" / f"{dataset.lower()}{suffix}"


def _resolve_base_configs(args: argparse.Namespace) -> list[Path]:
    if args.base_configs:
        return [Path(path) for path in args.base_configs]
    return [_base_config_for_dataset(dataset, args.mode) for dataset in args.datasets]


def _is_discovery_enabled(payload: dict[str, Any]) -> bool:
    return bool(_get_nested(payload, "discovery.enabled", False))


def make_configs(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    output_dir = Path(args.output_dir)
    rows: list[dict[str, str]] = []
    variants = _variant_specs(args.suite)

    for base_path in _resolve_base_configs(args):
        base_payload = _load_yaml(base_path)
        dataset = str(_get_nested(base_payload, "data.dataset_name", base_path.stem.split("_")[0]))
        mode = str(_get_nested(base_payload, "discovery.mode", "manual"))
        discovery_enabled = _is_discovery_enabled(base_payload)

        for variant in variants:
            if variant.requires_discovery and not discovery_enabled and not args.include_inactive:
                continue
            for seed in args.seeds:
                payload = _apply_patches(base_payload, variant.patches)
                _set_nested(payload, "seed", int(seed))
                _set_nested(payload, "epochs", int(args.epochs))
                _set_nested(payload, "device", args.device)
                _set_nested(payload, "output_dir", str(output_dir))
                exp_name = f"{dataset}_{mode}_{variant.name}_seed{seed}"
                _set_nested(payload, "exp_name", exp_name)
                payload["study"] = {
                    "suite": args.suite,
                    "variant": variant.name,
                    "note": variant.note,
                    "base_config": str(base_path),
                }

                config_path = out_dir / dataset / variant.name / f"seed_{seed}.yaml"
                _save_yaml(payload, config_path)
                rows.append(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "variant": variant.name,
                        "seed": str(seed),
                        "config_path": str(config_path),
                        "exp_name": exp_name,
                        "output_dir": str(output_dir),
                        "note": variant.note,
                        "requires_discovery": str(variant.requires_discovery),
                        "discovery_enabled": str(discovery_enabled),
                    }
                )

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} configs")
    print(f"Manifest: {manifest_path}")


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _filter_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    filtered = rows
    if getattr(args, "datasets", None):
        datasets = {item.lower() for item in args.datasets}
        filtered = [row for row in filtered if row.get("dataset", "").lower() in datasets]
    if getattr(args, "variants", None):
        variants = set(args.variants)
        filtered = [row for row in filtered if row.get("variant") in variants]
    if getattr(args, "seeds", None):
        seeds = {str(item) for item in args.seeds}
        filtered = [row for row in filtered if row.get("seed") in seeds]
    return filtered


def run_manifest(args: argparse.Namespace) -> None:
    rows = _filter_rows(_read_manifest(Path(args.manifest)), args)
    if args.max_runs is not None:
        rows = rows[: args.max_runs]

    for index, row in enumerate(rows, start=1):
        exp_dir = Path(row["output_dir"]) / row["exp_name"]
        metrics_path = exp_dir / "metrics.jsonl"
        if args.skip_existing and metrics_path.exists():
            print(f"[{index}/{len(rows)}] skip existing {exp_dir}")
            continue

        cmd = [
            args.python_exe,
            str(PROJECT_ROOT / "scripts" / "train_h2snn.py"),
            "--config",
            row["config_path"],
            "--device",
            args.device,
        ]
        print(f"[{index}/{len(rows)}] {' '.join(cmd)}")
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _flatten_metrics(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, name))
        else:
            flat[name] = value
    return flat


def _best_record(records: list[dict[str, Any]], split: str, metric: str, mode: str) -> dict[str, Any] | None:
    if not records:
        return None
    key = lambda rec: float(rec.get(split, {}).get(metric, float("nan")))
    valid = [rec for rec in records if not math.isnan(key(rec))]
    if not valid:
        return records[-1]
    return max(valid, key=key) if mode == "max" else min(valid, key=key)


def _parse_exp_name(exp_name: str) -> tuple[str, str, str]:
    match = re.match(r"(?P<dataset>[^_]+)_(?P<mode>manual|auto)_(?P<body>.+)_seed(?P<seed>\d+)$", exp_name)
    if match:
        return match.group("dataset"), match.group("mode"), match.group("body")
    parts = exp_name.split("_")
    dataset = parts[0] if parts else "unknown"
    return dataset, "unknown", exp_name


def _manifest_lookup(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {row["exp_name"]: row for row in _read_manifest(path)}


def collect_results(args: argparse.Namespace) -> None:
    experiments_dir = Path(args.experiments_dir)
    manifest = _manifest_lookup(Path(args.manifest) if args.manifest else None)
    rows: list[dict[str, Any]] = []

    for exp_dir in sorted(path for path in experiments_dir.iterdir() if path.is_dir()):
        config_path = exp_dir / "config.yaml"
        metrics_path = exp_dir / "metrics.jsonl"
        if not config_path.exists() or not metrics_path.exists():
            continue
        cfg = _load_yaml(config_path)
        records = _read_jsonl(metrics_path)
        metric = str(cfg.get("checkpoint_metric", "acc")).removeprefix("val_")
        mode = str(cfg.get("checkpoint_mode", "max"))
        best = _best_record(records, "val", metric, mode)
        final = records[-1] if records else None
        if best is None or final is None:
            continue

        exp_name = str(cfg.get("exp_name", exp_dir.name))
        parsed_dataset, parsed_mode, parsed_variant = _parse_exp_name(exp_name)
        manifest_row = manifest.get(exp_name, {})
        dataset = str(_get_nested(cfg, "data.dataset_name", manifest_row.get("dataset", parsed_dataset)))
        variant = manifest_row.get("variant", parsed_variant)
        seed = str(cfg.get("seed", manifest_row.get("seed", "")))

        row: dict[str, Any] = {
            "exp_name": exp_name,
            "exp_dir": str(exp_dir),
            "dataset": dataset,
            "mode": manifest_row.get("mode", parsed_mode),
            "variant": variant,
            "seed": seed,
            "best_epoch": best.get("epoch", ""),
            "final_epoch": final.get("epoch", ""),
            "model_K": _get_nested(cfg, "model.K", ""),
            "model_d": _get_nested(cfg, "model.d", ""),
            "face_sample_k": _get_nested(cfg, "data.face_sample_k", ""),
            "closure_scale": _get_nested(cfg, "data.closure_scale", ""),
            "ablate_no_faces": _get_nested(cfg, "model.ablate_no_faces", False),
            "ablate_no_delta1": _get_nested(cfg, "model.ablate_no_delta1", False),
            "ablate_identity_maps": _get_nested(cfg, "model.ablate_identity_maps", False),
            "instance_weight_mode": _get_nested(
                cfg, "discovery.instance_weight_mode", ""
            ),
        }
        flat_best = _flatten_metrics(best)
        flat_final = _flatten_metrics(final)
        for split in ("train", "val", "test"):
            for metric_name in METRIC_NAMES:
                row[f"best_{split}_{metric_name}"] = flat_best.get(f"{split}_{metric_name}", "")
                row[f"final_{split}_{metric_name}"] = flat_final.get(f"{split}_{metric_name}", "")
        for key in (
            "train_instance_weight_mean",
            "train_instance_weight_std",
            "val_instance_weight_mean",
            "val_instance_weight_std",
            "test_instance_weight_mean",
            "test_instance_weight_std",
        ):
            row[f"best_{key}"] = flat_best.get(key, "")
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} result rows to {out_csv}")


def _import_plotting() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    return plt, pd


def _save_bar_plot(df: Any, metric: str, out_path: Path, title: str) -> None:
    plt, _ = _import_plotting()
    summary = (
        df.groupby(["dataset", "variant"], dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    if summary.empty:
        return
    datasets = list(summary["dataset"].drop_duplicates())
    variants = list(summary["variant"].drop_duplicates())
    width = 0.8 / max(len(variants), 1)
    x_positions = list(range(len(datasets)))
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(variants)), 5))
    for variant_idx, variant in enumerate(variants):
        values = []
        errors = []
        for dataset in datasets:
            item = summary[(summary["dataset"] == dataset) & (summary["variant"] == variant)]
            values.append(float(item["mean"].iloc[0]) if not item.empty else 0.0)
            errors.append(float(item["std"].iloc[0]) if not item.empty and item["count"].iloc[0] > 1 else 0.0)
        offsets = [x + (variant_idx - (len(variants) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=variant, yerr=errors, capsize=3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_sweep(df: Any, x_col: str, metric: str, out_path: Path, title: str) -> None:
    plt, _ = _import_plotting()
    sweep_df = df[df[x_col] != ""].copy()
    if sweep_df.empty:
        return
    sweep_df[x_col] = sweep_df[x_col].astype(float)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for dataset, group in sweep_df.groupby("dataset"):
        grouped = group.groupby(x_col)[metric].mean().reset_index().sort_values(x_col)
        ax.plot(grouped[x_col], grouped[metric], marker="o", label=dataset)
    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_training_curves(summary_df: Any, out_dir: Path, metric: str) -> None:
    plt, _ = _import_plotting()
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for _, row in summary_df.head(12).iterrows():
        metrics_path = Path(row["exp_dir"]) / "metrics.jsonl"
        records = _read_jsonl(metrics_path)
        epochs = [rec["epoch"] for rec in records if metric in rec.get("val", {})]
        values = [rec["val"][metric] for rec in records if metric in rec.get("val", {})]
        if not epochs:
            continue
        label = f"{row['dataset']}:{row['variant']}:s{row['seed']}"
        ax.plot(epochs, values, label=label, alpha=0.85)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"val {metric}")
    ax.set_title(f"Validation {metric} curves")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"training_curves_val_{metric}.png", dpi=220)
    plt.close(fig)


def _plot_instance_weight_curves(summary_df: Any, out_dir: Path) -> None:
    plt, _ = _import_plotting()
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for _, row in summary_df.iterrows():
        metrics_path = Path(row["exp_dir"]) / "metrics.jsonl"
        records = _read_jsonl(metrics_path)
        epochs: list[int] = []
        mean_values: list[float] = []
        std_values: list[float] = []
        for rec in records:
            train = rec.get("train", {})
            if "instance_weight_mean" in train:
                epochs.append(int(rec["epoch"]))
                mean_values.append(float(train["instance_weight_mean"]))
                std_values.append(
                    float(train.get("instance_weight_std", float("nan")))
                )
        if not epochs:
            continue
        label = f"{row['dataset']}:{row['variant']}:s{row['seed']}"
        ax.plot(epochs, mean_values, label=f"mean {label}", alpha=0.85)
        if std_values and not all(math.isnan(value) for value in std_values):
            ax.plot(
                epochs,
                std_values,
                linestyle="--",
                label=f"std {label}",
                alpha=0.7,
            )
        plotted += 1
        if plotted >= 8:
            break
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("effective instance weight")
    ax.set_title("AMPSS instance-weight mean and standard deviation")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "instance_weight_curves.png", dpi=220)
    plt.close(fig)


def plot_results(args: argparse.Namespace) -> None:
    plt, pd = _import_plotting()
    del plt
    df = pd.read_csv(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric = args.metric
    if metric not in df.columns:
        raise ValueError(f"Metric column {metric!r} is missing from {args.summary_csv}")

    _save_bar_plot(
        df=df,
        metric=metric,
        out_path=out_dir / f"ablation_bar_{metric}.png",
        title=f"H2SNN ablation study ({metric})",
    )
    _plot_sweep(
        df=df[df["variant"].astype(str).str.startswith("depth_k") | (df["variant"] == "full")],
        x_col="model_K",
        metric=metric,
        out_path=out_dir / f"depth_sweep_{metric}.png",
        title=f"Diffusion depth sweep ({metric})",
    )
    _plot_sweep(
        df=df[df["variant"].astype(str).str.startswith("face_k") | (df["variant"] == "full")],
        x_col="face_sample_k",
        metric=metric,
        out_path=out_dir / f"face_count_sweep_{metric}.png",
        title=f"Face sample sweep ({metric})",
    )
    _plot_training_curves(df, out_dir, args.curve_metric)
    _plot_instance_weight_curves(df, out_dir)
    print(f"Wrote plots to {out_dir}")


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_heatmap(matrix: list[list[float]], row_labels: list[str], col_labels: list[str], out_path: Path, title: str) -> None:
    plt, _ = _import_plotting()
    if not matrix:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(col_labels) * 1.4), max(4, len(row_labels) * 0.45)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_face_summary(face_rows: list[dict[str, Any]], out_path: Path) -> None:
    plt, _ = _import_plotting()
    if not face_rows:
        return
    labels = [row["template_name"] for row in face_rows]
    counts = [int(row["num_faces"]) for row in face_rows]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 4.5))
    ax.bar(labels, counts)
    ax.set_ylabel("num faces")
    ax.set_title("Face count by schema/template")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_schema_scatter(schema_rows: list[dict[str, Any]], out_path: Path) -> None:
    plt, _ = _import_plotting()
    if not schema_rows:
        return
    x = [float(row["pre_quota_incidences"]) for row in schema_rows]
    y = [float(row["coverage_fraction"]) for row in schema_rows]
    sizes = [30 + 15 * float(row["probe_feature_dim"]) for row in schema_rows]
    labels = [str(row["candidate_index"]) for row in schema_rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=sizes, alpha=0.75)
    for label, x_value, y_value in zip(labels, x, y, strict=False):
        ax.text(x_value, y_value, label, fontsize=8)
    ax.set_xlabel("pre-quota incidences")
    ax.set_ylabel("observed coverage fraction")
    ax.set_title("AMPSS candidate cost vs coverage")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def diagnostics(args: argparse.Namespace) -> None:
    from scripts.train_h2snn import load_config
    from src.data.dataset import build_dataset
    from src.utils.misc import seed_everything

    cfg = load_config(Path(args.config))
    # Training runs skip these statistics (they are reporting-only and cost a
    # per-node two-hop pass); this study is their sole consumer, so opt in here.
    cfg.data.compute_relation_diagnostics = True
    seed_everything(cfg.seed, cfg.device)
    data = build_dataset(cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_values = data.metrics.detach().cpu().view(len(data.relation_names), len(RELATION_METRIC_COLUMNS))
    relation_rows = []
    for relation_id, relation_name in enumerate(data.relation_names):
        row = {"relation_id": relation_id, "relation_name": relation_name}
        for metric_name, value in zip(RELATION_METRIC_COLUMNS, metric_values[relation_id], strict=True):
            row[metric_name] = float(value.item())
        relation_rows.append(row)
    _write_rows_csv(relation_rows, out_dir / "relation_metrics.csv")
    _plot_heatmap(
        matrix=[[float(row[col]) for col in RELATION_METRIC_COLUMNS] for row in relation_rows],
        row_labels=[str(row["relation_name"]) for row in relation_rows],
        col_labels=list(RELATION_METRIC_COLUMNS),
        out_path=out_dir / "relation_metric_heatmap.png",
        title=f"{cfg.data.dataset_name.upper()} relation diagnostics",
    )

    face_rows = []
    for group_id, group in enumerate(data.face_groups):
        face_rows.append(
            {
                "group_id": group_id,
                "template_name": group.template_name,
                "node_type_sequence": "-".join(group.node_type_sequence),
                "num_faces": group.num_faces,
                "length": group.length,
                "closure_edges": int(group.closure_edge_index.size(0)),
                "nontrivial_negative_signs": int((group.face_edge_sign < 0).sum().item()),
            }
        )
    _write_rows_csv(face_rows, out_dir / "face_summary.csv")
    _plot_face_summary(face_rows, out_dir / "face_counts.png")

    schema_rows: list[dict[str, Any]] = []
    if data.selector_report is not None:
        schema_rows = [
            {
                "candidate_index": row.candidate_index,
                "identity": row.identity,
                "path_length": row.path_length,
                "viable": row.viable,
                "rejection_reason": row.rejection_reason,
                "pre_quota_instances": row.pre_quota_instances,
                "pre_quota_incidences": row.pre_quota_incidences,
                "probe_feature_dim": row.probe_feature_dim,
                "coverage_fraction": row.coverage_fraction,
                "selected": row.selected,
                "selection_step": row.selection_step,
            }
            for row in data.selector_report.candidate_statistics
        ]
    if schema_rows:
        _write_rows_csv(schema_rows, out_dir / "schema_diagnostics.csv")
        _plot_schema_scatter(schema_rows, out_dir / "schema_score_scatter.png")

    summary = {
        "dataset": cfg.data.dataset_name,
        "target_node_type": data.target_node_type,
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_relations": len(data.relation_names),
        "num_face_groups": len(data.face_groups),
        "num_faces": sum(group.num_faces for group in data.face_groups),
        "num_closure_edges": int(data.closure_edge_index.size(1)),
        "discovery_enabled": data.discovery is not None,
    }
    with (out_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote diagnostics to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, run, collect, and plot H2SNN paper ablation studies."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("make-configs", help="Generate H2SNN ablation configs.")
    make.add_argument("--datasets", nargs="+", default=["acm", "dblp", "imdb"])
    make.add_argument("--mode", choices=["manual", "auto"], default="manual")
    make.add_argument("--base-configs", nargs="*", default=None)
    make.add_argument("--suite", choices=["core", "sweeps", "all"], default="core")
    make.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    make.add_argument("--epochs", type=int, default=200)
    make.add_argument("--device", default="cuda")
    make.add_argument("--out-dir", default="configs/generated/h2snn_paper_ablation")
    make.add_argument("--output-dir", default="experiments/h2snn_paper_ablation")
    make.add_argument("--manifest", default="configs/generated/h2snn_paper_ablation/manifest.csv")
    make.add_argument("--include-inactive", action="store_true")
    make.set_defaults(func=make_configs)

    run = subparsers.add_parser("run", help="Run generated H2SNN configs.")
    run.add_argument("--manifest", required=True)
    run.add_argument("--python-exe", default=sys.executable)
    run.add_argument("--device", default="cuda")
    run.add_argument("--datasets", nargs="*", default=None)
    run.add_argument("--variants", nargs="*", default=None)
    run.add_argument("--seeds", nargs="*", type=int, default=None)
    run.add_argument("--max-runs", type=int, default=None)
    run.add_argument("--skip-existing", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=run_manifest)

    collect = subparsers.add_parser("collect", help="Collect H2SNN experiment metrics.")
    collect.add_argument("--experiments-dir", default="experiments/h2snn_paper_ablation")
    collect.add_argument("--manifest", default=None)
    collect.add_argument("--out-csv", default="experiments/h2snn_paper_ablation/summary.csv")
    collect.set_defaults(func=collect_results)

    plot = subparsers.add_parser("plot", help="Create H2SNN ablation and study plots.")
    plot.add_argument("--summary-csv", required=True)
    plot.add_argument("--out-dir", default="figures/h2snn_paper_ablation")
    plot.add_argument("--metric", default="best_test_macro_f1")
    plot.add_argument("--curve-metric", default="macro_f1")
    plot.set_defaults(func=plot_results)

    diag = subparsers.add_parser("diagnostics", help="Create dataset/schema diagnostic plots.")
    diag.add_argument("--config", required=True)
    diag.add_argument("--out-dir", required=True)
    diag.set_defaults(func=diagnostics)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
