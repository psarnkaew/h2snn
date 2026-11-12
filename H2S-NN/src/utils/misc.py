from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int, device: str | torch.device = "cpu") -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.default_generator.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _serialise_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialise_value(item) for item in value]
    if isinstance(value, list):
        return [_serialise_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialise_value(item) for key, item in value.items()}
    return value


def dataclass_to_dict(config: Any) -> dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError("Expected a dataclass instance.")
    return _serialise_value(asdict(config))


def save_config(config: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclass_to_dict(config)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    correct = (predictions == labels).sum().item()
    total = labels.numel()
    return 0.0 if total == 0 else correct / total


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    split_classes_only: bool = True,
) -> dict[str, float]:
    """Return acc, macro_f1, micro_f1 computed from detached tensors.

    Node classification (spec v1.15 §6, card Part M M8): Macro-F1 averages over the classes
    present in this split's true labels (`src/utils/nc_metrics.py`, shared with every baseline).
    `split_classes_only=False` keeps sklearn's default class set (labels present in the true or
    predicted labels): graph classification uses it, unchanged by M8.

    ROC-AUC is omitted: for multiclass node classification it requires OvR
    decomposition which is non-standard and rarely reported in GNN benchmarks.
    Uncomment the roc_auc block below if you need it for a binary task.
    """
    from sklearn.metrics import f1_score  # lazy import; sklearn is heavy

    if labels.numel() == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}

    preds = logits.detach().argmax(dim=-1)

    labels_np = labels.detach().cpu().numpy()
    preds_np = preds.cpu().numpy()

    acc = float((preds_np == labels_np).mean())
    if split_classes_only:
        from src.utils.nc_metrics import node_classification_f1

        f1 = node_classification_f1(labels_np, preds_np, is_multilabel=False)
        macro_f1, micro_f1 = f1["macro_f1"], f1["micro_f1"]
    else:
        macro_f1 = float(f1_score(labels_np, preds_np, average="macro", zero_division=0))
        micro_f1 = float(f1_score(labels_np, preds_np, average="micro", zero_division=0))

    # --- ROC-AUC (binary or multiclass OvR) ---
    # Uncomment if needed for binary node classification tasks.
    # from sklearn.metrics import roc_auc_score
    # probs_np = torch.softmax(logits.detach().float(), dim=-1).cpu().numpy()
    # num_classes = probs_np.shape[1]
    # try:
    #     roc_auc = float(roc_auc_score(labels_np, probs_np[:, 1])) if num_classes == 2 \
    #         else float(roc_auc_score(labels_np, probs_np, average="macro", multi_class="ovr"))
    # except ValueError:
    #     roc_auc = 0.0

    return {"acc": acc, "macro_f1": macro_f1, "micro_f1": micro_f1}


def compute_multilabel_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    protocol: str = "threshold",
) -> dict[str, float]:
    """Return subset accuracy plus macro/micro F1 for multi-label node classification.

    Two binarisation conventions exist and they are NOT comparable:

    * ``"threshold"`` (the default, and this project's convention): predict every class
      whose logit is positive. Also what HGB's own runners
      (``HGB/NC/benchmark/methods/GNN/run_multi.py:141``) and SeHGNN
      (``SeHGNN/hgb/utils.py:228``) do. Uses no test-set information.
    * ``"oracle_topk"``: predict the k highest-scoring classes, where k is the node's true
      label count. Used by PSHGCN (``PSHGCN/hgb/utils.py:Evaluation``) and H2SGNN. It reads
      each test node's ground-truth cardinality, and scores ~9 macro-F1 points higher on
      IMDB for a fixed trained model.

    An earlier docstring called this function "HGB-style" and implied the top-k rule was
    HGB's. It is not: HGB's ``evaluate()`` is metric-only and its own baselines threshold.
    See ``src/baselines/metrics.py`` for the shared implementation the baselines use.
    """
    if labels.numel() == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}

    labels_int = labels.detach().long()
    if protocol == "threshold":
        preds = (logits.detach() > 0).long()
    elif protocol == "oracle_topk":
        from src.baselines.metrics import binarize_multilabel

        preds = binarize_multilabel(logits.detach(), labels_int, protocol="oracle_topk")
    else:
        raise ValueError(
            f"protocol must be 'threshold' or 'oracle_topk', got {protocol!r}."
        )

    labels_np = labels_int.cpu().numpy()
    preds_np = preds.cpu().numpy()

    acc = float((preds_np == labels_np).all(axis=1).mean())
    # Spec v1.15 §6 (M8): a genre counts in Macro-F1 when at least one node of the split has it.
    from src.utils.nc_metrics import node_classification_f1

    f1 = node_classification_f1(labels_np, preds_np, is_multilabel=True)
    return {"acc": acc, "macro_f1": f1["macro_f1"], "micro_f1": f1["micro_f1"]}


def hetsheaf_f1(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Macro- and Micro-F1 of graph classification exactly as HetSheaf's `GraphClassifier`
    builds them (spec v1.16 K4): torchmetrics `F1Score(task="multiclass", average="macro" |
    "micro", num_classes=C)` on the logits (hetsheaf/graph_classification/graph_classifier.py:
    56-67, commit 6964e1e). torchmetrics 1.9.0 in `h2snn` (HetSheaf's uv.lock: 1.8.2)."""
    from torchmetrics.classification import F1Score

    num_classes = int(logits.size(-1))
    preds, target = logits.detach().float().cpu(), labels.detach().long().cpu()
    return {
        average + "_f1": float(F1Score(task="multiclass", average=average,
                                       num_classes=num_classes)(preds, target))
        for average in ("macro", "micro")
    }


def compute_graph_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """acc, Macro/Micro-F1 (HetSheaf's torchmetrics call, `hetsheaf_f1`) and AUROC of graph
    classification; used by H2S-NN's GC trainer and the baselines' GC loop."""
    from sklearn.metrics import roc_auc_score

    metrics = compute_classification_metrics(logits, labels, split_classes_only=False)
    if labels.numel() == 0:
        return {**metrics, "auroc": 0.0}
    metrics.update(hetsheaf_f1(logits, labels))

    probs_np = torch.softmax(logits.detach().float(), dim=-1).cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    try:
        if probs_np.shape[1] == 2:
            auroc = float(roc_auc_score(labels_np, probs_np[:, 1]))
        else:
            auroc = float(
                roc_auc_score(
                    labels_np,
                    probs_np,
                    average="macro",
                    multi_class="ovr",
                )
            )
    except ValueError:
        auroc = 0.0
    return {**metrics, "auroc": auroc}


LP_TEST_SETS: tuple[str, ...] = ("2hop", "random")


def lp_test_sets(data: Any) -> dict[str, tuple[Any, Any, str | None]]:
    """HGB's two test sets (spec v1.16 K1) of a prepared LP dataset or bundle: name ->
    (negatives, their relation ids, the reason when the set is empty)."""
    return {
        "2hop": (data.test_neg_edge_index, data.test_neg_edge_relation,
                 getattr(data, "test_2hop_empty_reason", None)),
        "random": (data.test_neg_random_edge_index, data.test_neg_random_edge_relation, None),
    }


def lp_test_set_metrics(score_pairs: Any, data: Any) -> dict[str, Any]:
    """ROC-AUC, AUPR and MRR per HGB test set, keys `<set>_<metric>` plus `<set>_negatives`.
    A set without negatives has metrics None and `<set>_reason` (never MRR = 1 or NaN).
    `score_pairs(edge_index, relation)` returns the model's scores of those pairs."""
    out: dict[str, Any] = {}
    pos, pos_rel = data.test_pos_edge_index, data.test_pos_edge_relation
    pos_scores = None
    for name, (neg, neg_rel, reason) in lp_test_sets(data).items():
        if neg is None:
            raise ValueError(f"the LP data carry no {name} test set (spec v1.16 K1).")
        out[f"{name}_negatives"] = int(neg.size(1))
        if neg.size(1) == 0:
            for metric in ("auroc", "aupr", "mrr"):
                out[f"{name}_{metric}"] = None
            out[f"{name}_reason"] = reason or "no negatives"
            continue
        if pos_scores is None:
            pos_scores = score_pairs(pos, pos_rel)
        metrics = compute_link_prediction_metrics(pos_scores, score_pairs(neg, neg_rel), pos,
                                                  neg, pos_rel, neg_rel)
        for metric, value in metrics.items():
            out[f"{name}_{metric}"] = value
    return out


def compute_link_prediction_metrics(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    pos_edge_index: torch.Tensor | None = None,
    neg_edge_index: torch.Tensor | None = None,
    pos_relation: torch.Tensor | None = None,
    neg_relation: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return AUROC/AUPR/MRR from raw positive/negative edge scores.

    MRR follows HGB's source-node grouping: for each source, rank all candidate
    tails by confidence and use the reciprocal rank of the highest-ranked
    positive tail.

    When ``pos_relation``/``neg_relation`` are supplied and the evaluation set spans
    more than one target relation, ``*_macro`` variants are added: each metric is
    computed independently per relation and then averaged unweighted. That is HGB's
    reported quantity on multi-relation targets -- every one of its link-prediction
    runners loops over ``dl.links_test['data'].keys()``, scores each relation on its own
    negatives, and divides the accumulated scores by the number of test relations
    (HGB/LP/benchmark/methods/HGT/link_predict.py:285-288 and the ``np.mean`` over
    ``test_score`` in .../GNN/homoGNN.py:155-159). The pooled figures stay under the
    unprefixed keys, so both are on the record and the tables can state which they use.
    """
    from collections import defaultdict

    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = torch.cat([pos_scores, neg_scores]).detach()
    labels = torch.cat(
        [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]
    ).detach()

    if labels.numel() == 0:
        return {"auroc": 0.0, "aupr": 0.0}

    probs_np = torch.sigmoid(scores.float()).cpu().numpy()
    labels_np = labels.cpu().numpy()

    try:
        auroc = float(roc_auc_score(labels_np, probs_np))
    except ValueError:
        auroc = 0.0
    aupr = float(average_precision_score(labels_np, probs_np))

    mrr = 0.0
    if pos_edge_index is not None and neg_edge_index is not None:
        edge_heads = torch.cat([pos_edge_index[0], neg_edge_index[0]]).detach().cpu().numpy()
        raw_scores_np = scores.float().cpu().numpy()
        grouped_labels: dict[int, list[float]] = defaultdict(list)
        grouped_scores: dict[int, list[float]] = defaultdict(list)
        for head, label, score in zip(edge_heads, labels_np, raw_scores_np):
            head_id = int(head)
            grouped_labels[head_id].append(float(label))
            grouped_scores[head_id].append(float(score))

        reciprocal_ranks: list[float] = []
        for head_id, head_labels in grouped_labels.items():
            order = np.argsort(-np.array(grouped_scores[head_id]))
            sorted_labels = np.array(head_labels)[order]
            positive_positions = np.where(sorted_labels == 1.0)[0]
            if positive_positions.size > 0:
                reciprocal_ranks.append(1.0 / float(positive_positions.min() + 1))
        if reciprocal_ranks:
            mrr = float(np.mean(reciprocal_ranks))

    metrics = {"auroc": auroc, "aupr": aupr, "mrr": mrr}

    if pos_relation is None or neg_relation is None:
        return metrics

    relations = torch.cat([pos_relation, neg_relation]).detach().cpu()
    distinct = torch.unique(relations)
    if distinct.numel() <= 1:
        # Single-relation target: the macro average is the pooled value by definition.
        # Emitted anyway so downstream records have a uniform key set across datasets.
        metrics.update(
            {"auroc_macro": auroc, "aupr_macro": aupr, "mrr_macro": mrr}
        )
        return metrics

    per_relation: dict[str, list[float]] = {"auroc": [], "aupr": [], "mrr": []}
    for relation_id in distinct.tolist():
        selector = relations == relation_id
        pos_selector = selector[: pos_scores.numel()]
        neg_selector = selector[pos_scores.numel() :]
        if not bool(pos_selector.any()) or not bool(neg_selector.any()):
            # A relation with no positives or no negatives has no defined AUROC; HGB's
            # per-relation loop never encounters this because get_test_neigh() samples
            # one negative per positive within each relation.
            continue
        relation_metrics = compute_link_prediction_metrics(
            pos_scores[pos_selector.to(pos_scores.device)],
            neg_scores[neg_selector.to(neg_scores.device)],
            None
            if pos_edge_index is None
            else pos_edge_index[:, pos_selector.to(pos_edge_index.device)],
            None
            if neg_edge_index is None
            else neg_edge_index[:, neg_selector.to(neg_edge_index.device)],
        )
        for key, values in per_relation.items():
            values.append(relation_metrics[key])

    for key, values in per_relation.items():
        metrics[f"{key}_macro"] = float(np.mean(values)) if values else 0.0
    metrics["num_eval_relations"] = float(len(per_relation["auroc"]))
    return metrics
