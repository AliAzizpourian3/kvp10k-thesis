"""KVP10k evaluation compatible with IBM's public benchmark implementation.

The paper describes IoU as strictly greater than 0.3, while the released
benchmark code accepts IoU equal to the threshold. This module follows the
executable benchmark for reproducibility and exposes pooled counts only as
diagnostic metadata; headline precision and recall are macro-averaged over
documents with non-empty ground truth, then combined into F1.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


KVP_TYPES = ("kvp", "unkeyed", "unvalued")
MODES = ("text_only", "location_only", "text_location")


def normalized_edit_distance(first: str, second: str) -> float:
    first = str(first)
    second = str(second)
    rows, columns = len(first), len(second)
    if not rows and not columns:
        return 0.0
    if not rows or not columns:
        return 1.0

    distances = list(range(columns + 1))
    for row in range(1, rows + 1):
        previous = distances[0]
        distances[0] = row
        for column in range(1, columns + 1):
            diagonal = previous
            previous = distances[column]
            substitution = 0 if first[row - 1] == second[column - 1] else 1
            distances[column] = min(
                distances[column] + 1,
                distances[column - 1] + 1,
                diagonal + substitution,
            )
    return distances[columns] / max(rows, columns)


def intersection_over_union(first: Sequence[float], second: Sequence[float]) -> float:
    x_overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    y_overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = x_overlap * y_overlap
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _components(kvp: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    kvp_type = kvp.get("type", "kvp")
    if kvp_type == "unkeyed":
        return (kvp.get("value", {}),)
    if kvp_type == "unvalued":
        return (kvp.get("key", {}),)
    return (kvp.get("key", {}), kvp.get("value", {}))


def kvps_match(
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
    mode: str,
    ned_threshold: float = 0.2,
    iou_threshold: float = 0.3,
) -> bool:
    """Return whether two KVPs match under the released IBM benchmark rules."""
    if mode not in MODES:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    if ground_truth.get("type", "kvp") != prediction.get("type", "kvp"):
        return False

    gt_components = _components(ground_truth)
    pred_components = _components(prediction)
    if mode != "location_only":
        for gt_component, pred_component in zip(gt_components, pred_components):
            if normalized_edit_distance(
                gt_component.get("text", ""), pred_component.get("text", "")
            ) >= ned_threshold:
                return False
    if mode != "text_only":
        for gt_component, pred_component in zip(gt_components, pred_components):
            gt_bbox = gt_component.get("bbox")
            pred_bbox = pred_component.get("bbox")
            if not gt_bbox or not pred_bbox or len(gt_bbox) < 4 or len(pred_bbox) < 4:
                return False
            if intersection_over_union(gt_bbox, pred_bbox) < iou_threshold:
                return False
    return True


def match_document(
    predictions: Sequence[dict[str, Any]],
    ground_truths: Sequence[dict[str, Any]],
    mode: str,
    ned_threshold: float = 0.2,
    iou_threshold: float = 0.3,
) -> dict[str, float | int]:
    """Greedily match predictions in their supplied order, one GT at most once."""
    matched_ground_truths: set[int] = set()
    true_positives = 0
    for prediction in predictions:
        for gt_index, ground_truth in enumerate(ground_truths):
            if gt_index in matched_ground_truths:
                continue
            if kvps_match(
                ground_truth,
                prediction,
                mode,
                ned_threshold=ned_threshold,
                iou_threshold=iou_threshold,
            ):
                matched_ground_truths.add(gt_index)
                true_positives += 1
                break

    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truths) - true_positives
    precision = true_positives / len(predictions) if predictions else 0.0
    recall = true_positives / len(ground_truths) if ground_truths else 0.0
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_documents(
    documents: Iterable[dict[str, Any]],
    mode: str,
    kvp_type: str | None = None,
    ned_threshold: float = 0.2,
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    """Evaluate documents using the official macro-per-document aggregation."""
    if kvp_type is not None and kvp_type not in KVP_TYPES:
        raise ValueError(f"Unknown KVP type: {kvp_type}")

    per_document = []
    total_documents = 0
    for document in documents:
        total_documents += 1
        predictions = list(document.get("predictions", []))
        ground_truths = list(document.get("ground_truths", []))
        if kvp_type is not None:
            predictions = [item for item in predictions if item.get("type", "kvp") == kvp_type]
            ground_truths = [item for item in ground_truths if item.get("type", "kvp") == kvp_type]

        # IBM's released evaluator excludes documents with empty GT after filtering.
        if not ground_truths:
            continue
        metrics = match_document(
            predictions,
            ground_truths,
            mode,
            ned_threshold=ned_threshold,
            iou_threshold=iou_threshold,
        )
        per_document.append({"document_id": document.get("document_id"), **metrics})

    if not per_document:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "documents_total": total_documents,
            "documents_scored": 0,
            "documents_excluded_empty_gt": total_documents,
            "per_document": [],
        }

    precision = sum(item["precision"] for item in per_document) / len(per_document)
    recall = sum(item["recall"] for item in per_document) / len(per_document)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "tp": sum(item["tp"] for item in per_document),
        "fp": sum(item["fp"] for item in per_document),
        "fn": sum(item["fn"] for item in per_document),
        "documents_total": total_documents,
        "documents_scored": len(per_document),
        "documents_excluded_empty_gt": total_documents - len(per_document),
        "per_document": per_document,
    }


def evaluate_table(
    documents: Sequence[dict[str, Any]],
    ned_threshold: float = 0.2,
    iou_threshold: float = 0.3,
) -> dict[str, dict[str, Any]]:
    """Return All/Regular/Unkeyed/Unvalued by all three benchmark modes."""
    categories = {
        "all": None,
        "regular": "kvp",
        "unkeyed": "unkeyed",
        "unvalued": "unvalued",
    }
    return {
        category: {
            mode: evaluate_documents(
                documents,
                mode,
                kvp_type=kvp_type,
                ned_threshold=ned_threshold,
                iou_threshold=iou_threshold,
            )
            for mode in MODES
        }
        for category, kvp_type in categories.items()
    }