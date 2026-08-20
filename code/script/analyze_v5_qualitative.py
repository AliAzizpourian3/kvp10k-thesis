"""Select reproducible qualitative V5 pair and entity-error examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_kvp10k_benchmark import load_documents
from kvp10k_official_eval import kvps_match, normalized_edit_distance
from train_stage4b_v5 import utc_now, write_json


def match_indices(predictions: list[dict], ground_truths: list[dict]) -> tuple[dict[int, int], set[int]]:
    matches: dict[int, int] = {}
    used: set[int] = set()
    for prediction_index, prediction in enumerate(predictions):
        for ground_truth_index, ground_truth in enumerate(ground_truths):
            if ground_truth_index in used:
                continue
            if kvps_match(ground_truth, prediction, "text_location", 0.2, 0.3):
                matches[prediction_index] = ground_truth_index
                used.add(ground_truth_index)
                break
    return matches, used


def _text(item: dict, component: str) -> str:
    return str(item.get(component, {}).get("text", "")).strip()


def _readable_pair(item: dict) -> bool:
    key, value = _text(item, "key"), _text(item, "value")
    return (
        1 <= len(key) <= 100
        and 1 <= len(value) <= 100
        and any(character.isalnum() for character in key)
        and any(character.isalnum() for character in value)
    )


def _pair_record(document_id: str, prediction: dict | None, ground_truth: dict | None, **extra: Any) -> dict:
    return {
        "document_id": document_id,
        "prediction": prediction,
        "ground_truth": ground_truth,
        **extra,
    }


def select_distinct_documents(records: list[dict], count: int) -> list[dict]:
    selected, seen = [], set()
    for record in records:
        if record["document_id"] in seen:
            continue
        selected.append(record)
        seen.add(record["document_id"])
        if len(selected) == count:
            break
    return selected


def analyze_pairs(documents: list[dict], cluster_map: dict[str, dict], count: int) -> dict:
    correct, wrong_links, missed = [], [], []
    for document in documents:
        document_id = document["document_id"]
        predictions = [item for item in document["predictions"] if item.get("type", "kvp") == "kvp"]
        ground_truths = [item for item in document["ground_truths"] if item.get("type", "kvp") == "kvp"]
        matches, matched_ground_truths = match_indices(predictions, ground_truths)
        cluster = str(cluster_map.get(document_id, {}).get("cluster", "geometry_unavailable"))
        for prediction_index, ground_truth_index in matches.items():
            prediction = predictions[prediction_index]
            if _readable_pair(prediction):
                correct.append(
                    _pair_record(
                        document_id,
                        prediction,
                        ground_truths[ground_truth_index],
                        cluster=cluster,
                        link_confidence=float(prediction.get("link_confidence", 0.0)),
                    )
                )
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index in matches or not _readable_pair(prediction):
                continue
            key_matches = [
                index for index, ground_truth in enumerate(ground_truths)
                if normalized_edit_distance(_text(ground_truth, "key"), _text(prediction, "key")) < 0.2
            ]
            value_matches = [
                index for index, ground_truth in enumerate(ground_truths)
                if normalized_edit_distance(_text(ground_truth, "value"), _text(prediction, "value")) < 0.2
            ]
            if not key_matches or not value_matches or set(key_matches).intersection(value_matches):
                continue
            expected_index = key_matches[0]
            wrong_links.append(
                _pair_record(
                    document_id,
                    prediction,
                    ground_truths[expected_index],
                    cluster=cluster,
                    link_confidence=float(prediction.get("link_confidence", 0.0)),
                    explanation="predicted key and value match different ground-truth pairs",
                    predicted_value_matches_ground_truth_indices=value_matches,
                )
            )
        for ground_truth_index, ground_truth in enumerate(ground_truths):
            if ground_truth_index in matched_ground_truths or not _readable_pair(ground_truth):
                continue
            missed.append(
                _pair_record(document_id, None, ground_truth, cluster=cluster, explanation="no official text+location match")
            )

    correct.sort(key=lambda item: (-item["link_confidence"], item["document_id"]))
    wrong_links.sort(key=lambda item: (-item["link_confidence"], item["document_id"]))
    missed.sort(
        key=lambda item: (
            len(_text(item["ground_truth"], "key")) + len(_text(item["ground_truth"], "value")),
            item["document_id"],
        )
    )
    return {
        "candidate_counts": {
            "correct_pairs": len(correct),
            "clear_wrong_links": len(wrong_links),
            "missed_pairs": len(missed),
        },
        "selected": {
            "correct_pairs": select_distinct_documents(correct, count),
            "wrong_links": select_distinct_documents(wrong_links, count),
            "missed_pairs": select_distinct_documents(missed, count),
        },
        "candidates": {
            "correct_pairs": correct[:50],
            "wrong_links": wrong_links[:50],
            "missed_pairs": missed[:50],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_dir", type=Path, required=True)
    parser.add_argument("--ground_truth_dir", type=Path, required=True)
    parser.add_argument("--cluster_map", type=Path, required=True)
    parser.add_argument("--entity_candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples_per_category", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    documents = load_documents(args.prediction_dir, args.ground_truth_dir)
    with args.cluster_map.open(encoding="utf-8") as handle:
        cluster_map = json.load(handle)
    with args.entity_candidates.open(encoding="utf-8") as handle:
        entity = json.load(handle)
    pair_analysis = analyze_pairs(documents, cluster_map, args.examples_per_category)
    entity_selected = {
        kind: select_distinct_documents(records, args.examples_per_category)
        for kind, records in entity["top_examples"].items()
    }
    write_json(
        args.output,
        {
            "metadata": {
                "created_at": utc_now(),
                "documents": len(documents),
                "ned_threshold": 0.2,
                "iou_threshold": 0.3,
                "matching_mode": "official text_location",
                "selection": "deterministic candidate ranking; inspect selected examples before publication",
            },
            "pair_analysis": pair_analysis,
            "entity_errors": {
                "confusion_matrix": entity["entity_confusion_matrix"],
                "candidate_counts": entity["candidate_counts"],
                "selected": entity_selected,
            },
        },
    )


if __name__ == "__main__":
    main()
