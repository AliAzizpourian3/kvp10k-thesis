"""Reconcile V5 final-evaluation artifacts and write a machine-readable audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_stage4b_v5 import utc_now, write_json


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _check(checks: dict[str, bool], name: str, condition: bool) -> None:
    checks[name] = bool(condition)
    if not condition:
        raise AssertionError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--postprocessed", type=Path, required=True)
    parser.add_argument("--original-test", type=Path, required=True)
    parser.add_argument("--entity-candidates", type=Path, required=True)
    parser.add_argument("--run-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    direct = _load(args.direct)
    post = _load(args.postprocessed)
    original = _load(args.original_test)
    entity = _load(args.entity_candidates)
    run_info = _load(args.run_info)
    checks: dict[str, bool] = {}

    expected_hash = run_info["selected_checkpoint"]["weights_sha256"]
    _check(checks, "selected_checkpoint_hash_matches_original_test", original["checkpoint_sha256"] == expected_hash)
    _check(checks, "selected_checkpoint_hash_matches_entity_export", entity["metadata"]["checkpoint_sha256"] == expected_hash)
    _check(checks, "test_was_selected_by_validation_only", original["selected_by_validation_only"] is True)
    _check(checks, "link_score_threshold_is_0_5", original["metrics"]["score_threshold"] == 0.5)
    _check(checks, "ned_threshold_is_0_2", direct["metadata"]["ned_threshold"] == 0.2)
    _check(checks, "iou_threshold_is_0_3", direct["metadata"]["iou_threshold"] == 0.3)
    _check(checks, "no_document_specific_pixels", entity["metadata"]["pixel_min"] == entity["metadata"]["pixel_max"] == 1.0)
    _check(checks, "direct_regular_matches_original_test", direct["results"]["regular"] == original["metrics"]["official_pair_metrics"]["regular"])
    _check(checks, "direct_all_matches_original_test", direct["results"]["all"] == original["metrics"]["official_pair_metrics"]["all"])
    _check(checks, "postprocessing_preserves_regular_overall", direct["results"]["regular"] == post["results"]["regular"])
    _check(
        checks,
        "postprocessing_preserves_regular_per_cluster",
        all(
            direct["cluster_results"][cluster]["results"]["regular"]
            == post["cluster_results"][cluster]["results"]["regular"]
            for cluster in direct["cluster_results"]
        ),
    )

    matrix = entity["entity_confusion_matrix"]["rows_target_columns_prediction"]
    key_tp, value_tp = matrix[1][1], matrix[2][2]
    key_fp, key_fn = matrix[0][1] + matrix[2][1], matrix[1][0] + matrix[1][2]
    value_fp, value_fn = matrix[0][2] + matrix[1][2], matrix[2][0] + matrix[2][1]
    stored_entity = original["metrics"]["entity_metrics"]
    reconstructed = {
        "key": {"tp": key_tp, "fp": key_fp, "fn": key_fn},
        "value": {"tp": value_tp, "fp": value_fp, "fn": value_fn},
        "micro": {
            "tp": key_tp + value_tp,
            "fp": key_fp + value_fp,
            "fn": key_fn + value_fn,
        },
    }
    for label in ("key", "value"):
        _check(
            checks,
            f"confusion_matrix_reconstructs_{label}_counts",
            reconstructed[label]
            == {name: stored_entity["per_class"][label][name] for name in ("tp", "fp", "fn")},
        )
    _check(
        checks,
        "confusion_matrix_reconstructs_micro_counts",
        reconstructed["micro"] == {name: stored_entity["micro"][name] for name in ("tp", "fp", "fn")},
    )

    direct_all = direct["results"]["all"]
    post_all = post["results"]["all"]
    all_f1_change = {
        mode: post_all[mode]["f1"] - direct_all[mode]["f1"]
        for mode in ("text_only", "location_only", "text_location")
    }
    _check(checks, "postprocessing_improves_all_f1_in_all_modes", all(value > 0.0 for value in all_f1_change.values()))

    write_json(
        args.output,
        {
            "verified_at": utc_now(),
            "status": "pass",
            "checks": checks,
            "checkpoint_sha256": expected_hash,
            "reconstructed_entity_counts": reconstructed,
            "postprocessing_all_f1_change": all_f1_change,
            "regular_unchanged_by_postprocessing": True,
        },
    )


if __name__ == "__main__":
    main()
