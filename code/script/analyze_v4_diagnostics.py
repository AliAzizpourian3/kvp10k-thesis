"""Focused V4 matching-threshold and qualitative-candidate analysis.

The model is run once at the fixed link-score operating threshold. The exact
same predicted pairs are then re-scored under four diagnostic settings:

* legacy combined (stored as ``official`` for compatibility): NED < 0.2 and IoU > 0.3
* relaxed location: NED < 0.2 and IoU > 0.1
* relaxed text: NED < 0.3 and IoU > 0.3
* text only: NED < 0.2, without an IoU condition

Matching preserves the evaluator's confidence-ordered greedy, one-to-one
assignment. Outputs include aggregate CSV/JSON measurements, per-record
diagnostics, and candidate examples for subsequent manual review. This script
does not tune on the test set or alter the official headline metric.
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch.utils.data import DataLoader

from evaluate_mistral import _iou, _ned
from evaluate_stage4b import _collect_link_pairs, _get_gt_kvps, _load_checkpoint
from layoutlm_model_v2 import create_model as create_model_v2
from stage4_kvp_dataset import LayoutLMv3PreparedDataset, PaddedBatchCollator


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


SETTINGS = (
    {"name": "official", "ned": 0.2, "iou": 0.3, "use_bbox": True},
    {"name": "relaxed_location", "ned": 0.2, "iou": 0.1, "use_bbox": True},
    {"name": "relaxed_text", "ned": 0.3, "iou": 0.3, "use_bbox": True},
    {"name": "text_only", "ned": 0.2, "iou": None, "use_bbox": False},
)


def _valid_box(box):
    return box is not None and len(box) >= 4


def _pair_metrics(prediction, ground_truth):
    pred_key, pred_value, _confidence, pred_key_box, pred_value_box = prediction
    gt_key, gt_value, gt_key_box, gt_value_box = ground_truth
    return {
        "key_ned": _ned(pred_key.strip().lower(), gt_key.strip().lower()),
        "value_ned": _ned(pred_value.strip().lower(), gt_value.strip().lower()),
        "key_iou": _iou(pred_key_box, gt_key_box)
        if _valid_box(pred_key_box) and _valid_box(gt_key_box) else None,
        "value_iou": _iou(pred_value_box, gt_value_box)
        if _valid_box(pred_value_box) and _valid_box(gt_value_box) else None,
    }


def _passes(metrics, prediction, ground_truth, setting):
    # This diagnostic follows the paper's strict boundary wording.
    if metrics["key_ned"] >= setting["ned"] or metrics["value_ned"] >= setting["ned"]:
        return False
    if not setting["use_bbox"]:
        return True

    # Preserve evaluate_stage4b._bbox_ok semantics: absent or malformed boxes
    # do not reject a text match because no reliable geometry is available.
    pred_key_box, pred_value_box = prediction[3], prediction[4]
    gt_key_box, gt_value_box = ground_truth[2], ground_truth[3]
    if _valid_box(pred_key_box) and _valid_box(gt_key_box):
        if metrics["key_iou"] <= setting["iou"]:
            return False
    if _valid_box(pred_value_box) and _valid_box(gt_value_box):
        if metrics["value_iou"] <= setting["iou"]:
            return False
    return True


def _match_document(predictions, ground_truths, setting):
    """Match exactly as the existing evaluator: confidence order, first valid GT."""
    metric_cache = {}
    matched_predictions = {}
    matched_ground_truths = {}
    prediction_order = sorted(
        range(len(predictions)), key=lambda index: predictions[index][2], reverse=True
    )

    for prediction_index in prediction_order:
        prediction = predictions[prediction_index]
        for gt_index, ground_truth in enumerate(ground_truths):
            if gt_index in matched_ground_truths:
                continue
            cache_key = (prediction_index, gt_index)
            metrics = metric_cache.setdefault(
                cache_key, _pair_metrics(prediction, ground_truth)
            )
            if not _passes(metrics, prediction, ground_truth, setting):
                continue
            matched_predictions[prediction_index] = gt_index
            matched_ground_truths[gt_index] = prediction_index
            break

    return matched_predictions, matched_ground_truths, metric_cache


def _nearest_ground_truth(prediction, ground_truths):
    if not ground_truths:
        return None, None
    candidates = []
    for gt_index, ground_truth in enumerate(ground_truths):
        metrics = _pair_metrics(prediction, ground_truth)
        # Text drives this explanatory nearest-neighbour choice; available IoU
        # breaks ties but never changes official matching.
        iou_bonus = sum(
            value for value in (metrics["key_iou"], metrics["value_iou"])
            if value is not None
        )
        rank = metrics["key_ned"] + metrics["value_ned"] - 0.01 * iou_bonus
        candidates.append((rank, gt_index, metrics))
    _, gt_index, metrics = min(candidates, key=lambda item: item[0])
    return gt_index, metrics


def _nearest_prediction(ground_truth, predictions):
    if not predictions:
        return None, None
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        metrics = _pair_metrics(prediction, ground_truth)
        iou_bonus = sum(
            value for value in (metrics["key_iou"], metrics["value_iou"])
            if value is not None
        )
        rank = metrics["key_ned"] + metrics["value_ned"] - 0.01 * iou_bonus
        candidates.append((rank, prediction_index, metrics))
    _, prediction_index, metrics = min(candidates, key=lambda item: item[0])
    return prediction_index, metrics


def _failure_reason(prediction, ground_truths, nearest_metrics):
    if not ground_truths or nearest_metrics is None:
        return "no_ground_truth_pair"

    key_matches = {
        index for index, ground_truth in enumerate(ground_truths)
        if _ned(prediction[0].lower(), ground_truth[0].lower()) < 0.2
    }
    value_matches = {
        index for index, ground_truth in enumerate(ground_truths)
        if _ned(prediction[1].lower(), ground_truth[1].lower()) < 0.2
    }
    if key_matches and value_matches and not key_matches.intersection(value_matches):
        return "wrong_relation_candidate"

    text_ok = nearest_metrics["key_ned"] < 0.2 and nearest_metrics["value_ned"] < 0.2
    available_ious = (
        nearest_metrics["key_iou"], nearest_metrics["value_iou"]
    )
    location_ok = all(value is None or value > 0.3 for value in available_ious)
    if text_ok and not location_ok:
        return "localization"
    if location_ok and not text_ok:
        return "text_or_span_boundary"
    if not text_ok and not location_ok:
        return "text_and_localization_or_relation"
    return "one_to_one_assignment_conflict"


def _json_value(value):
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(value) for key, value in row.items()})


def _flatten_record(record):
    nearest = record.get("nearest_metrics") or {}
    outcomes = record.get("outcomes", {})
    return {
        "record_type": record["record_type"],
        "document_id": record["document_id"],
        "source_json": record["source_json"],
        "prediction_index": record.get("prediction_index"),
        "ground_truth_index": record.get("ground_truth_index"),
        "predicted_key": record.get("predicted_key"),
        "predicted_value": record.get("predicted_value"),
        "predicted_key_box": record.get("predicted_key_box"),
        "predicted_value_box": record.get("predicted_value_box"),
        "link_confidence": record.get("link_confidence"),
        "ground_truth_key": record.get("ground_truth_key"),
        "ground_truth_value": record.get("ground_truth_value"),
        "ground_truth_key_box": record.get("ground_truth_key_box"),
        "ground_truth_value_box": record.get("ground_truth_value_box"),
        "key_ned": nearest.get("key_ned"),
        "value_ned": nearest.get("value_ned"),
        "key_iou": nearest.get("key_iou"),
        "value_iou": nearest.get("value_iou"),
        "official_outcome": outcomes.get("official"),
        "relaxed_location_outcome": outcomes.get("relaxed_location"),
        "relaxed_text_outcome": outcomes.get("relaxed_text"),
        "text_only_outcome": outcomes.get("text_only"),
        "failure_reason": record.get("failure_reason"),
    }


def analyze(collected, json_files, candidate_count):
    setting_results = {
        setting["name"]: {"tp": 0, "total_pred": 0, "total_gt": 0}
        for setting in SETTINGS
    }
    records = []
    document_matches = []

    for document_index, ((predictions, ground_truths), json_file) in enumerate(
        zip(collected, json_files)
    ):
        gt_pairs, document_id = _get_gt_kvps(json_file)
        if gt_pairs != ground_truths:
            raise RuntimeError(f"Ground-truth order changed for {json_file}")
        if not document_id:
            document_id = Path(json_file).stem

        matches = {}
        for setting in SETTINGS:
            pred_matches, gt_matches, _ = _match_document(
                predictions, ground_truths, setting
            )
            matches[setting["name"]] = {
                "pred": pred_matches,
                "gt": gt_matches,
            }
            result = setting_results[setting["name"]]
            result["tp"] += len(pred_matches)
            result["total_pred"] += len(predictions)
            result["total_gt"] += len(ground_truths)

        document_matches.append(matches)

        for prediction_index, prediction in enumerate(predictions):
            nearest_gt_index, nearest_metrics = _nearest_ground_truth(
                prediction, ground_truths
            )
            matched_gt_indices = {
                name: setting_match["pred"].get(prediction_index)
                for name, setting_match in matches.items()
            }
            reference_gt_index = next(
                (
                    matched_gt_indices[name]
                    for name in (
                        "official", "relaxed_location", "relaxed_text", "text_only"
                    )
                    if matched_gt_indices[name] is not None
                ),
                nearest_gt_index,
            )
            reference_gt = (
                ground_truths[reference_gt_index]
                if reference_gt_index is not None else None
            )
            reference_metrics = (
                _pair_metrics(prediction, reference_gt) if reference_gt else None
            )
            outcomes = {
                name: "TP" if prediction_index in setting_match["pred"] else "FP"
                for name, setting_match in matches.items()
            }
            record = {
                "record_type": "prediction",
                "document_id": document_id,
                "source_json": str(json_file),
                "document_index": document_index,
                "prediction_index": prediction_index,
                "predicted_key": prediction[0],
                "predicted_value": prediction[1],
                "link_confidence": prediction[2],
                "predicted_key_box": prediction[3],
                "predicted_value_box": prediction[4],
                "ground_truth_index": reference_gt_index,
                "ground_truth_key": reference_gt[0] if reference_gt else None,
                "ground_truth_value": reference_gt[1] if reference_gt else None,
                "ground_truth_key_box": reference_gt[2] if reference_gt else None,
                "ground_truth_value_box": reference_gt[3] if reference_gt else None,
                "nearest_ground_truth_index": nearest_gt_index,
                "nearest_metrics": reference_metrics,
                "outcomes": outcomes,
                "matched_ground_truth_indices": matched_gt_indices,
                "failure_reason": None,
            }
            if outcomes["official"] == "FP":
                record["failure_reason"] = _failure_reason(
                    prediction, ground_truths, nearest_metrics
                )
            records.append(record)

        for gt_index, ground_truth in enumerate(ground_truths):
            nearest_prediction_index, nearest_metrics = _nearest_prediction(
                ground_truth, predictions
            )
            nearest_prediction = (
                predictions[nearest_prediction_index]
                if nearest_prediction_index is not None else None
            )
            outcomes = {
                name: "matched" if gt_index in setting_match["gt"] else "FN"
                for name, setting_match in matches.items()
            }
            records.append({
                "record_type": "ground_truth",
                "document_id": document_id,
                "source_json": str(json_file),
                "document_index": document_index,
                "prediction_index": nearest_prediction_index,
                "predicted_key": nearest_prediction[0] if nearest_prediction else None,
                "predicted_value": nearest_prediction[1] if nearest_prediction else None,
                "link_confidence": nearest_prediction[2] if nearest_prediction else None,
                "predicted_key_box": nearest_prediction[3] if nearest_prediction else None,
                "predicted_value_box": nearest_prediction[4] if nearest_prediction else None,
                "ground_truth_index": gt_index,
                "ground_truth_key": ground_truth[0],
                "ground_truth_value": ground_truth[1],
                "ground_truth_key_box": ground_truth[2],
                "ground_truth_value_box": ground_truth[3],
                "nearest_metrics": nearest_metrics,
                "outcomes": outcomes,
                "failure_reason": "unmatched_ground_truth"
                if outcomes["official"] == "FN" else None,
            })

    measurements = []
    for setting in SETTINGS:
        result = setting_results[setting["name"]]
        tp = result["tp"]
        total_pred = result["total_pred"]
        total_gt = result["total_gt"]
        fp = total_pred - tp
        fn = total_gt - tp
        precision = tp / total_pred if total_pred else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        measurements.append({
            "setting": setting["name"],
            "ned_threshold": setting["ned"],
            "iou_threshold": setting["iou"],
            "uses_location": setting["use_bbox"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predictions": total_pred,
            "ground_truth": total_gt,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    prediction_records = [r for r in records if r["record_type"] == "prediction"]
    gt_records = [r for r in records if r["record_type"] == "ground_truth"]

    spatial = [
        r for r in prediction_records
        if r["outcomes"]["official"] == "FP"
        and r["outcomes"]["relaxed_location"] == "TP"
    ]
    textual = [
        r for r in prediction_records
        if r["outcomes"]["official"] == "FP"
        and r["outcomes"]["relaxed_text"] == "TP"
    ]
    persistent_fp = [
        r for r in prediction_records
        if all(outcome == "FP" for outcome in r["outcomes"].values())
    ]
    persistent_fn = [
        r for r in gt_records
        if all(outcome == "FN" for outcome in r["outcomes"].values())
    ]
    official_tp = [
        r for r in prediction_records if r["outcomes"]["official"] == "TP"
    ]

    spatial.sort(key=lambda r: r["link_confidence"], reverse=True)
    textual.sort(key=lambda r: r["link_confidence"], reverse=True)
    persistent_fp.sort(key=lambda r: r["link_confidence"], reverse=True)
    persistent_fn.sort(
        key=lambda r: (
            (r["nearest_metrics"] or {}).get("key_ned", 2.0)
            + (r["nearest_metrics"] or {}).get("value_ned", 2.0)
        )
    )
    official_tp.sort(key=lambda r: r["link_confidence"], reverse=True)

    candidates = {
        "spatial_near_misses": spatial[:candidate_count],
        "text_near_misses": textual[:candidate_count],
        "persistent_false_positives": persistent_fp[:candidate_count],
        "persistent_false_negatives": persistent_fn[:candidate_count],
        "official_true_positives": official_tp[:candidate_count],
        "available_counts": {
            "spatial_near_misses": len(spatial),
            "text_near_misses": len(textual),
            "persistent_false_positives": len(persistent_fp),
            "persistent_false_negatives": len(persistent_fn),
            "official_true_positives": len(official_tp),
        },
    }
    return measurements, records, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", default="data/outputs/stage4b_v4")
    parser.add_argument("--data_dir", default="data/prepared")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--candidate_count", type=int, default=5)
    parser.add_argument(
        "--output_dir",
        default="data/outputs/stage4b_v4/diagnostic_analysis",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    model = create_model_v2(use_linker=True, device=device)
    model = _load_checkpoint(model, args.checkpoint_dir, device)
    model.eval()

    processor = model.encoder.processor if hasattr(model.encoder, "processor") else None
    dataset = LayoutLMv3PreparedDataset(
        data_dir=args.data_dir,
        split="test",
        processor=processor,
        max_seq_length=512,
        include_images=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PaddedBatchCollator(),
        num_workers=0,
    )

    logger.info(
        "Running one inference pass at fixed score threshold %.2f", args.score_threshold
    )
    collected = _collect_link_pairs(
        model, dataset, dataloader, device, args.score_threshold, oracle=False
    )
    measurements, records, candidates = analyze(
        collected, dataset.json_files, args.candidate_count
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_dir": args.checkpoint_dir,
        "split": "test",
        "score_threshold": args.score_threshold,
        "metric_scope": "legacy pooled regular-pair diagnostic; not IBM macro benchmark",
        "matching": "confidence-ordered greedy one-to-one",
        "comparison_operators": "NED < threshold; IoU > threshold",
        "settings": list(SETTINGS),
    }
    with open(output_dir / "measurements.json", "w", encoding="utf-8") as output_file:
        json.dump({"metadata": metadata, "measurements": measurements}, output_file, indent=2)
    _write_csv(output_dir / "measurements.csv", measurements)

    with open(output_dir / "diagnostic_records.json", "w", encoding="utf-8") as output_file:
        json.dump({"metadata": metadata, "records": records}, output_file, indent=2)
    _write_csv(
        output_dir / "diagnostic_records.csv",
        [_flatten_record(record) for record in records],
    )

    with open(output_dir / "candidate_examples.json", "w", encoding="utf-8") as output_file:
        json.dump({"metadata": metadata, **candidates}, output_file, indent=2)
    candidate_rows = []
    for category, category_records in candidates.items():
        if category == "available_counts":
            continue
        for record in category_records:
            row = {"category": category, **_flatten_record(record)}
            candidate_rows.append(row)
    _write_csv(output_dir / "candidate_examples.csv", candidate_rows)

    official = next(row for row in measurements if row["setting"] == "official")
    relaxed_location = next(
        row for row in measurements if row["setting"] == "relaxed_location"
    )
    relaxed_text = next(
        row for row in measurements if row["setting"] == "relaxed_text"
    )
    summary = {
        "paper_literal_pooled_f1": official["f1"],
        "f1_gain_relaxed_location": relaxed_location["f1"] - official["f1"],
        "f1_gain_relaxed_text": relaxed_text["f1"] - official["f1"],
        "candidate_counts": candidates["available_counts"],
        "note": "Candidate failure reasons are heuristic and require manual review before thesis use.",
    }
    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print("\nV4 DIAGNOSTIC MEASUREMENTS")
    print("=" * 88)
    for row in measurements:
        print(
            f"{row['setting']:<20} TP={row['tp']:4d} FP={row['fp']:4d} "
            f"FN={row['fn']:4d} P={row['precision']:.4f} "
            f"R={row['recall']:.4f} F1={row['f1']:.4f}"
        )
    print("=" * 88)
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()