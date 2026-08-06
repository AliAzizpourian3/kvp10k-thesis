"""Re-score saved regular-KVP predictions under official macro aggregation."""

import argparse
import json
from pathlib import Path

from evaluate_kvp10k_benchmark import load_documents
from kvp10k_official_eval import evaluate_documents


SETTINGS = (
    ("official", "text_location", 0.2, 0.3),
    ("relaxed_location", "text_location", 0.2, 0.1),
    ("relaxed_text", "text_location", 0.3, 0.3),
    ("text_only", "text_only", 0.2, 0.3),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_dir", required=True, type=Path)
    parser.add_argument("--ground_truth_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    documents = load_documents(args.prediction_dir, args.ground_truth_dir)
    measurements = []
    for name, mode, ned_threshold, iou_threshold in SETTINGS:
        result = evaluate_documents(
            documents,
            mode,
            kvp_type="kvp",
            ned_threshold=ned_threshold,
            iou_threshold=iou_threshold,
        )
        measurements.append({
            "setting": name,
            "mode": mode,
            "ned_threshold": ned_threshold,
            "iou_threshold": None if mode == "text_only" else iou_threshold,
            **{key: value for key, value in result.items() if key != "per_document"},
        })

    output = {
        "metadata": {
            "aggregation": "IBM KVP10k macro per-document",
            "category": "regular KVP",
            "predictions_fixed": True,
        },
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    for measurement in measurements:
        print(f"{measurement['setting']:20s} F1={measurement['f1']:.4f}")


if __name__ == "__main__":
    main()