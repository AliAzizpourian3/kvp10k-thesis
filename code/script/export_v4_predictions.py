"""Export saved V4 diagnostic predictions in standard KVP10k JSON format."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--ground_truth_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    with args.records.open(encoding="utf-8") as handle:
        records = json.load(handle)["records"]

    predictions_by_document = defaultdict(list)
    for record in records:
        if record.get("record_type") != "prediction":
            continue
        predictions_by_document[record["document_id"]].append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    document_count = 0
    prediction_count = 0
    for ground_truth_path in sorted(args.ground_truth_dir.glob("*.json")):
        document_id = ground_truth_path.stem
        records_for_document = sorted(
            predictions_by_document.get(document_id, []),
            key=lambda record: record["link_confidence"],
            reverse=True,
        )
        kvps = [
            {
                "type": "kvp",
                "key": {
                    "text": record["predicted_key"],
                    "bbox": record["predicted_key_box"],
                },
                "value": {
                    "text": record["predicted_value"],
                    "bbox": record["predicted_value_box"],
                },
                "link_confidence": record["link_confidence"],
            }
            for record in records_for_document
        ]
        with (args.output_dir / ground_truth_path.name).open("w", encoding="utf-8") as handle:
            json.dump({"kvps_list": kvps}, handle, indent=2)
        document_count += 1
        prediction_count += len(kvps)

    print(f"Exported {prediction_count} predictions across {document_count} documents")


if __name__ == "__main__":
    main()