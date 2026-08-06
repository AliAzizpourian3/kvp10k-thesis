"""Evaluate saved predictions with IBM KVP10k's released benchmark semantics."""

import argparse
import json
from pathlib import Path

from kvp10k_official_eval import evaluate_table


def _load_kvps(path: Path, ground_truth: bool) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if ground_truth:
        data = data.get("gt_kvps", data)
    return data.get("kvps_list", [])


def load_documents(prediction_dir: Path, ground_truth_dir: Path) -> list[dict]:
    documents = []
    for ground_truth_path in sorted(ground_truth_dir.glob("*.json")):
        prediction_path = prediction_dir / ground_truth_path.name
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {prediction_path}")
        documents.append({
            "document_id": ground_truth_path.stem,
            "predictions": _load_kvps(prediction_path, ground_truth=False),
            "ground_truths": _load_kvps(ground_truth_path, ground_truth=True),
        })
    return documents


def _compact(table: dict) -> dict:
    return {
        category: {
            mode: {key: value for key, value in metrics.items() if key != "per_document"}
            for mode, metrics in modes.items()
        }
        for category, modes in table.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_dir", required=True, type=Path)
    parser.add_argument("--ground_truth_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cluster_map", type=Path)
    parser.add_argument("--ned_threshold", type=float, default=0.2)
    parser.add_argument("--iou_threshold", type=float, default=0.3)
    args = parser.parse_args()

    documents = load_documents(args.prediction_dir, args.ground_truth_dir)
    table = evaluate_table(
        documents,
        ned_threshold=args.ned_threshold,
        iou_threshold=args.iou_threshold,
    )
    output = {
        "metadata": {
            "implementation": "IBM/KVP10k benchmark-compatible",
            "reference": "https://github.com/IBM/KVP10k/blob/main/benchmark/metrics_calculator.py",
            "aggregation": "macro precision/recall over documents with non-empty filtered GT",
            "f1": "harmonic mean of macro precision and macro recall",
            "matching": "prediction-order greedy one-to-one with equal KVP type",
            "ned_comparison": "strict (<)",
            "iou_comparison": "inclusive (>=), matching released IBM code",
            "paper_code_discrepancy": "paper says IoU exceeds threshold; released code uses >=",
            "malformed_box_policy": "fail location match instead of raising an exception",
            "ned_threshold": args.ned_threshold,
            "iou_threshold": args.iou_threshold,
            "documents_loaded": len(documents),
        },
        "results": _compact(table),
    }
    if args.cluster_map:
        with args.cluster_map.open(encoding="utf-8") as handle:
            cluster_map = json.load(handle)
        cluster_names = sorted({str(item["cluster"]) for item in cluster_map.values()})
        output["cluster_results"] = {}
        for cluster_name in cluster_names:
            cluster_documents = [
                document
                for document in documents
                if str(cluster_map.get(document["document_id"], {}).get("cluster"))
                == cluster_name
            ]
            output["cluster_results"][cluster_name] = {
                "documents": len(cluster_documents),
                "results": _compact(
                    evaluate_table(
                        cluster_documents,
                        ned_threshold=args.ned_threshold,
                        iou_threshold=args.iou_threshold,
                    )
                ),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Evaluated {len(documents)} documents")
    for category, modes in output["results"].items():
        values = "  ".join(
            f"{mode} F1={metrics['f1']:.4f}" for mode, metrics in modes.items()
        )
        print(f"{category:10s} {values}")
    for cluster_name, cluster in output.get("cluster_results", {}).items():
        regular = cluster["results"]["regular"]
        print(
            f"cluster {cluster_name} ({cluster['documents']} docs)  "
            f"regular text F1={regular['text_only']['f1']:.4f}  "
            f"regular text+location F1={regular['text_location']['f1']:.4f}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()