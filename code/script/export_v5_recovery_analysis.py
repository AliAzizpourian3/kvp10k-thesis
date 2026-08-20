"""Export V5 all-type predictions and token-level entity error candidates.

The all-type decoder is imported unchanged from ``evaluate_stage4b``.  A model
forward hook captures the same pass's entity logits, avoiding a second GPU
inference pass while leaving the historical recovery implementation untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader

from evaluate_stage4b import _extract_all_type_predictions, _load_checkpoint
from export_v4_all_types import _to_kvp
from layoutlm_model_v2 import create_model
from stage4_kvp_dataset import LayoutLMv3PreparedDataset, PaddedBatchCollator
from train_stage4b_v5 import raw_sample_context, sha256_file, utc_now, write_json


LABEL_NAMES = {0: "OTHER", 1: "KEY", 2: "VALUE"}


def entity_error_kind(target: int, prediction: int) -> str | None:
    if target == prediction:
        return None
    if target == 1 and prediction == 2:
        return "key_predicted_as_value"
    if target == 2 and prediction == 1:
        return "value_predicted_as_key"
    if target in (1, 2) and prediction == 0:
        return "missed_entity"
    if target == 0 and prediction in (1, 2):
        return "spurious_entity"
    return "other_entity_error"


def _read_cluster_map(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        document_id: str(item.get("cluster", "geometry_unavailable"))
        for document_id, item in raw.items()
    }


def _readable_word(text: str) -> bool:
    text = text.strip()
    return 1 <= len(text) <= 80 and any(character.isalnum() for character in text)


def collect_entity_errors(
    outputs: dict,
    batch: dict,
    dataset: LayoutLMv3PreparedDataset,
    index_start: int,
    cluster_map: dict[str, str],
    confusion: list[list[int]],
    candidates: dict[str, list[dict]],
) -> None:
    probabilities = torch.softmax(outputs["entity_logits"].detach().cpu(), dim=-1)
    predictions = probabilities.argmax(dim=-1)
    labels = batch["entity_labels"].detach().cpu()
    attention = batch["attention_mask"].detach().cpu()

    for offset in range(labels.shape[0]):
        sample_index = index_start + offset
        data, words, boxes, word_ids, json_path = raw_sample_context(dataset, sample_index)
        document_id = data.get("hash_name", json_path.stem)
        seen_words: set[int] = set()
        for token_index in range(labels.shape[1]):
            if not int(attention[offset, token_index]):
                continue
            target = int(labels[offset, token_index])
            prediction = int(predictions[offset, token_index])
            if target not in LABEL_NAMES or prediction not in LABEL_NAMES:
                continue
            confusion[target][prediction] += 1
            word_id = word_ids[token_index] if token_index < len(word_ids) else None
            if word_id is None or word_id in seen_words or word_id >= len(words):
                continue
            seen_words.add(word_id)
            kind = entity_error_kind(target, prediction)
            word = words[word_id]
            if kind is None or not _readable_word(word):
                continue
            confidence = float(probabilities[offset, token_index, prediction])
            candidates[kind].append(
                {
                    "document_id": document_id,
                    "source_json": str(json_path),
                    "cluster": cluster_map.get(document_id, "geometry_unavailable"),
                    "word": word,
                    "bbox": boxes[word_id] if word_id < len(boxes) else None,
                    "context": " ".join(words[max(0, word_id - 5): word_id + 6]),
                    "target": LABEL_NAMES[target],
                    "prediction": LABEL_NAMES[prediction],
                    "prediction_confidence": confidence,
                    "token_index": token_index,
                    "word_index": word_id,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_root", type=Path, required=True)
    parser.add_argument("--expected_sha256", required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--cluster_map", type=Path, required=True)
    parser.add_argument("--prediction_dir", type=Path, required=True)
    parser.add_argument("--entity_output", type=Path, required=True)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_documents", type=int)
    parser.add_argument("--candidate_count", type=int, default=50)
    args = parser.parse_args()

    selected_weights = args.checkpoint_root / "best_model" / "pytorch_model.bin"
    actual_sha256 = sha256_file(selected_weights)
    if actual_sha256 != args.expected_sha256:
        raise RuntimeError(
            f"Selected checkpoint SHA-256 mismatch: {actual_sha256} != {args.expected_sha256}"
        )
    if args.prediction_dir.exists() or args.entity_output.exists():
        raise FileExistsError("Refusing to overwrite V5 final-analysis output")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V5 recovery export requires a CUDA allocation")
    model = create_model(use_linker=True, device=device)
    model = _load_checkpoint(model, args.checkpoint_root, device)
    model.eval()

    dataset = LayoutLMv3PreparedDataset(
        data_dir=str(args.data_dir),
        split="test",
        processor=model.encoder.processor if hasattr(model.encoder, "processor") else None,
        max_seq_length=512,
        include_images=False,
    )
    document_limit = min(args.max_documents or len(dataset), len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PaddedBatchCollator(),
        num_workers=0,
    )
    cluster_map = _read_cluster_map(args.cluster_map)
    args.prediction_dir.mkdir(parents=True)

    captured: list[dict] = []
    hook = model.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    counts: Counter[str] = Counter()
    confusion = [[0, 0, 0] for _ in range(3)]
    candidates: dict[str, list[dict]] = defaultdict(list)
    visual_min = float("inf")
    visual_max = float("-inf")
    index_start = 0
    try:
        for batch in loader:
            if index_start >= document_limit:
                break
            captured.clear()
            batch_predictions = _extract_all_type_predictions(
                model, batch, device, dataset, index_start, args.score_threshold
            )
            if len(captured) != 1:
                raise RuntimeError(f"Expected one captured model output, got {len(captured)}")
            collect_entity_errors(
                captured[0], batch, dataset, index_start, cluster_map, confusion, candidates
            )
            pixels = batch["pixel_values"]
            visual_min = min(visual_min, float(pixels.min()))
            visual_max = max(visual_max, float(pixels.max()))
            for offset, predictions in enumerate(batch_predictions):
                sample_index = index_start + offset
                if sample_index >= document_limit:
                    break
                ordered = sorted(
                    predictions, key=lambda item: item.get("confidence", 0.0), reverse=True
                )
                kvps = [_to_kvp(prediction) for prediction in ordered]
                for prediction in ordered:
                    counts[prediction["type"]] += 1
                write_json(args.prediction_dir / dataset.json_files[sample_index].name, {"kvps_list": kvps})
            index_start += len(batch_predictions)
    finally:
        hook.remove()

    if index_start < document_limit:
        raise RuntimeError(f"Processed {index_start}, expected {document_limit} documents")
    if visual_min != 1.0 or visual_max != 1.0:
        raise RuntimeError(f"Expected constant blank visual tensor, got [{visual_min}, {visual_max}]")

    top_examples = {
        kind: sorted(
            records,
            key=lambda item: (-item["prediction_confidence"], item["document_id"], item["word_index"]),
        )[: args.candidate_count]
        for kind, records in sorted(candidates.items())
    }
    source = inspect.getsource(_extract_all_type_predictions).encode("utf-8")
    write_json(
        args.entity_output,
        {
            "metadata": {
                "created_at": utc_now(),
                "checkpoint_path": str(selected_weights.resolve()),
                "checkpoint_sha256": actual_sha256,
                "documents": document_limit,
                "score_threshold": args.score_threshold,
                "ned_threshold": 0.2,
                "iou_threshold": 0.3,
                "recovery_function": "evaluate_stage4b._extract_all_type_predictions",
                "recovery_function_source_sha256": hashlib.sha256(source).hexdigest(),
                "visual_input": "constant blank processor tensor",
                "pixel_min": visual_min,
                "pixel_max": visual_max,
            },
            "prediction_counts": dict(sorted(counts.items())),
            "entity_confusion_matrix": {
                "labels": ["OTHER", "KEY", "VALUE"],
                "rows_target_columns_prediction": confusion,
                "key_predicted_as_value": confusion[1][2],
                "value_predicted_as_key": confusion[2][1],
            },
            "candidate_counts": {kind: len(records) for kind, records in sorted(candidates.items())},
            "top_examples": top_examples,
        },
    )


if __name__ == "__main__":
    main()
