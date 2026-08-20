"""Select a V4 checkpoint by official validation pair F1, then test it once.

This is an evaluation-only utility. It never updates model weights. The script:

1. discovers and SHA-256-deduplicates ``best_model`` and ``checkpoint-*``;
2. reconstructs the exact seed-42 validation split used during training;
3. runs every unique checkpoint on validation at link threshold 0.5;
4. selects the highest regular text+location macro F1 from the released-style
   KVP10k evaluator; and
5. evaluates only that selected checkpoint on the test split.

For comparability, entity output contains both the historical implementation
and corrected one-vs-rest KEY/VALUE counts. The primary corrected entity metric
is micro-averaged over KEY and VALUE; per-class and macro metrics are included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader, random_split

from evaluate_stage4b import _span_to_bbox, _span_to_text
from kvp10k_official_eval import evaluate_table
from layoutlm_model_v2 import create_model as create_model_v2
from stage4_kvp_dataset import (
    LayoutLMv3PreparedDataset,
    PaddedBatchCollator,
    _load_processor,
)


LOGGER = logging.getLogger(__name__)
SCORE_THRESHOLD = 0.5
NED_THRESHOLD = 0.2
IOU_THRESHOLD = 0.3
SELECTION_CATEGORY = "regular"
SELECTION_MODE = "text_location"
ENTITY_CLASSES = {1: "key", 2: "value"}


@dataclass(frozen=True)
class CheckpointCandidate:
    """One unique checkpoint payload and every path that aliases it."""

    candidate_id: str
    canonical_path: Path
    sha256: str
    aliases: tuple[Path, ...]

    def to_dict(self, root: Path | None = None) -> dict:
        def display(path: Path) -> str:
            if root is not None:
                try:
                    return str(path.relative_to(root))
                except ValueError:
                    pass
            return str(path)

        return {
            "candidate_id": self.candidate_id,
            "canonical_path": display(self.canonical_path),
            "sha256": self.sha256,
            "aliases": [display(path) for path in self.aliases],
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    parent = path.parent.name
    if parent.startswith("checkpoint-"):
        try:
            return (0, int(parent.split("-")[-1]), str(path))
        except ValueError:
            return (0, sys.maxsize, str(path))
    if parent == "best_model":
        return (1, 0, str(path))
    return (2, 0, str(path))


def discover_unique_checkpoints(
    checkpoint_root: Path,
    hash_function: Callable[[Path], str] = sha256_file,
) -> list[CheckpointCandidate]:
    """Find V4 model files and group byte-identical payloads by SHA-256."""
    paths = []
    best = checkpoint_root / "best_model" / "pytorch_model.bin"
    if best.is_file():
        paths.append(best.resolve())
    paths.extend(
        path.resolve()
        for path in checkpoint_root.glob("checkpoint-*/pytorch_model.bin")
        if path.is_file()
    )
    paths = sorted(set(paths), key=checkpoint_sort_key)
    if not paths:
        raise FileNotFoundError(f"No V4 checkpoint weights found under {checkpoint_root}")

    grouped: dict[str, list[Path]] = {}
    for index, path in enumerate(paths, start=1):
        LOGGER.info("Hashing checkpoint %d/%d: %s", index, len(paths), path)
        grouped.setdefault(hash_function(path), []).append(path)

    candidates = []
    for digest, aliases in grouped.items():
        aliases = sorted(aliases, key=checkpoint_sort_key)
        canonical = aliases[0]
        candidate_id = f"{canonical.parent.name}__{digest[:12]}"
        candidates.append(
            CheckpointCandidate(candidate_id, canonical, digest, tuple(aliases))
        )
    return sorted(candidates, key=lambda item: checkpoint_sort_key(item.canonical_path))


def new_entity_counts() -> dict:
    return {
        "legacy": {"tp": 0, "fp": 0, "fn": 0},
        "classes": {
            name: {"tp": 0, "fp": 0, "fn": 0}
            for name in ENTITY_CLASSES.values()
        },
    }


def update_entity_counts(
    counts: dict,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    """Accumulate both historical and corrected class-aware entity counts."""
    active = attention_mask == 1
    gt_entity = ((labels == 1) | (labels == 2)) & active
    pred_entity = ((predictions == 1) | (predictions == 2)) & active

    legacy = counts["legacy"]
    legacy["tp"] += int(((predictions == labels) & gt_entity).sum().item())
    legacy["fp"] += int((pred_entity & ~gt_entity).sum().item())
    legacy["fn"] += int((~pred_entity & gt_entity).sum().item())

    for class_id, name in ENTITY_CLASSES.items():
        predicted_class = (predictions == class_id) & active
        true_class = (labels == class_id) & active
        class_counts = counts["classes"][name]
        class_counts["tp"] += int((predicted_class & true_class).sum().item())
        class_counts["fp"] += int((predicted_class & ~true_class).sum().item())
        class_counts["fn"] += int((~predicted_class & true_class).sum().item())


def prf_from_counts(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def finalize_entity_metrics(counts: dict) -> dict:
    legacy = prf_from_counts(**counts["legacy"])
    per_class = {
        name: prf_from_counts(**class_counts)
        for name, class_counts in counts["classes"].items()
    }
    micro_counts = {
        key: sum(class_counts[key] for class_counts in counts["classes"].values())
        for key in ("tp", "fp", "fn")
    }
    corrected_micro = prf_from_counts(**micro_counts)
    corrected_macro = {
        metric: sum(values[metric] for values in per_class.values()) / len(per_class)
        for metric in ("precision", "recall", "f1")
    }
    return {
        "primary_corrected_metric": "micro one-vs-rest over KEY and VALUE",
        "legacy_buggy": legacy,
        "corrected_micro": corrected_micro,
        "corrected_macro": corrected_macro,
        "per_class": per_class,
    }


def compact_official_table(table: dict) -> dict:
    return {
        category: {
            mode: {
                key: value
                for key, value in metrics.items()
                if key != "per_document"
            }
            for mode, metrics in modes.items()
        }
        for category, modes in table.items()
    }


def reconstruct_datasets(data_dir: Path, processor) -> tuple[LayoutLMv3PreparedDataset, LayoutLMv3PreparedDataset, dict]:
    full_train = LayoutLMv3PreparedDataset(
        data_dir=str(data_dir),
        split="train",
        processor=processor,
        max_seq_length=512,
        include_images=False,
    )
    total = len(full_train)
    validation_size = int(total * 0.1)
    training_size = total - validation_size
    _, validation_subset = random_split(
        full_train,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )
    validation_indices = sorted(validation_subset.indices)
    full_train.json_files = [full_train.json_files[index] for index in validation_indices]

    test = LayoutLMv3PreparedDataset(
        data_dir=str(data_dir),
        split="test",
        processor=processor,
        max_seq_length=512,
        include_images=False,
    )
    split_metadata = {
        "source_train_pages": total,
        "training_pages": training_size,
        "validation_pages": validation_size,
        "validation_fraction": 0.1,
        "seed": 42,
        "validation_indices": validation_indices,
        "validation_document_ids": [path.stem for path in full_train.json_files],
        "test_pages": len(test),
    }
    return full_train, test, split_metadata


def load_checkpoint_weights(model, candidate: CheckpointCandidate) -> None:
    state = torch.load(candidate.canonical_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(cleaned, strict=True)
    del state, cleaned
    model.eval()


def prepare_word_mapping(dataset: LayoutLMv3PreparedDataset, sample_index: int) -> tuple[dict, list[str], list[list[int]], list[int | None]]:
    json_path = dataset.json_files[sample_index]
    with json_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    words, word_bboxes = dataset._parse_lmdx_text(data.get("lmdx_text", ""), data)
    normalized = dataset._normalize_bboxes(
        word_bboxes,
        data.get("image_width", 1),
        data.get("image_height", 1),
    )
    from PIL import Image

    blank_image = Image.new("RGB", (224, 224), (255, 255, 255))
    encoded = dataset.processor(
        images=blank_image,
        text=words,
        boxes=normalized,
        return_tensors="pt",
        padding="max_length",
        max_length=dataset.max_seq_length,
        truncation=True,
    )
    return data, words, word_bboxes, encoded.word_ids()


def decode_regular_pairs(
    outputs: dict,
    dataset: LayoutLMv3PreparedDataset,
    sample_indices: list[int],
) -> list[tuple[dict, list[dict]]]:
    """Decode V2 outputs as confidence-ordered regular KVP predictions."""
    link_scores = outputs["link_scores"]
    key_spans = outputs["key_indices"]
    value_spans = outputs["value_indices"]
    decoded = []

    for batch_offset, sample_index in enumerate(sample_indices):
        data, words, word_bboxes, word_ids = prepare_word_mapping(dataset, sample_index)
        predictions = []
        scores = link_scores[batch_offset] if link_scores is not None else None
        keys = key_spans[batch_offset] if key_spans is not None else []
        values = value_spans[batch_offset] if value_spans is not None else []

        if scores is not None and len(keys) and len(values):
            best_value_positions = torch.argmax(scores, dim=1)
            best_scores = torch.sigmoid(
                scores[
                    torch.arange(len(keys), device=scores.device),
                    best_value_positions,
                ]
            )
            for key_position, key_span in enumerate(keys):
                confidence = float(best_scores[key_position].item())
                if confidence < SCORE_THRESHOLD:
                    continue
                value_position = int(best_value_positions[key_position].item())
                value_span = values[value_position]
                key_text = _span_to_text(
                    key_span[0], key_span[1], word_ids, words
                )
                value_text = _span_to_text(
                    value_span[0], value_span[1], word_ids, words
                )
                if not key_text or not value_text:
                    continue
                predictions.append(
                    {
                        "type": "kvp",
                        "key": {
                            "text": key_text,
                            "bbox": _span_to_bbox(
                                key_span[0], key_span[1], word_ids, word_bboxes
                            ),
                        },
                        "value": {
                            "text": value_text,
                            "bbox": _span_to_bbox(
                                value_span[0], value_span[1], word_ids, word_bboxes
                            ),
                        },
                        "link_confidence": confidence,
                    }
                )

        predictions.sort(key=lambda item: item["link_confidence"], reverse=True)
        decoded.append((data, predictions))
    return decoded


def evaluate_split(
    model,
    dataset: LayoutLMv3PreparedDataset,
    device: torch.device,
    batch_size: int,
    predictions_dir: Path,
    split_name: str,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PaddedBatchCollator(),
        num_workers=0,
    )
    predictions_dir.mkdir(parents=True, exist_ok=True)
    entity_counts = new_entity_counts()
    documents = []
    sample_index = 0

    for batch_index, batch in enumerate(loader, start=1):
        current_batch_size = batch["input_ids"].size(0)
        sample_indices = list(range(sample_index, sample_index + current_batch_size))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bbox = batch["bbox"].to(device)
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        with torch.inference_mode():
            outputs = model(input_ids, attention_mask, bbox, pixel_values)
        predicted_labels = torch.argmax(outputs["entity_logits"], dim=-1)
        update_entity_counts(
            entity_counts,
            predicted_labels.cpu(),
            batch["entity_labels"],
            batch["attention_mask"],
        )
        decoded = decode_regular_pairs(outputs, dataset, sample_indices)

        for offset, (data, predictions) in enumerate(decoded):
            json_path = dataset.json_files[sample_indices[offset]]
            document_id = data.get("hash_name", json_path.stem)
            ground_truths = data.get("gt_kvps", {}).get("kvps_list", [])
            documents.append(
                {
                    "document_id": document_id,
                    "predictions": predictions,
                    "ground_truths": ground_truths,
                }
            )
            write_json(predictions_dir / json_path.name, {"kvps_list": predictions})

        sample_index += current_batch_size
        if batch_index == 1 or batch_index % 50 == 0 or sample_index == len(dataset):
            LOGGER.info(
                "%s inference: %d/%d documents", split_name, sample_index, len(dataset)
            )

    if sample_index != len(dataset):
        raise RuntimeError(
            f"{split_name}: processed {sample_index} documents, expected {len(dataset)}"
        )
    official = compact_official_table(
        evaluate_table(
            documents,
            ned_threshold=NED_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
        )
    )
    return {
        "split": split_name,
        "documents": len(documents),
        "prediction_files": len(list(predictions_dir.glob("*.json"))),
        "score_threshold": SCORE_THRESHOLD,
        "official_thresholds": {
            "ned_strict_less_than": NED_THRESHOLD,
            "iou_inclusive_greater_equal": IOU_THRESHOLD,
        },
        "official_pair_metrics": official,
        "entity_metrics": finalize_entity_metrics(entity_counts),
    }


def selection_sort_key(result: dict) -> tuple[float, tuple[int, int, str]]:
    f1 = result["metrics"]["official_pair_metrics"][SELECTION_CATEGORY][SELECTION_MODE]["f1"]
    return (-f1, checkpoint_sort_key(Path(result["canonical_path"])))


def select_best_validation_result(results: list[dict]) -> dict:
    if not results:
        raise ValueError("No validation results available for checkpoint selection")
    return sorted(results, key=selection_sort_key)[0]


def reusable_result(path: Path, candidate: CheckpointCandidate, split: str) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    expected = (
        result.get("checkpoint", {}).get("sha256") == candidate.sha256
        and result.get("metrics", {}).get("split") == split
        and result.get("metrics", {}).get("score_threshold") == SCORE_THRESHOLD
    )
    if not expected:
        raise RuntimeError(f"Refusing to reuse incompatible result: {path}")
    return result


def run(args: argparse.Namespace) -> dict:
    checkpoint_root = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = discover_unique_checkpoints(checkpoint_root)
    LOGGER.info("Discovered %d unique checkpoint payloads", len(candidates))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    LOGGER.info("Evaluation device: %s", device)

    processor = _load_processor()
    validation_dataset, test_dataset, split_metadata = reconstruct_datasets(
        args.data_dir.resolve(), processor
    )
    manifest = {
        "status": "running",
        "started_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "checkpoint_root": str(checkpoint_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "batch_size": args.batch_size,
        "selection": {
            "category": SELECTION_CATEGORY,
            "mode": SELECTION_MODE,
            "score_threshold": SCORE_THRESHOLD,
            "ned_threshold": NED_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "tie_break": "numeric checkpoint order, then path",
        },
        "split": split_metadata,
        "candidates": [candidate.to_dict(checkpoint_root) for candidate in candidates],
    }
    write_json(output_dir / "run_manifest.json", manifest)

    model = create_model_v2(use_linker=True, device="cpu")
    model = model.to(device)
    validation_results = []
    for position, candidate in enumerate(candidates, start=1):
        candidate_dir = output_dir / "validation" / candidate.candidate_id
        result_path = candidate_dir / "result.json"
        prior = reusable_result(result_path, candidate, "validation")
        if prior is not None:
            LOGGER.info(
                "Reusing validation result %d/%d: %s",
                position,
                len(candidates),
                candidate.candidate_id,
            )
            validation_results.append(prior)
            continue

        LOGGER.info(
            "Evaluating validation checkpoint %d/%d: %s",
            position,
            len(candidates),
            candidate.candidate_id,
        )
        load_checkpoint_weights(model, candidate)
        metrics = evaluate_split(
            model,
            validation_dataset,
            device,
            args.batch_size,
            candidate_dir / "predictions",
            "validation",
        )
        result = {
            "checkpoint": candidate.to_dict(checkpoint_root),
            "canonical_path": str(candidate.canonical_path),
            "completed_at": utc_now(),
            "metrics": metrics,
        }
        write_json(result_path, result)
        validation_results.append(result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected = select_best_validation_result(validation_results)
    selected_sha = selected["checkpoint"]["sha256"]
    selected_candidate = next(
        candidate for candidate in candidates if candidate.sha256 == selected_sha
    )
    ranked = sorted(validation_results, key=selection_sort_key)
    summary = {
        "selection_metric": {
            "category": SELECTION_CATEGORY,
            "mode": SELECTION_MODE,
            "score_threshold": SCORE_THRESHOLD,
        },
        "selected_checkpoint": selected["checkpoint"],
        "selected_validation_f1": selected["metrics"]["official_pair_metrics"]
        [SELECTION_CATEGORY][SELECTION_MODE]["f1"],
        "ranking": [
            {
                "rank": rank,
                "checkpoint": result["checkpoint"],
                "regular_text_location": result["metrics"]["official_pair_metrics"]
                [SELECTION_CATEGORY][SELECTION_MODE],
                "entity_legacy_buggy": result["metrics"]["entity_metrics"]
                ["legacy_buggy"],
                "entity_corrected_micro": result["metrics"]["entity_metrics"]
                ["corrected_micro"],
                "entity_corrected_macro": result["metrics"]["entity_metrics"]
                ["corrected_macro"],
            }
            for rank, result in enumerate(ranked, start=1)
        ],
    }
    write_json(output_dir / "validation_summary.json", summary)

    test_dir = output_dir / "test" / selected_candidate.candidate_id
    test_result_path = test_dir / "result.json"
    test_result = reusable_result(test_result_path, selected_candidate, "test")
    if test_result is None:
        LOGGER.info(
            "Evaluating selected checkpoint on test exactly once: %s",
            selected_candidate.candidate_id,
        )
        load_checkpoint_weights(model, selected_candidate)
        test_metrics = evaluate_split(
            model,
            test_dataset,
            device,
            args.batch_size,
            test_dir / "predictions",
            "test",
        )
        test_result = {
            "checkpoint": selected_candidate.to_dict(checkpoint_root),
            "canonical_path": str(selected_candidate.canonical_path),
            "selected_by_validation": {
                "category": SELECTION_CATEGORY,
                "mode": SELECTION_MODE,
                "score_threshold": SCORE_THRESHOLD,
                "validation_f1": summary["selected_validation_f1"],
            },
            "completed_at": utc_now(),
            "metrics": test_metrics,
        }
        write_json(test_result_path, test_result)
    else:
        LOGGER.info("Reusing compatible selected-checkpoint test result")

    final = {
        "completed_at": utc_now(),
        "selected_checkpoint": selected_candidate.to_dict(checkpoint_root),
        "validation": selected["metrics"],
        "test": test_result["metrics"],
    }
    write_json(output_dir / "final_result.json", final)
    manifest["status"] = "completed"
    manifest["completed_at"] = final["completed_at"]
    manifest["selected_checkpoint"] = final["selected_checkpoint"]
    write_json(output_dir / "run_manifest.json", manifest)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("data/outputs/stage4b_v4"),
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/prepared"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/outputs/stage4b_v4_checkpoint_selection_official"),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    try:
        final = run(args)
    except Exception as error:
        error_payload = {
            "failed_at": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            write_json(args.output_dir.resolve() / "run_error.json", error_payload)
        finally:
            LOGGER.exception("Checkpoint selection failed")
        raise

    selected = final["selected_checkpoint"]
    validation_f1 = final["validation"]["official_pair_metrics"][
        SELECTION_CATEGORY
    ][SELECTION_MODE]["f1"]
    test_f1 = final["test"]["official_pair_metrics"][SELECTION_CATEGORY][
        SELECTION_MODE
    ]["f1"]
    print(
        f"Selected {selected['candidate_id']} "
        f"validation regular text+location F1={validation_f1:.10f} "
        f"test F1={test_f1:.10f}"
    )


if __name__ == "__main__":
    main()
