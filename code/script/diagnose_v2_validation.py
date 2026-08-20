#!/usr/bin/env python3
"""Run the V2 diagnostic checks on the fixed seed-42 validation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layoutlm_model_v2 as model_module
from layoutlm_model_v2 import collapse_link_labels_to_spans, create_model
from stage4_kvp_dataset import LayoutLMv3PreparedDataset, PaddedBatchCollator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_group_contiguous_spans(entity_preds, attention_mask, bbox, label_id):
    """Reproduce the historical strict bbox filter used in the V2 diagnosis."""
    spans = []
    in_span = False
    start = 0
    bbox_valid = (bbox[:, 0] < bbox[:, 2]) & (bbox[:, 1] < bbox[:, 3])
    for index in range(entity_preds.size(0)):
        is_target = (
            entity_preds[index] == label_id
            and attention_mask[index] == 1
            and bbox_valid[index]
        )
        if is_target and not in_span:
            start = index
            in_span = True
        elif not is_target and in_span:
            spans.append((start, index - 1))
            in_span = False
    if in_span:
        spans.append((start, entity_preds.size(0) - 1))
    return spans


def relaxed_group_contiguous_spans(entity_preds, attention_mask, bbox, label_id):
    """Use the corrected filter, which excludes only [0,0,0,0] boxes."""
    spans = []
    in_span = False
    start = 0
    bbox_valid = ~(
        (bbox[:, 0] == 0)
        & (bbox[:, 1] == 0)
        & (bbox[:, 2] == 0)
        & (bbox[:, 3] == 0)
    )
    for index in range(entity_preds.size(0)):
        is_target = (
            entity_preds[index] == label_id
            and attention_mask[index] == 1
            and bbox_valid[index]
        )
        if is_target and not in_span:
            start = index
            in_span = True
        elif not is_target and in_span:
            spans.append((start, index - 1))
            in_span = False
    if in_span:
        spans.append((start, entity_preds.size(0) - 1))
    return spans


def array_summary(values):
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "percentiles": {
            "1": float(np.percentile(array, 1)),
            "25": float(np.percentile(array, 25)),
            "50": float(np.percentile(array, 50)),
            "75": float(np.percentile(array, 75)),
            "99": float(np.percentile(array, 99)),
        },
    }


def build_validation_subset(data_dir: str, val_fraction: float, seed: int):
    base_dataset = LayoutLMv3PreparedDataset(
        data_dir=data_dir,
        split="train",
        include_images=False,
    )
    num_val = int(len(base_dataset) * val_fraction)
    num_train = len(base_dataset) - num_val
    _, validation_subset = random_split(
        base_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return base_dataset, validation_subset


def load_checkpoint(model, checkpoint_dir: Path, device):
    weights_path = checkpoint_dir / "best_model" / "pytorch_model.bin"
    if not weights_path.is_file():
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        weights_path = checkpoints[-1] / "pytorch_model.bin"
    state = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    return weights_path


def run_model_diagnostics(model, loader, device, max_samples):
    raw_logits = []
    best_logits = []
    counters = {
        "documents": 0,
        "documents_with_strict_key_and_value_spans": 0,
        "documents_without_strict_key_and_value_spans": 0,
        "predicted_key_tokens_raw": 0,
        "predicted_value_tokens_raw": 0,
        "predicted_key_tokens_strict": 0,
        "predicted_value_tokens_strict": 0,
        "predicted_key_tokens_relaxed": 0,
        "predicted_value_tokens_relaxed": 0,
        "predicted_key_spans_strict": 0,
        "predicted_value_spans_strict": 0,
        "predicted_key_spans_relaxed": 0,
        "predicted_value_spans_relaxed": 0,
    }

    with torch.no_grad():
        for sample_index, batch in enumerate(tqdm(loader, desc="Validation diagnostics")):
            if sample_index >= max_samples:
                break
            counters["documents"] += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            outputs = model(input_ids, attention_mask, bbox, pixel_values)
            predictions = outputs["entity_logits"].argmax(dim=-1)[0]
            mask = attention_mask[0]
            boxes = bbox[0]
            active = mask == 1

            raw_key = (predictions == 1) & active
            raw_value = (predictions == 2) & active
            strict_valid = (
                (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])
            )
            relaxed_valid = ~(
                (boxes[:, 0] == 0)
                & (boxes[:, 1] == 0)
                & (boxes[:, 2] == 0)
                & (boxes[:, 3] == 0)
            )

            counters["predicted_key_tokens_raw"] += int(raw_key.sum())
            counters["predicted_value_tokens_raw"] += int(raw_value.sum())
            counters["predicted_key_tokens_strict"] += int((raw_key & strict_valid).sum())
            counters["predicted_value_tokens_strict"] += int((raw_value & strict_valid).sum())
            counters["predicted_key_tokens_relaxed"] += int((raw_key & relaxed_valid).sum())
            counters["predicted_value_tokens_relaxed"] += int((raw_value & relaxed_valid).sum())

            strict_keys = strict_group_contiguous_spans(predictions, mask, boxes, 1)
            strict_values = strict_group_contiguous_spans(predictions, mask, boxes, 2)
            relaxed_keys = relaxed_group_contiguous_spans(predictions, mask, boxes, 1)
            relaxed_values = relaxed_group_contiguous_spans(predictions, mask, boxes, 2)
            counters["predicted_key_spans_strict"] += len(strict_keys)
            counters["predicted_value_spans_strict"] += len(strict_values)
            counters["predicted_key_spans_relaxed"] += len(relaxed_keys)
            counters["predicted_value_spans_relaxed"] += len(relaxed_values)

            scores = outputs["link_scores"]
            if scores is None or scores[0] is None:
                counters["documents_without_strict_key_and_value_spans"] += 1
                continue
            counters["documents_with_strict_key_and_value_spans"] += 1
            score_matrix = scores[0]
            raw_logits.extend(score_matrix.detach().cpu().flatten().tolist())
            best_logits.extend(
                score_matrix.max(dim=1).values.detach().cpu().tolist()
            )

    strict_key_removed = (
        counters["predicted_key_tokens_raw"]
        - counters["predicted_key_tokens_strict"]
    )
    strict_value_removed = (
        counters["predicted_value_tokens_raw"]
        - counters["predicted_value_tokens_strict"]
    )
    raw_key = max(counters["predicted_key_tokens_raw"], 1)
    raw_value = max(counters["predicted_value_tokens_raw"], 1)
    counters["strict_filter_key_removed"] = strict_key_removed
    counters["strict_filter_value_removed"] = strict_value_removed
    counters["strict_filter_key_removed_fraction"] = strict_key_removed / raw_key
    counters["strict_filter_value_removed_fraction"] = strict_value_removed / raw_value

    raw_sigmoid = torch.sigmoid(torch.tensor(raw_logits)) if raw_logits else torch.tensor([])
    best_sigmoid = torch.sigmoid(torch.tensor(best_logits)) if best_logits else torch.tensor([])
    return {
        "counts": counters,
        "all_pair_logits": array_summary(raw_logits),
        "best_logit_per_key_span": array_summary(best_logits),
        "threshold_counts": {
            "all_pairs": {
                str(threshold): int((raw_sigmoid > threshold).sum())
                for threshold in (0.1, 0.3, 0.5)
            },
            "best_per_key": {
                str(threshold): int((best_sigmoid > threshold).sum())
                for threshold in (0.1, 0.3, 0.5)
            },
        },
    }


def run_label_audit(base_dataset, validation_subset, audit_samples):
    totals = {
        "documents": 0,
        "ground_truth_regular_pairs": 0,
        "recovered_representative_word_links": 0,
        "positive_token_pair_cells": 0,
        "positive_span_pair_labels": 0,
    }
    records = []
    for subset_index in range(min(audit_samples, len(validation_subset))):
        base_index = validation_subset.indices[subset_index]
        json_path = base_dataset.json_files[base_index]
        with json_path.open() as handle:
            data = json.load(handle)
        words, word_boxes = base_dataset._parse_lmdx_text(
            data.get("lmdx_text", ""), data
        )
        kvps = data.get("gt_kvps", {}).get("kvps_list", [])
        _, word_links = base_dataset._generate_labels(
            words,
            kvps,
            word_boxes,
            data.get("image_width", 1),
            data.get("image_height", 1),
        )
        ground_truth_pairs = sum(1 for kvp in kvps if kvp.get("type") == "kvp")
        word_link_count = int(word_links.sum())

        item = base_dataset[base_index]
        token_links = item["link_labels"]
        token_link_count = int(token_links.sum())
        key_spans = relaxed_group_contiguous_spans(
            item["entity_labels"], item["attention_mask"], item["bbox"], 1
        )
        value_spans = relaxed_group_contiguous_spans(
            item["entity_labels"], item["attention_mask"], item["bbox"], 2
        )
        if key_spans and value_spans:
            span_labels = collapse_link_labels_to_spans(
                token_links, key_spans, value_spans
            )
            span_link_count = int(span_labels.sum())
        else:
            span_link_count = 0

        record = {
            "validation_subset_index": subset_index,
            "base_train_index": int(base_index),
            "hash_name": data.get("hash_name"),
            "ground_truth_regular_pairs": ground_truth_pairs,
            "recovered_representative_word_links": word_link_count,
            "positive_token_pair_cells": token_link_count,
            "positive_span_pair_labels": span_link_count,
        }
        records.append(record)
        totals["documents"] += 1
        for key in (
            "ground_truth_regular_pairs",
            "recovered_representative_word_links",
            "positive_token_pair_cells",
            "positive_span_pair_labels",
        ):
            totals[key] += record[key]
    return {"totals": totals, "documents": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("data/outputs/stage4b_v2_tf_p7"),
    )
    parser.add_argument("--data_dir", default="data/prepared")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--max_model_samples", type=int, default=200)
    parser.add_argument("--audit_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    if args.output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_json}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Validation model diagnostics require a CUDA allocation")

    base_dataset, validation_subset = build_validation_subset(
        args.data_dir, args.val_fraction, args.seed
    )
    loader = DataLoader(
        validation_subset,
        batch_size=1,
        shuffle=False,
        collate_fn=PaddedBatchCollator(),
    )

    corrected_group = model_module.group_contiguous_spans
    model_module.group_contiguous_spans = strict_group_contiguous_spans
    try:
        model = create_model(use_linker=True, device=device)
        weights_path = load_checkpoint(model, args.checkpoint_dir, device)
        model.to(device)
        model.eval()
        model_diagnostics = run_model_diagnostics(
            model, loader, device, args.max_model_samples
        )
    finally:
        model_module.group_contiguous_spans = corrected_group

    label_audit = run_label_audit(
        base_dataset, validation_subset, args.audit_samples
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "split_source": "data/prepared/train",
        "split_seed": args.seed,
        "validation_fraction": args.val_fraction,
        "train_pages_total": len(base_dataset),
        "validation_pages_total": len(validation_subset),
        "model_samples": min(args.max_model_samples, len(validation_subset)),
        "label_audit_samples": min(args.audit_samples, len(validation_subset)),
        "real_images_loaded": False,
        "checkpoint_dir": str(args.checkpoint_dir),
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "historical_bbox_filter_reproduced": "x1 < x2 and y1 < y2",
        "model_diagnostics": model_diagnostics,
        "label_pipeline_audit": label_audit,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=False)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
