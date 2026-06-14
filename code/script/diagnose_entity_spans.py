#!/usr/bin/env python3
"""
Diagnose WHY the entity classifier produces so few spans at inference.

Entity F1=0.84 (token-level) should yield many key/value tokens, but
group_contiguous_spans only finds ~498 key spans across 581 docs.

This script checks:
  1. How many tokens are predicted as Key/Value/Other (raw counts)
  2. How many of those pass the bbox_valid filter
  3. How many contiguous spans form
  4. Comparison with GT entity labels
  5. Per-document breakdown
"""

import torch
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from layoutlm_model_v2 import create_model, group_contiguous_spans
from stage4_kvp_dataset import LayoutLMv3PreparedDataset, PaddedBatchCollator
from torch.utils.data import DataLoader


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_dir", default="data/prepared")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_model(use_linker=True, device=device)
    ckpt_dir = Path(args.checkpoint_dir)
    best_model = ckpt_dir / "best_model" / "pytorch_model.bin"
    if best_model.exists():
        state = torch.load(best_model, map_location=device, weights_only=False)
        print(f"Loaded: {best_model}")
    else:
        raise FileNotFoundError(f"No best_model found in {ckpt_dir}")

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Load data
    dataset = LayoutLMv3PreparedDataset(args.data_dir, "test")
    loader = DataLoader(dataset, batch_size=1, collate_fn=PaddedBatchCollator())

    # Counters
    total_docs = 0
    docs_with_pred_keys = 0
    docs_with_pred_vals = 0
    docs_with_gt_keys = 0
    docs_with_gt_vals = 0

    total_pred_key_tokens = 0
    total_pred_val_tokens = 0
    total_gt_key_tokens = 0
    total_gt_val_tokens = 0

    total_pred_key_spans = 0
    total_pred_val_spans = 0
    total_gt_key_spans = 0
    total_gt_val_spans = 0

    total_active_tokens = 0

    # Per-doc details for first 10
    details = []

    # bbox_valid filtering stats
    total_pred_key_tokens_pre_bbox = 0
    total_pred_val_tokens_pre_bbox = 0
    bbox_filtered_key = 0
    bbox_filtered_val = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Diagnosing")):
            if i >= args.max_samples:
                break
            total_docs += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            entity_labels = batch.get("entity_labels")
            if entity_labels is not None:
                entity_labels = entity_labels.to(device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)

            outputs = model(input_ids, attention_mask, bbox, pixel_values)
            entity_logits = outputs["entity_logits"]
            preds = torch.argmax(entity_logits, dim=-1)[0]  # [seq_len]
            mask = attention_mask[0]
            bb = bbox[0]

            active = mask == 1
            active_count = active.sum().item()
            total_active_tokens += active_count

            # Raw predicted token counts (before bbox filter)
            pred_key_mask = (preds == 1) & active
            pred_val_mask = (preds == 2) & active
            n_pred_key = pred_key_mask.sum().item()
            n_pred_val = pred_val_mask.sum().item()
            total_pred_key_tokens_pre_bbox += n_pred_key
            total_pred_val_tokens_pre_bbox += n_pred_val

            # bbox_valid filter
            bbox_valid = (bb[:, 0] < bb[:, 2]) & (bb[:, 1] < bb[:, 3])
            pred_key_after_bbox = (pred_key_mask & bbox_valid).sum().item()
            pred_val_after_bbox = (pred_val_mask & bbox_valid).sum().item()
            total_pred_key_tokens += pred_key_after_bbox
            total_pred_val_tokens += pred_val_after_bbox
            bbox_filtered_key += n_pred_key - pred_key_after_bbox
            bbox_filtered_val += n_pred_val - pred_val_after_bbox

            if pred_key_after_bbox > 0:
                docs_with_pred_keys += 1
            if pred_val_after_bbox > 0:
                docs_with_pred_vals += 1

            # Predicted spans
            key_spans = group_contiguous_spans(preds, mask, bb, label_id=1)
            val_spans = group_contiguous_spans(preds, mask, bb, label_id=2)
            total_pred_key_spans += len(key_spans)
            total_pred_val_spans += len(val_spans)

            # GT tokens and spans
            gt_key_tokens = 0
            gt_val_tokens = 0
            gt_key_sp = 0
            gt_val_sp = 0
            if entity_labels is not None:
                gt = entity_labels[0]
                gt_key_tokens = ((gt == 1) & active).sum().item()
                gt_val_tokens = ((gt == 2) & active).sum().item()
                total_gt_key_tokens += gt_key_tokens
                total_gt_val_tokens += gt_val_tokens
                if gt_key_tokens > 0:
                    docs_with_gt_keys += 1
                if gt_val_tokens > 0:
                    docs_with_gt_vals += 1
                gt_key_sp = len(group_contiguous_spans(gt, mask, bb, label_id=1))
                gt_val_sp = len(group_contiguous_spans(gt, mask, bb, label_id=2))
                total_gt_key_spans += gt_key_sp
                total_gt_val_spans += gt_val_sp

            if i < 20:
                details.append({
                    "doc": i,
                    "active_tokens": active_count,
                    "pred_key_tok": n_pred_key,
                    "pred_val_tok": n_pred_val,
                    "pred_key_tok_after_bbox": pred_key_after_bbox,
                    "pred_val_tok_after_bbox": pred_val_after_bbox,
                    "pred_key_spans": len(key_spans),
                    "pred_val_spans": len(val_spans),
                    "gt_key_tok": gt_key_tokens,
                    "gt_val_tok": gt_val_tokens,
                    "gt_key_spans": gt_key_sp,
                    "gt_val_spans": gt_val_sp,
                })

    print(f"\n{'='*70}")
    print(f"ENTITY SPAN DIAGNOSTIC ({total_docs} documents)")
    print(f"{'='*70}")

    print(f"\n--- Document-Level ---")
    print(f"Docs with GT key tokens:   {docs_with_gt_keys}/{total_docs} ({100*docs_with_gt_keys/total_docs:.0f}%)")
    print(f"Docs with GT val tokens:   {docs_with_gt_vals}/{total_docs} ({100*docs_with_gt_vals/total_docs:.0f}%)")
    print(f"Docs with PRED key tokens: {docs_with_pred_keys}/{total_docs} ({100*docs_with_pred_keys/total_docs:.0f}%)")
    print(f"Docs with PRED val tokens: {docs_with_pred_vals}/{total_docs} ({100*docs_with_pred_vals/total_docs:.0f}%)")

    print(f"\n--- Token Counts ---")
    print(f"Total active tokens:        {total_active_tokens}")
    print(f"GT key tokens:              {total_gt_key_tokens}")
    print(f"GT val tokens:              {total_gt_val_tokens}")
    print(f"PRED key tokens (raw):      {total_pred_key_tokens_pre_bbox}")
    print(f"PRED key tokens (bbox ok):  {total_pred_key_tokens}  (bbox filtered: {bbox_filtered_key})")
    print(f"PRED val tokens (raw):      {total_pred_val_tokens_pre_bbox}")
    print(f"PRED val tokens (bbox ok):  {total_pred_val_tokens}  (bbox filtered: {bbox_filtered_val})")

    print(f"\n--- Span Counts ---")
    print(f"GT key spans:    {total_gt_key_spans}  (avg {total_gt_key_spans/total_docs:.1f}/doc)")
    print(f"GT val spans:    {total_gt_val_spans}  (avg {total_gt_val_spans/total_docs:.1f}/doc)")
    print(f"PRED key spans:  {total_pred_key_spans}  (avg {total_pred_key_spans/total_docs:.1f}/doc)")
    print(f"PRED val spans:  {total_pred_val_spans}  (avg {total_pred_val_spans/total_docs:.1f}/doc)")

    print(f"\n--- Per-Document Details (first 20) ---")
    print(f"{'Doc':>4} {'Active':>6} | {'GT_K':>4} {'GT_V':>4} {'GT_Ksp':>6} {'GT_Vsp':>6} | "
          f"{'PR_K':>4} {'PR_V':>4} {'PR_K_bb':>7} {'PR_V_bb':>7} {'PR_Ksp':>6} {'PR_Vsp':>6}")
    print("-" * 95)
    for d in details:
        print(f"{d['doc']:4d} {d['active_tokens']:6d} | "
              f"{d['gt_key_tok']:4d} {d['gt_val_tok']:4d} {d['gt_key_spans']:6d} {d['gt_val_spans']:6d} | "
              f"{d['pred_key_tok']:4d} {d['pred_val_tok']:4d} {d['pred_key_tok_after_bbox']:7d} {d['pred_val_tok_after_bbox']:7d} "
              f"{d['pred_key_spans']:6d} {d['pred_val_spans']:6d}")


if __name__ == "__main__":
    main()
