"""
Stage 4b link-score threshold sweep.

Runs inference ONCE (collecting every key's best value + its sigmoid score at
score_threshold=0.0), then re-scores link F1 across a grid of decision
thresholds on CPU. Directly targets the low-recall regime (2601 predicted vs
3776 GT links at the fixed 0.5 cut).

To avoid test-set leakage, pick the best threshold on --split val, then
evaluate that single threshold on test with evaluate_stage4b.py.

Usage:
    python code/script/sweep_link_threshold.py \
        --checkpoint_dir data/outputs/stage4b_v4 \
        --data_dir data/prepared --model_version v2 \
        --split val --ned_thresh 0.2 --iou_thresh 0.3
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from layoutlm_model import create_model as create_model_v1
from layoutlm_model_v2 import create_model as create_model_v2
from stage4_kvp_dataset import LayoutLMv3PreparedDataset, PaddedBatchCollator

from evaluate_stage4b import (
    _load_checkpoint,
    _collect_link_pairs,
    _score_link_pairs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _filter(collected, thr):
    """Keep only predicted pairs whose link score >= thr (gt pairs untouched)."""
    return [([p for p in preds if p[2] >= thr], gts) for preds, gts in collected]


def main():
    ap = argparse.ArgumentParser(description="Sweep Stage 4b link score threshold")
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--data_dir", default="data/prepared")
    ap.add_argument("--split", default="val",
                    help="'val' (seed-42 10%% slice of train, held out from linker "
                         "training), 'test', or 'train'")
    ap.add_argument("--val_fraction", type=float, default=0.1,
                    help="Must match training (create_stage4_dataloaders default 0.1)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Must match training random_split seed (42)")
    ap.add_argument("--model_version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--ned_thresh", type=float, default=0.2)
    ap.add_argument("--iou_thresh", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grid", type=str, default="",
                    help="Comma-separated thresholds; default 0.05..0.90 step 0.05")
    args = ap.parse_args()

    grid = ([float(x) for x in args.grid.split(",")] if args.grid
            else [round(0.05 * i, 2) for i in range(1, 19)])  # 0.05 .. 0.90

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.model_version == "v2":
        model = create_model_v2(use_linker=True, device=device)
    else:
        model = create_model_v1(use_linker=True, device=device)
    model = _load_checkpoint(model, args.checkpoint_dir, device)
    model.eval()

    proc = model.encoder.processor if hasattr(model.encoder, "processor") else None
    if args.split == "val":
        # Reconstruct the EXACT held-out validation slice used during training:
        # a deterministic random 10% of train (random_split, seed 42). These
        # documents were never used for linker gradient updates, so tuning the
        # decision threshold on them does not leak into the test set.
        from torch.utils.data import random_split
        full_train = LayoutLMv3PreparedDataset(
            data_dir=args.data_dir, split="train", processor=proc,
            max_seq_length=512, include_images=False,
        )
        n_total = len(full_train)
        n_val = int(n_total * args.val_fraction)
        n_train = n_total - n_val
        _, val_split = random_split(
            full_train, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_indices = sorted(val_split.indices)
        full_train.json_files = [full_train.json_files[i] for i in val_indices]
        dataset = full_train
        logger.info(f"Reconstructed val slice: {len(dataset)} samples "
                    f"(seed={args.seed}, val_fraction={args.val_fraction})")
    else:
        dataset = LayoutLMv3PreparedDataset(
            data_dir=args.data_dir, split=args.split, processor=proc,
            max_seq_length=512, include_images=False,
        )
        logger.info(f"Split '{args.split}': {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=PaddedBatchCollator(), num_workers=0)

    # Collect ONCE at threshold 0.0 so every key's best value + score is kept.
    logger.info("Collecting link pairs once at score_threshold=0.0 ...")
    collected = _collect_link_pairs(model, dataset, loader, device, score_threshold=0.0)

    rows = []
    for thr in grid:
        fc = _filter(collected, thr)
        to = _score_link_pairs(fc, args.ned_thresh, args.iou_thresh, use_bbox=False)
        tb = _score_link_pairs(fc, args.ned_thresh, args.iou_thresh, use_bbox=True)
        rows.append({"thr": thr, "text_only": to, "text_bbox": tb})

    # Report
    print("\n" + "=" * 78)
    print(f"LINK THRESHOLD SWEEP  split={args.split}  NED<={args.ned_thresh}  IoU>={args.iou_thresh}")
    print("=" * 78)
    print(f"{'thr':>5} | {'text-only F1':>12} {'P':>6} {'R':>6} {'pred':>6} "
          f"| {'txt+box F1':>10} {'P':>6} {'R':>6}")
    print("-" * 78)
    for r in rows:
        to, tb = r["text_only"], r["text_bbox"]
        print(f"{r['thr']:>5.2f} | {to['f1']:>12.4f} {to['precision']:>6.3f} "
              f"{to['recall']:>6.3f} {to['total_pred']:>6d} "
              f"| {tb['f1']:>10.4f} {tb['precision']:>6.3f} {tb['recall']:>6.3f}")

    best_to = max(rows, key=lambda r: r["text_only"]["f1"])
    best_tb = max(rows, key=lambda r: r["text_bbox"]["f1"])
    print("-" * 78)
    print(f"BEST text-only : thr={best_to['thr']:.2f}  F1={best_to['text_only']['f1']:.4f}")
    print(f"BEST txt+box   : thr={best_tb['thr']:.2f}  F1={best_tb['text_bbox']['f1']:.4f}")
    print("=" * 78)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.checkpoint_dir) / f"link_threshold_sweep_{args.split}_{ts}.json"
    with open(out, "w") as f:
        json.dump({"split": args.split, "ned_thresh": args.ned_thresh,
                   "iou_thresh": args.iou_thresh, "grid": grid, "rows": rows}, f, indent=2)
    logger.info(f"Saved sweep to {out}")


if __name__ == "__main__":
    main()
