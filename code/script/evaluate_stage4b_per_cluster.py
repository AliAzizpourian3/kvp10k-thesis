"""
Stage 4 V4 per-cluster evaluation.

Loads the test cluster map, splits the 581 test JSONs by cluster, runs V4
inference on each subset, and reports entity F1 + link F1 (text-only and
text+location) per cluster.

Usage:
    python code/script/evaluate_stage4b_per_cluster.py \
        --checkpoint_dir data/outputs/stage4b_v4 \
        --data_dir data/prepared \
        --cluster_map data/outputs/stage2/test_cluster_map.json \
        --model_version v2 \
        --ned_thresh 0.2 --iou_thresh 0.3
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
from torch.utils.data import DataLoader, Subset

from layoutlm_model import create_model as create_model_v1
from layoutlm_model_v2 import create_model as create_model_v2
from stage4_kvp_dataset import LayoutLMv3PreparedDataset, PaddedBatchCollator

from evaluate_stage4b import (
    _load_checkpoint,
    _collect_link_pairs,
    _score_link_pairs,
    compute_entity_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical cluster names for reporting (map from raw map values)
CLUSTER_LABELS = {
    "Cluster_0_Dense": "Cluster 0 (text-rich)",
    "Cluster_1_Sparse": "Cluster 1 (text-sparse)",
    0: "Cluster 0 (text-rich)",
    1: "Cluster 1 (text-sparse)",
}


def _load_cluster_map(path):
    """Return {hash_name: cluster_label} mapping."""
    raw = json.load(open(path))
    # Values may be strings like 'Cluster_0_Dense' or ints 0/1.
    return {k: v for k, v in raw.items()}


def _normalize_cluster_label(raw_label):
    """Map raw cluster values to stable display labels."""
    if isinstance(raw_label, dict):
        raw_label = raw_label.get("cluster", raw_label.get("label", raw_label))
    return CLUSTER_LABELS.get(raw_label, str(raw_label))


def _build_cluster_indices(dataset, cluster_map):
    """Return {label: [dataset_indices]} mapping."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, jf in enumerate(dataset.json_files):
        try:
            h = json.load(open(jf)).get("hash_name", "")
        except Exception:
            continue
        raw_label = cluster_map.get(h)
        if raw_label is None:
            continue
        label = _normalize_cluster_label(raw_label)
        buckets[label].append(i)
    return dict(buckets)


def _eval_subset(model, dataset, indices, device, score_threshold,
                 ned_thresh, iou_thresh):
    """Run entity + link eval on a subset of the dataset by index list."""
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False,
                        collate_fn=PaddedBatchCollator(), num_workers=0)

    # Entity F1
    ent = compute_entity_metrics(model, loader, device)

    # Link F1 — need to pass original dataset + offset for JSON lookup
    # Rebuild a minimal dataset view with only the selected json_files so
    # that _collect_link_pairs can look up GT JSONs by dataset.json_files[idx].
    sub_dataset = object.__new__(LayoutLMv3PreparedDataset)
    sub_dataset.__dict__.update(dataset.__dict__)
    sub_dataset.json_files = [dataset.json_files[i] for i in indices]
    sub_loader = DataLoader(sub_dataset, batch_size=1, shuffle=False,
                            collate_fn=PaddedBatchCollator(), num_workers=0)

    collected = _collect_link_pairs(model, sub_dataset, sub_loader, device,
                                    score_threshold=score_threshold)
    to = _score_link_pairs(collected, ned_thresh, iou_thresh, use_bbox=False)
    tb = _score_link_pairs(collected, ned_thresh, iou_thresh, use_bbox=True)

    return ent, to, tb


def main():
    ap = argparse.ArgumentParser(description="V4 per-cluster evaluation")
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--data_dir", default="data/prepared")
    ap.add_argument("--cluster_map",
                    default="data/outputs/stage2/test_cluster_map.json")
    ap.add_argument("--model_version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--score_threshold", type=float, default=0.5)
    ap.add_argument("--ned_thresh", type=float, default=0.2)
    ap.add_argument("--iou_thresh", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.model_version == "v2":
        model = create_model_v2(use_linker=True, device=device)
    else:
        model = create_model_v1(use_linker=True, device=device)
    model = _load_checkpoint(model, args.checkpoint_dir, device)
    model.eval()

    proc = model.encoder.processor if hasattr(model.encoder, "processor") else None
    dataset = LayoutLMv3PreparedDataset(
        data_dir=args.data_dir, split="test", processor=proc,
        max_seq_length=512, include_images=False,
    )
    logger.info(f"Test set: {len(dataset)} samples")

    cluster_map = _load_cluster_map(args.cluster_map)
    buckets = _build_cluster_indices(dataset, cluster_map)
    logger.info(f"Cluster assignment: " +
                ", ".join(f"{k}={len(v)}" for k, v in sorted(buckets.items())))

    results = {}
    for label, indices in sorted(buckets.items()):
        logger.info(f"\n--- {label}: {len(indices)} docs ---")
        ent, to, tb = _eval_subset(
            model, dataset, indices, device,
            args.score_threshold, args.ned_thresh, args.iou_thresh
        )
        results[label] = {"n_docs": len(indices), "entity": ent,
                          "link_text_only": to, "link_text_bbox": tb}
        logger.info(f"  Entity F1: {ent['entity_f1']:.4f} "
                    f"(P={ent['entity_precision']:.4f} R={ent['entity_recall']:.4f})")
        logger.info(f"  Link text-only  F1: {to['f1']:.4f} "
                    f"(P={to['precision']:.4f} R={to['recall']:.4f}) "
                    f"TP={to['tp']}/{to['total_gt']}")
        logger.info(f"  Link text+loc   F1: {tb['f1']:.4f} "
                    f"(P={tb['precision']:.4f} R={tb['recall']:.4f}) "
                    f"TP={tb['tp']}/{tb['total_gt']}")

    # -------------------------------------------------------------------------
    # Print summary table
    print("\n" + "=" * 80)
    print(f"V4 PER-CLUSTER RESULTS  NED<={args.ned_thresh}  IoU>={args.iou_thresh}")
    print("=" * 80)
    print(f"{'Cluster':<30} {'Docs':>5} {'EntityF1':>9} {'LinkF1(to)':>11} "
          f"{'LinkF1(tb)':>11} {'TP/GT(to)':>12}")
    print("-" * 80)
    for label, r in sorted(results.items()):
        e = r["entity"]
        to = r["link_text_only"]
        tb = r["link_text_bbox"]
        print(f"{label:<30} {r['n_docs']:>5} {e['entity_f1']:>9.4f} "
              f"{to['f1']:>11.4f} {tb['f1']:>11.4f} "
              f"{to['tp']:>5}/{to['total_gt']:<5}")
    print("=" * 80)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.checkpoint_dir) / f"eval_per_cluster_{ts}.json"
    with open(out, "w") as f:
        json.dump({"ned_thresh": args.ned_thresh, "iou_thresh": args.iou_thresh,
                   "score_threshold": args.score_threshold, "clusters": results}, f, indent=2)
    logger.info(f"Saved to {out}")


if __name__ == "__main__":
    main()
