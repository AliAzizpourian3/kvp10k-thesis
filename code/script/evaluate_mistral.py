"""
Evaluate Mistral-7B predictions against prepared ground truth.

Compares prediction JSONs (from mistral_baseline.py predict) against
ground truth KVPs (from prepare_data.py).

Metrics:
  - Precision, Recall, F1 at entity level (key and value separately)
  - Text-only matching (NED ≤ threshold)
  - Text+bbox matching (NED ≤ threshold AND IoU ≥ threshold)
  - Per-type breakdown (kvp, unkeyed, unvalued)

Usage:
  python evaluate_mistral.py \
      --pred_dir data/outputs/stage3_mistral/predictions \
      --gt_dir   data/prepared/test \
      --output   data/outputs/stage3_mistral/evaluation.json
"""

import os
import json
import glob
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _ned(s1: str, s2: str) -> float:
    """Normalised edit distance (Levenshtein)."""
    m, n = len(s1), len(s2)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            temp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[n] / max(m, n)


def _iou(b1: List[int], b2: List[int]) -> float:
    """IoU between two bboxes [left, top, right, bottom]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Entity extraction  (from a single KVP dict → list of (text, bbox_or_None) )
# ---------------------------------------------------------------------------

def _extract_entities(kvps: List[Dict]) -> List[Tuple[str, Optional[List[int]], str]]:
    """
    Flatten KVPs into (text, bbox, role) tuples.
    role ∈ {key, value, unvalued_key, unkeyed_value}
    """
    entities = []
    for kvp in kvps:
        t = kvp.get("type", "kvp")
        key = kvp.get("key", {})
        val = kvp.get("value", {})

        if t == "unvalued":
            entities.append((key.get("text", ""), key.get("bbox"), "unvalued_key"))
        elif t == "unkeyed":
            entities.append((val.get("text", ""), val.get("bbox"), "unkeyed_value"))
        else:  # regular kvp
            entities.append((key.get("text", ""), key.get("bbox"), "key"))
            entities.append((val.get("text", ""), val.get("bbox"), "value"))
    return entities


# ---------------------------------------------------------------------------
# Matching at document level
# ---------------------------------------------------------------------------

def match_entities(
    preds: List[Tuple[str, Optional[List[int]], str]],
    gts: List[Tuple[str, Optional[List[int]], str]],
    ned_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    use_bbox: bool = True,
) -> Dict:
    """
    Greedily match predicted entities to GT entities.
    Returns {tp, fp, fn}.
    """
    matched_gt = set()
    tp = 0

    for p_text, p_bbox, p_role in preds:
        best_score = -1.0
        best_idx = -1

        for j, (g_text, g_bbox, g_role) in enumerate(gts):
            if j in matched_gt:
                continue

            ned = _ned(p_text.lower(), g_text.lower())
            if ned > ned_thresh:
                continue

            if use_bbox and p_bbox and g_bbox:
                iou = _iou(p_bbox, g_bbox)
                if iou < iou_thresh:
                    continue
                score = (1 - ned) + iou
            else:
                score = 1 - ned

            if score > best_score:
                best_score = score
                best_idx = j

        if best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1

    return {"tp": tp, "fp": len(preds) - tp, "fn": len(gts) - len(matched_gt)}


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

def evaluate(pred_dir: str, gt_dir: str, ned_thresh: float = 0.5, iou_thresh: float = 0.5):
    """Evaluate all prediction files against GT."""
    pred_files = {Path(f).stem: f for f in sorted(glob.glob(os.path.join(pred_dir, "*.json")))}
    gt_files = {Path(f).stem: f for f in sorted(glob.glob(os.path.join(gt_dir, "*.json")))}

    common = sorted(set(pred_files) & set(gt_files))
    logger.info(f"Predictions: {len(pred_files)}, GT: {len(gt_files)}, Common: {len(common)}")

    if not common:
        logger.error("No overlapping hash_names between predictions and GT.")
        return {}

    # Accumulators for different matching modes
    modes = {
        "text_only": {"use_bbox": False},
        "text_bbox": {"use_bbox": True},
    }
    accum = {m: {"tp": 0, "fp": 0, "fn": 0} for m in modes}

    per_doc = []

    for hash_name in common:
        with open(pred_files[hash_name]) as f:
            pred_data = json.load(f)
        with open(gt_files[hash_name]) as f:
            gt_data = json.load(f)

        pred_kvps = pred_data.get("kvps_list", [])
        gt_kvps = gt_data.get("gt_kvps", {}).get("kvps_list", gt_data.get("kvps_list", []))

        pred_ents = _extract_entities(pred_kvps)
        gt_ents = _extract_entities(gt_kvps)

        doc_row = {"hash_name": hash_name, "n_pred": len(pred_kvps), "n_gt": len(gt_kvps)}

        for mode_name, mode_kw in modes.items():
            res = match_entities(pred_ents, gt_ents, ned_thresh, iou_thresh, **mode_kw)
            for k in ("tp", "fp", "fn"):
                accum[mode_name][k] += res[k]
            doc_row[f"{mode_name}_tp"] = res["tp"]
            doc_row[f"{mode_name}_fp"] = res["fp"]
            doc_row[f"{mode_name}_fn"] = res["fn"]

        per_doc.append(doc_row)

    # Aggregate
    results = {"n_docs": len(common), "ned_threshold": ned_thresh, "iou_threshold": iou_thresh}
    for mode_name, a in accum.items():
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results[mode_name] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                              "tp": tp, "fp": fp, "fn": fn}

    return results, per_doc


def main():
    p = argparse.ArgumentParser(description="Evaluate Mistral KVP predictions")
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--output", default=None, help="Save results JSON")
    p.add_argument("--ned_thresh", type=float, default=0.5)
    p.add_argument("--iou_thresh", type=float, default=0.5)
    args = p.parse_args()

    results, per_doc = evaluate(args.pred_dir, args.gt_dir, args.ned_thresh, args.iou_thresh)

    print("\n" + "=" * 60)
    print("MISTRAL-7B EVALUATION RESULTS")
    print("=" * 60)
    print(f"Documents evaluated: {results['n_docs']}")
    print(f"NED threshold: {results['ned_threshold']}")
    print(f"IoU threshold: {results['iou_threshold']}")
    for mode in ("text_only", "text_bbox"):
        m = results[mode]
        print(f"\n  {mode}:")
        print(f"    Precision: {m['precision']:.4f}")
        print(f"    Recall:    {m['recall']:.4f}")
        print(f"    F1:        {m['f1']:.4f}")
        print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"summary": results, "per_document": per_doc}, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
