"""
Stage 4b Evaluation: Entity F1 + Link F1 on test set.

Loads a trained checkpoint, runs inference on the test split, and reports:
  1. Entity-level F1 (token-level key/value classification)
  2. Link-level F1 (predicted key-value pairs vs ground-truth pairs)

Link matching uses normalized text overlap (case-insensitive substring containment)
because the model operates on subword tokens whose boundaries rarely align perfectly
with the ground-truth multi-word key/value spans.

Usage:
    python evaluate_stage4b.py --checkpoint_dir data/outputs/stage4b_lambda10
    python evaluate_stage4b.py --checkpoint_dir data/outputs/stage4b_canary_B
    python evaluate_stage4b.py --checkpoint_dir data/outputs/stage4b_v2 --model_version v2
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from layoutlm_model import create_model as create_model_v1
from layoutlm_model_v2 import create_model as create_model_v2
from stage4_kvp_dataset import LayoutLMv3PreparedDataset, PaddedBatchCollator

# Reuse the EXACT text/bbox matching functions from the official KVP10k
# evaluation so link-pair matching is consistent with the headline metric.
from evaluate_mistral import _ned, _iou

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(model, checkpoint_dir, device):
    """Load best checkpoint from a training output directory."""
    ckpt_dir = Path(checkpoint_dir)

    # Look for best_model first, then latest checkpoint
    best_model_dir = ckpt_dir / "best_model"
    best_model_file = ckpt_dir / "best_model.pt"
    if best_model_dir.is_dir():
        for name in ["pytorch_model.bin", "model.pt"]:
            p = best_model_dir / name
            if p.exists():
                state = torch.load(p, map_location=device, weights_only=False)
                logger.info(f"Loaded {name} from {best_model_dir}")
                break
        else:
            raise FileNotFoundError(f"No model file in {best_model_dir}")
    elif best_model_file.exists():
        state = torch.load(best_model_file, map_location=device, weights_only=False)
        logger.info(f"Loaded best_model.pt from {ckpt_dir}")
    else:
        # Find latest numbered checkpoint
        ckpt_dirs = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if not ckpt_dirs:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        latest = ckpt_dirs[-1]
        # Try model.pt then pytorch_model.bin
        for name in ["model.pt", "pytorch_model.bin"]:
            p = latest / name
            if p.exists():
                state = torch.load(p, map_location=device, weights_only=False)
                logger.info(f"Loaded {name} from {latest}")
                break
        else:
            raise FileNotFoundError(f"No model file in {latest}")

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    # Handle key prefix mismatches
    cleaned = {}
    for k, v in state.items():
        k_clean = k.replace("module.", "")
        cleaned[k_clean] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    return model


def _get_gt_kvps(json_path):
    """Load ground-truth KVP pairs from a prepared JSON file.

    Returns list of (key_text, value_text) for type=="kvp" entries only.
    """
    with open(json_path) as f:
        data = json.load(f)
    kvps_list = data.get("gt_kvps", {}).get("kvps_list", [])
    pairs = []
    for kvp in kvps_list:
        if kvp.get("type") != "kvp":
            continue
        key = kvp.get("key", {})
        val = kvp.get("value", {})
        key_text = key.get("text", "").strip().lower()
        val_text = val.get("text", "").strip().lower()
        if key_text and val_text:
            pairs.append((key_text, val_text, key.get("bbox"), val.get("bbox")))
    return pairs, data.get("hash_name", "")


def _text_overlap(pred_text, gt_text, ned_thresh=0.5):
    """Decide whether a predicted text matches a ground-truth text.

    Uses the SAME normalised edit distance (NED) criterion as the official
    KVP10k evaluation (``evaluate_mistral._ned``): a match requires
    ``NED(pred, gt) <= ned_thresh``. This replaces the previous ad-hoc rule
    (bidirectional substring containment OR word-level F1 >= 0.5), which was
    more lenient than the headline metric and therefore inflated link F1.
    """
    pred_text = pred_text.strip().lower()
    gt_text = gt_text.strip().lower()

    if not pred_text or not gt_text:
        return False

    return _ned(pred_text, gt_text) <= ned_thresh


def _span_to_text(start, end, word_ids, words):
    """Convert a token span (start, end inclusive) to text via word_ids mapping."""
    seen = []
    for t in range(start, end + 1):
        if t < len(word_ids) and word_ids[t] is not None:
            wid = word_ids[t]
            if wid not in seen:
                seen.append(wid)
    parts = [words[wid] for wid in seen if wid < len(words)]
    return " ".join(parts)


def _span_to_bbox(start, end, word_ids, word_bboxes):
    """Union bounding box of the words covered by a token span (start..end).

    Uses the raw lmdx word boxes, which live in the SAME coordinate space as
    the ground-truth key/value boxes in the prepared JSON, so IoU is directly
    comparable. Returns None if no word maps into the span.
    """
    seen = []
    for t in range(start, end + 1):
        if t < len(word_ids) and word_ids[t] is not None:
            wid = word_ids[t]
            if wid not in seen:
                seen.append(wid)
    boxes = [word_bboxes[wid] for wid in seen if wid < len(word_bboxes)]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def _extract_predicted_pairs(model, batch, device, dataset, idx_start, score_threshold=0.5):
    """Run model inference and extract predicted key-value text pairs.

    Works with both V1 (token-level indices) and V2 (span tuples).
    Returns a list of lists (one per batch item) of
    (key_text, val_text, score, key_bbox, val_bbox), where the bboxes are in
    the same coordinate space as the ground-truth boxes (or None if unknown).
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    bbox = batch["bbox"].to(device)
    pixel_values = batch["pixel_values"].to(device) if "pixel_values" in batch else None

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, bbox, pixel_values)

    entity_logits = outputs["entity_logits"]
    link_scores = outputs["link_scores"]
    key_indices = outputs["key_indices"]
    value_indices = outputs["value_indices"]

    # Detect V2 (span tuples) vs V1 (token index tensors)
    is_v2 = (key_indices is not None
             and len(key_indices) > 0
             and key_indices[0] is not None
             and len(key_indices[0]) > 0
             and isinstance(key_indices[0][0], tuple))

    batch_size = input_ids.size(0)
    all_pairs = []

    for b in range(batch_size):
        pairs = []
        sample_idx = idx_start + b
        if sample_idx >= len(dataset):
            all_pairs.append(pairs)
            continue

        # Get words for this sample from the dataset JSON
        json_file = dataset.json_files[sample_idx]
        try:
            with open(json_file) as f:
                data = json.load(f)
            lmdx_text = data.get("lmdx_text", "")
            words, _ = dataset._parse_lmdx_text(lmdx_text, data)
        except Exception:
            all_pairs.append(pairs)
            continue

        # Map token indices back to word indices via the processor
        try:
            image_width = data.get("image_width", 1)
            image_height = data.get("image_height", 1)
            _, word_bboxes = dataset._parse_lmdx_text(lmdx_text, data)
            bboxes_norm = dataset._normalize_bboxes(word_bboxes, image_width, image_height)
            from PIL import Image
            dummy_img = Image.new("RGB", (224, 224), (255, 255, 255))
            encoded = dataset.processor(
                images=dummy_img, text=words, boxes=bboxes_norm,
                return_tensors="pt", padding="max_length",
                max_length=dataset.max_seq_length, truncation=True
            )
            word_ids = encoded.word_ids()
        except Exception:
            all_pairs.append(pairs)
            continue

        if link_scores is None or link_scores[b] is None:
            all_pairs.append(pairs)
            continue

        scores_b = link_scores[b]  # [nk, nv]
        k_idx = key_indices[b]
        v_idx = value_indices[b]

        if len(k_idx) == 0 or len(v_idx) == 0:
            all_pairs.append(pairs)
            continue

        # For each key, find its best value
        best_val_pos = torch.argmax(scores_b, dim=1)
        best_scores = torch.sigmoid(scores_b[range(len(k_idx)), best_val_pos])

        for i, ki in enumerate(k_idx):
            if best_scores[i].item() < score_threshold:
                continue
            vi = v_idx[best_val_pos[i]]

            if is_v2:
                # V2: ki and vi are (start, end) tuples
                key_text = _span_to_text(ki[0], ki[1], word_ids, words)
                val_text = _span_to_text(vi[0], vi[1], word_ids, words)
                key_bbox = _span_to_bbox(ki[0], ki[1], word_ids, word_bboxes)
                val_bbox = _span_to_bbox(vi[0], vi[1], word_ids, word_bboxes)
            else:
                # V1: ki and vi are single token index tensors
                ki_int = ki.item()
                vi_int = vi.item()
                key_bbox = val_bbox = None
                if ki_int < len(word_ids) and word_ids[ki_int] is not None:
                    key_word_idx = word_ids[ki_int]
                    key_text = words[key_word_idx] if key_word_idx < len(words) else ""
                    if key_word_idx < len(word_bboxes):
                        key_bbox = word_bboxes[key_word_idx]
                else:
                    key_text = ""
                if vi_int < len(word_ids) and word_ids[vi_int] is not None:
                    val_word_idx = word_ids[vi_int]
                    val_text = words[val_word_idx] if val_word_idx < len(words) else ""
                    if val_word_idx < len(word_bboxes):
                        val_bbox = word_bboxes[val_word_idx]
                else:
                    val_text = ""

            if key_text and val_text:
                pairs.append((key_text.lower(), val_text.lower(),
                              best_scores[i].item(), key_bbox, val_bbox))

        all_pairs.append(pairs)

    return all_pairs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_entity_metrics(model, dataloader, device):
    """Compute token-level entity classification P/R/F1 on key+value tokens."""
    tp = fp = fn = 0
    model.eval()

    for batch in tqdm(dataloader, desc="Entity eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bbox = batch["bbox"].to(device)
        entity_labels = batch["entity_labels"].to(device)
        pixel_values = batch["pixel_values"].to(device) if "pixel_values" in batch else None

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, bbox, pixel_values)

        preds = torch.argmax(outputs["entity_logits"], dim=-1)
        mask = attention_mask == 1

        key_val_gt = ((entity_labels == 1) | (entity_labels == 2)) & mask
        key_val_pred = ((preds == 1) | (preds == 2)) & mask

        tp += ((preds == entity_labels) & key_val_gt).sum().item()
        fp += (key_val_pred & ~key_val_gt).sum().item()
        fn += (~key_val_pred & key_val_gt).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"entity_precision": precision, "entity_recall": recall, "entity_f1": f1}


def _bbox_ok(pred_bbox, gt_bbox, iou_thresh):
    """IoU gate for a predicted/GT box pair.

    Mirrors ``evaluate_mistral.match_entities``: when either box is missing we
    fall back to text-only matching (do not reject on geometry we don't have).
    """
    if pred_bbox is None or gt_bbox is None:
        return True
    return _iou(pred_bbox, gt_bbox) >= iou_thresh


def _collect_link_pairs(model, dataset, dataloader, device, score_threshold=0.5):
    """Run model inference ONCE and collect (pred_pairs, gt_pairs) per document.

    Separating inference from scoring lets us evaluate several matching modes
    (text-only, text+bbox) and thresholds without re-running the GPU model.
    """
    model.eval()
    collected = []
    idx_start = 0
    for batch in tqdm(dataloader, desc="Link eval (inference)"):
        batch_size = batch["input_ids"].size(0)
        pred_pairs_batch = _extract_predicted_pairs(
            model, batch, device, dataset, idx_start, score_threshold
        )
        for b in range(batch_size):
            sample_idx = idx_start + b
            if sample_idx >= len(dataset):
                continue
            json_file = dataset.json_files[sample_idx]
            gt_pairs, _ = _get_gt_kvps(json_file)
            collected.append((pred_pairs_batch[b], gt_pairs))
        idx_start += batch_size
    return collected


def _score_link_pairs(collected, ned_thresh=0.5, iou_thresh=0.5, use_bbox=True):
    """Score collected pairs under one matching mode.

    A predicted pair matches a GT pair if NED(key)<=ned_thresh and
    NED(value)<=ned_thresh (text mode), and additionally IoU(key)>=iou_thresh
    and IoU(value)>=iou_thresh when ``use_bbox`` is True. Greedy, best-score
    first; each GT pair matched at most once.
    """
    tp = 0
    total_pred = 0
    total_gt = 0
    for pred_pairs, gt_pairs in collected:
        total_gt += len(gt_pairs)
        total_pred += len(pred_pairs)

        pred_sorted = sorted(pred_pairs, key=lambda x: x[2], reverse=True)
        gt_matched = [False] * len(gt_pairs)

        for pk, pv, _score, pkb, pvb in pred_sorted:
            for j, (gk, gv, gkb, gvb) in enumerate(gt_pairs):
                if gt_matched[j]:
                    continue
                if not (_text_overlap(pk, gk, ned_thresh) and _text_overlap(pv, gv, ned_thresh)):
                    continue
                if use_bbox and not (_bbox_ok(pkb, gkb, iou_thresh) and _bbox_ok(pvb, gvb, iou_thresh)):
                    continue
                tp += 1
                gt_matched[j] = True
                break

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "total_pred": total_pred,
        "total_gt": total_gt,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 4b checkpoint")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to training output dir (e.g. data/outputs/stage4b_lambda10)")
    parser.add_argument("--data_dir", default="data/prepared",
                        help="Path to data/prepared/")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Link score threshold for pair extraction")
    parser.add_argument("--ned_thresh", type=float, default=0.5,
                        help="NED threshold for key/value text matching (official metric: <=0.5)")
    parser.add_argument("--iou_thresh", type=float, default=0.5,
                        help="IoU threshold for key/value box matching in text+bbox mode")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Evaluation batch size")
    parser.add_argument("--model_version", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="Model version: v1 (token-level) or v2 (span-level)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Create model
    if args.model_version == "v2":
        logger.info("Using V2 model (span-level linker)")
        model = create_model_v2(use_linker=True, device=device)
    else:
        logger.info("Using V1 model (token-level linker)")
        model = create_model_v1(use_linker=True, device=device)

    # Load checkpoint
    model = _load_checkpoint(model, args.checkpoint_dir, device)
    model.eval()

    # Create test dataset
    dataset = LayoutLMv3PreparedDataset(
        data_dir=args.data_dir,
        split="test",
        processor=model.encoder.processor if hasattr(model.encoder, 'processor') else None,
        max_seq_length=512,
        include_images=False,
    )
    logger.info(f"Test set: {len(dataset)} samples")

    collator = PaddedBatchCollator()
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=0
    )

    # Entity metrics
    logger.info("Computing entity metrics...")
    entity_metrics = compute_entity_metrics(model, dataloader, device)
    logger.info(f"Entity P={entity_metrics['entity_precision']:.4f}  "
                f"R={entity_metrics['entity_recall']:.4f}  "
                f"F1={entity_metrics['entity_f1']:.4f}")

    # Link metrics: run inference once, then score both matching modes.
    logger.info("Computing link metrics (single inference pass)...")
    collected = _collect_link_pairs(model, dataset, dataloader, device, args.score_threshold)

    text_only = _score_link_pairs(collected, args.ned_thresh, args.iou_thresh, use_bbox=False)
    text_bbox = _score_link_pairs(collected, args.ned_thresh, args.iou_thresh, use_bbox=True)

    # Backwards-compatible top-level keys mirror the text+bbox (strictest) mode.
    link_metrics = {
        "link_precision": text_bbox["precision"],
        "link_recall": text_bbox["recall"],
        "link_f1": text_bbox["f1"],
        "link_tp": text_bbox["tp"],
        "link_total_pred": text_bbox["total_pred"],
        "link_total_gt": text_bbox["total_gt"],
        "link_text_only": text_only,
        "link_text_bbox": text_bbox,
    }
    logger.info(f"Link text-only  P={text_only['precision']:.4f}  "
                f"R={text_only['recall']:.4f}  F1={text_only['f1']:.4f}  "
                f"TP={text_only['tp']}/{text_only['total_gt']}")
    logger.info(f"Link text+bbox  P={text_bbox['precision']:.4f}  "
                f"R={text_bbox['recall']:.4f}  F1={text_bbox['f1']:.4f}  "
                f"TP={text_bbox['tp']}/{text_bbox['total_gt']}")

    # Save results
    results = {**entity_metrics, **link_metrics, "checkpoint_dir": args.checkpoint_dir,
               "score_threshold": args.score_threshold, "ned_thresh": args.ned_thresh,
               "iou_thresh": args.iou_thresh}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.checkpoint_dir) / f"eval_results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 50)
    print(f"EVALUATION: {args.checkpoint_dir}")
    print(f"  NED<={args.ned_thresh}  IoU>={args.iou_thresh}  score_thr={args.score_threshold}")
    print("=" * 50)
    print(f"  Entity F1:        {entity_metrics['entity_f1']:.4f}  "
          f"(P={entity_metrics['entity_precision']:.4f}, R={entity_metrics['entity_recall']:.4f})")
    print(f"  Link F1 (text):   {text_only['f1']:.4f}  "
          f"(P={text_only['precision']:.4f}, R={text_only['recall']:.4f})  "
          f"{text_only['tp']}/{text_only['total_gt']} GT, {text_only['total_pred']} pred")
    print(f"  Link F1 (txt+box):{text_bbox['f1']:.4f}  "
          f"(P={text_bbox['precision']:.4f}, R={text_bbox['recall']:.4f})  "
          f"{text_bbox['tp']}/{text_bbox['total_gt']} GT, {text_bbox['total_pred']} pred")
    print("=" * 50)


if __name__ == "__main__":
    main()
