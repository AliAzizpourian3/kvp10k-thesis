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
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from layoutlm_model import create_model
from stage4_kvp_dataset import LayoutLMv3PreparedDataset, PaddedBatchCollator

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
        key_text = kvp.get("key", {}).get("text", "").strip().lower()
        val_text = kvp.get("value", {}).get("text", "").strip().lower()
        if key_text and val_text:
            pairs.append((key_text, val_text))
    return pairs, data.get("hash_name", "")


def _text_overlap(pred_text, gt_text):
    """Check if predicted text and ground truth text have significant overlap.

    The model predicts at the token level and reconstructs text from individual
    tokens, so exact match is too strict. We check bidirectional containment
    or word-level F1 >= 0.5 instead.
    """
    pred_text = pred_text.strip().lower()
    gt_text = gt_text.strip().lower()

    if not pred_text or not gt_text:
        return False

    # Exact match
    if pred_text == gt_text:
        return True

    # Containment
    if pred_text in gt_text or gt_text in pred_text:
        return True

    # Word-level F1
    pred_words = set(pred_text.split())
    gt_words = set(gt_text.split())
    if not pred_words or not gt_words:
        return False
    common = pred_words & gt_words
    if not common:
        return False
    precision = len(common) / len(pred_words)
    recall = len(common) / len(gt_words)
    f1 = 2 * precision * recall / (precision + recall)
    return f1 >= 0.5


def _extract_predicted_pairs(model, batch, device, dataset, idx_start, score_threshold=0.5):
    """Run model inference and extract predicted key-value text pairs.

    Returns a list of lists (one per batch item) of (key_text, val_text, score).
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

        # For each predicted key token, find its best value
        best_val_pos = torch.argmax(scores_b, dim=1)
        best_scores = torch.sigmoid(scores_b[range(len(k_idx)), best_val_pos])

        for i, ki in enumerate(k_idx):
            if best_scores[i].item() < score_threshold:
                continue
            vi = v_idx[best_val_pos[i]]
            ki_int = ki.item()
            vi_int = vi.item()

            # Map token index to word index
            if ki_int < len(word_ids) and word_ids[ki_int] is not None:
                key_word_idx = word_ids[ki_int]
                key_text = words[key_word_idx] if key_word_idx < len(words) else ""
            else:
                key_text = ""

            if vi_int < len(word_ids) and word_ids[vi_int] is not None:
                val_word_idx = word_ids[vi_int]
                val_text = words[val_word_idx] if val_word_idx < len(words) else ""
            else:
                val_text = ""

            if key_text and val_text:
                pairs.append((key_text.lower(), val_text.lower(), best_scores[i].item()))

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


def compute_link_metrics(model, dataset, dataloader, device, score_threshold=0.5):
    """Compute link-level P/R/F1 by comparing predicted pairs to GT pairs.

    A predicted pair (pred_key, pred_val) matches a GT pair (gt_key, gt_val) if
    both _text_overlap(pred_key, gt_key) and _text_overlap(pred_val, gt_val).
    Each GT pair can be matched at most once (greedy, best-score first).
    """
    tp = 0
    total_pred = 0
    total_gt = 0
    model.eval()

    idx_start = 0
    for batch in tqdm(dataloader, desc="Link eval"):
        batch_size = batch["input_ids"].size(0)

        # Get predicted pairs
        pred_pairs_batch = _extract_predicted_pairs(
            model, batch, device, dataset, idx_start, score_threshold
        )

        for b in range(batch_size):
            sample_idx = idx_start + b
            if sample_idx >= len(dataset):
                continue

            # Get ground truth
            json_file = dataset.json_files[sample_idx]
            gt_pairs, _ = _get_gt_kvps(json_file)
            pred_pairs = pred_pairs_batch[b]

            total_gt += len(gt_pairs)
            total_pred += len(pred_pairs)

            # Greedy matching: sort predictions by score (descending), match each
            pred_sorted = sorted(pred_pairs, key=lambda x: x[2], reverse=True)
            gt_matched = [False] * len(gt_pairs)

            for pk, pv, _ in pred_sorted:
                for j, (gk, gv) in enumerate(gt_pairs):
                    if gt_matched[j]:
                        continue
                    if _text_overlap(pk, gk) and _text_overlap(pv, gv):
                        tp += 1
                        gt_matched[j] = True
                        break

        idx_start += batch_size

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "link_precision": precision,
        "link_recall": recall,
        "link_f1": f1,
        "link_tp": tp,
        "link_total_pred": total_pred,
        "link_total_gt": total_gt,
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
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Evaluation batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Create model
    model = create_model(use_linker=True, device=device)

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

    # Link metrics
    logger.info("Computing link metrics...")
    link_metrics = compute_link_metrics(
        model, dataset, dataloader, device, args.score_threshold
    )
    logger.info(f"Link   P={link_metrics['link_precision']:.4f}  "
                f"R={link_metrics['link_recall']:.4f}  "
                f"F1={link_metrics['link_f1']:.4f}")
    logger.info(f"Link   TP={link_metrics['link_tp']}  "
                f"Pred={link_metrics['link_total_pred']}  "
                f"GT={link_metrics['link_total_gt']}")

    # Save results
    results = {**entity_metrics, **link_metrics, "checkpoint_dir": args.checkpoint_dir,
               "score_threshold": args.score_threshold}
    out_path = Path(args.checkpoint_dir) / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 50)
    print(f"EVALUATION: {args.checkpoint_dir}")
    print("=" * 50)
    print(f"  Entity F1:  {entity_metrics['entity_f1']:.4f}  "
          f"(P={entity_metrics['entity_precision']:.4f}, R={entity_metrics['entity_recall']:.4f})")
    print(f"  Link F1:    {link_metrics['link_f1']:.4f}  "
          f"(P={link_metrics['link_precision']:.4f}, R={link_metrics['link_recall']:.4f})")
    print(f"  Link pairs: {link_metrics['link_tp']}/{link_metrics['link_total_gt']} GT matched, "
          f"{link_metrics['link_total_pred']} predicted")
    print("=" * 50)


if __name__ == "__main__":
    main()
