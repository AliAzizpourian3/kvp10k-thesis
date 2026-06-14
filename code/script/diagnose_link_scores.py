#!/usr/bin/env python3
"""
Quick diagnostic: dump link score distribution from a trained model.

For each test document, reports:
  - Number of key/value spans found
  - Raw logit scores (before sigmoid) for all span pairs
  - Best score per key span
  - Whether the best score passes various thresholds

This tells us whether the linker is producing meaningful scores
or if everything is near zero.
"""

import torch
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from layoutlm_model_v2 import create_model
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
        ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        state = torch.load(ckpts[-1] / "pytorch_model.bin", map_location=device, weights_only=False)
        print(f"Loaded: {ckpts[-1]}")

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Load data
    dataset = LayoutLMv3PreparedDataset(args.data_dir, "test")
    loader = DataLoader(dataset, batch_size=1, collate_fn=PaddedBatchCollator())

    all_best_logits = []
    all_best_sigmoids = []
    all_raw_logits = []
    docs_with_spans = 0
    docs_no_spans = 0
    total_key_spans = 0
    total_val_spans = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Scoring")):
            if i >= args.max_samples:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)

            outputs = model(input_ids, attention_mask, bbox, pixel_values)

            link_scores = outputs["link_scores"]
            key_idx = outputs["key_indices"]
            val_idx = outputs["value_indices"]

            if link_scores is None or link_scores[0] is None:
                docs_no_spans += 1
                continue

            scores = link_scores[0]  # [nk, nv]
            nk = len(key_idx[0])
            nv = len(val_idx[0])
            docs_with_spans += 1
            total_key_spans += nk
            total_val_spans += nv

            # All raw logits
            flat = scores.cpu().numpy().flatten()
            all_raw_logits.extend(flat.tolist())

            # Best per key
            best_per_key = scores.max(dim=1).values.cpu().numpy()
            all_best_logits.extend(best_per_key.tolist())
            all_best_sigmoids.extend((1 / (1 + np.exp(-best_per_key))).tolist())

    print(f"\n{'='*60}")
    print(f"LINK SCORE DIAGNOSTIC ({min(args.max_samples, i+1)} documents)")
    print(f"{'='*60}")
    print(f"Docs with key+val spans: {docs_with_spans}")
    print(f"Docs with NO spans:      {docs_no_spans}")
    print(f"Avg key spans/doc:       {total_key_spans/max(docs_with_spans,1):.1f}")
    print(f"Avg val spans/doc:       {total_val_spans/max(docs_with_spans,1):.1f}")
    print()

    if all_raw_logits:
        raw = np.array(all_raw_logits)
        print(f"ALL LOGITS (key×val pairs): n={len(raw)}")
        print(f"  min={raw.min():.4f}  max={raw.max():.4f}  mean={raw.mean():.4f}  std={raw.std():.4f}")
        print(f"  Percentiles: 1%={np.percentile(raw,1):.4f}  25%={np.percentile(raw,25):.4f}  "
              f"50%={np.percentile(raw,50):.4f}  75%={np.percentile(raw,75):.4f}  99%={np.percentile(raw,99):.4f}")
        sigs = 1 / (1 + np.exp(-raw))
        print(f"  Sigmoid > 0.5: {(sigs > 0.5).sum()} / {len(sigs)}")
        print(f"  Sigmoid > 0.3: {(sigs > 0.3).sum()} / {len(sigs)}")
        print(f"  Sigmoid > 0.1: {(sigs > 0.1).sum()} / {len(sigs)}")
        print()

    if all_best_logits:
        best = np.array(all_best_logits)
        best_sig = np.array(all_best_sigmoids)
        print(f"BEST LOGIT PER KEY SPAN: n={len(best)}")
        print(f"  min={best.min():.4f}  max={best.max():.4f}  mean={best.mean():.4f}  std={best.std():.4f}")
        print(f"  Sigmoid > 0.5: {(best_sig > 0.5).sum()} / {len(best_sig)}")
        print(f"  Sigmoid > 0.3: {(best_sig > 0.3).sum()} / {len(best_sig)}")
        print(f"  Sigmoid > 0.1: {(best_sig > 0.1).sum()} / {len(best_sig)}")
        print()

        # Histogram
        print("BEST SIGMOID DISTRIBUTION:")
        for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                       (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
            count = ((best_sig >= lo) & (best_sig < hi)).sum()
            bar = "█" * int(count / max(len(best_sig), 1) * 50)
            print(f"  [{lo:.1f}-{hi:.1f}): {count:5d}  {bar}")


if __name__ == "__main__":
    main()
