"""Export V4 predictions including recovered unvalued/unkeyed spans.

The headline V4 export keeps only linker-consumed regular pairs. This script
runs the same checkpoint and additionally emits every detected key span with no
above-threshold link as ``unvalued`` and every value span never chosen by a
regular pair as ``unkeyed`` (see
``evaluate_stage4b._extract_all_type_predictions``). It writes standard KVP10k
``kvps_list`` files so ``evaluate_kvp10k_benchmark.py`` can score the All /
Unkeyed / Unvalued categories without retraining.
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
from torch.utils.data import DataLoader

from evaluate_stage4b import _extract_all_type_predictions, _load_checkpoint
from layoutlm_model import create_model as create_model_v1
from layoutlm_model_v2 import create_model as create_model_v2
from stage4_kvp_dataset import LayoutLMv3PreparedDataset, PaddedBatchCollator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _to_kvp(prediction: dict) -> dict:
    kvp_type = prediction["type"]
    entry = {"type": kvp_type, "link_confidence": prediction.get("confidence", 0.0)}
    if kvp_type != "unkeyed":
        entry["key"] = prediction["key"]
    if kvp_type != "unvalued":
        entry["value"] = prediction["value"]
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_dir", default="data/prepared")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_version", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.model_version == "v2":
        model = create_model_v2(use_linker=True, device=device)
    else:
        model = create_model_v1(use_linker=True, device=device)
    model = _load_checkpoint(model, args.checkpoint_dir, device)
    model.eval()

    dataset = LayoutLMv3PreparedDataset(
        data_dir=args.data_dir,
        split="test",
        processor=model.encoder.processor if hasattr(model.encoder, "processor") else None,
        max_seq_length=512,
        include_images=False,
    )
    logger.info(f"Test set: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=PaddedBatchCollator(), num_workers=0
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    document_count = 0
    counts = {"kvp": 0, "unvalued": 0, "unkeyed": 0}
    idx_start = 0
    for batch in dataloader:
        batch_predictions = _extract_all_type_predictions(
            model, batch, device, dataset, idx_start, args.score_threshold
        )
        for offset, predictions in enumerate(batch_predictions):
            sample_idx = idx_start + offset
            if sample_idx >= len(dataset):
                continue
            document_name = dataset.json_files[sample_idx].name
            ordered = sorted(
                predictions, key=lambda item: item.get("confidence", 0.0), reverse=True
            )
            kvps = [_to_kvp(prediction) for prediction in ordered]
            for prediction in ordered:
                counts[prediction["type"]] += 1
            with (args.output_dir / document_name).open("w", encoding="utf-8") as handle:
                json.dump({"kvps_list": kvps}, handle, indent=2)
            document_count += 1
        idx_start += len(batch_predictions)

    logger.info(
        f"Exported {document_count} documents  "
        f"kvp={counts['kvp']}  unvalued={counts['unvalued']}  unkeyed={counts['unkeyed']}"
    )


if __name__ == "__main__":
    main()
