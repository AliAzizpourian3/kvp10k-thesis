"""
LayoutLMv3 dataset adapters for prepared KVP10k data.

Extends kvp_dataset.py with:
- LayoutLMv3PreparedDataset: Loads from data/prepared/train/*.json files
- create_stage4_dataloaders: Creates train/val/test splits with LayoutLMv3Processor

IMPORTANT — offline operation:
  Compute nodes have no internet access. The LayoutLMv3 processor/tokenizer
  must be pre-downloaded on a login node before submitting jobs:

    python - <<'EOF'
    from transformers import LayoutLMv3Processor
    LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    print("Cache ready.")
    EOF

  All from_pretrained() calls below use local_files_only=True and will raise
  a clear error if the cache is missing rather than timing out.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import LayoutLMv3Processor
from PIL import Image
import io
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LAYOUTLMV3_MODEL = "microsoft/layoutlmv3-base"


def _load_processor(local_files_only: bool = True) -> LayoutLMv3Processor:
    """
    Load LayoutLMv3Processor from local HuggingFace cache.

    local_files_only=True (default) prevents any network call on compute nodes.
    Raises OSError with a helpful message if the cache is missing.
    """
    try:
        processor = LayoutLMv3Processor.from_pretrained(
            LAYOUTLMV3_MODEL,
            apply_ocr=False,
            local_files_only=local_files_only
        )
        logger.info(f"Processor loaded from local cache: {LAYOUTLMV3_MODEL}")
        return processor
    except OSError as e:
        raise OSError(
            f"\n\nLayoutLMv3 processor not found in local HF cache.\n"
            f"Pre-download it on a login node (internet access required):\n\n"
            f"  python -c \"from transformers import LayoutLMv3Processor; "
            f"LayoutLMv3Processor.from_pretrained('{LAYOUTLMV3_MODEL}', apply_ocr=False)\"\n\n"
            f"Then re-submit the SLURM job.\n"
            f"Original error: {e}"
        ) from e


class LayoutLMv3PreparedDataset(Dataset):
    """
    PyTorch Dataset for LayoutLMv3 loading from prepared JSON files (Stage 3 output).

    Input: data/prepared/train/*.json or data/prepared/test/*.json
    Format: IBM KVP10k standard with lmdx_text encoding

    Each JSON contains:
      - hash_name: Unique document ID
      - lmdx_text: Text with word-level coordinates (format: "word x1|y1|x2|y2")
      - image_width, image_height: Document dimensions
      - gt_kvps: Ground-truth key-value pairs

    Output: LayoutLMv3-compatible tensors with:
      - entity_labels: 0=Other, 1=Key, 2=Value
      - link_labels: Key-value relationship adjacency matrix [seq_len x seq_len]
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        processor: Optional[LayoutLMv3Processor] = None,
        max_seq_length: int = 512,
        include_images: bool = False,
        image_base_dir: Optional[str] = None
    ):
        """
        Args:
            data_dir: Path to data/prepared/ directory
            split: 'train' or 'test'
            processor: LayoutLMv3Processor (pass pre-loaded instance to avoid
                       loading it twice; if None, loads from local cache)
            max_seq_length: Maximum sequence length (LayoutLMv3 limit is 512)
            include_images: Whether to load actual pixel values from image files
            image_base_dir: Base directory for PDF images (if include_images=True)
        """
        self.data_dir = Path(data_dir) / split
        # Use provided processor or load from local cache (no network call).
        self.processor = processor if processor is not None else _load_processor()
        self.max_seq_length = max_seq_length
        self.include_images = include_images
        self.image_base_dir = image_base_dir

        self.json_files = sorted(self.data_dir.glob("*.json"))

        if not self.json_files:
            raise ValueError(f"No JSON files found in {self.data_dir}")

        logger.info(f"Loaded {len(self.json_files)} samples from {self.data_dir}")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        """Load and process a single sample from prepared JSON."""
        json_file = self.json_files[idx]

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {json_file}: {e}")
            return self._get_empty_example()

        lmdx_text = data.get('lmdx_text', '')
        words, word_bboxes = self._parse_lmdx_text(lmdx_text, data)

        if not words or not word_bboxes:
            logger.warning(f"Could not parse lmdx_text from {json_file}")
            return self._get_empty_example()

        image_width = data.get('image_width', 1)
        image_height = data.get('image_height', 1)

        bboxes_normalized = self._normalize_bboxes(word_bboxes, image_width, image_height)

        image = None
        if self.include_images and self.image_base_dir:
            image_path = Path(self.image_base_dir) / f"{data['hash_name']}.png"
            if image_path.exists():
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    logger.debug(f"Could not load image {image_path}: {e}")

        if image is None:
            image = Image.new('RGB', (224, 224), color=(255, 255, 255))

        gt_kvps = data.get('gt_kvps', {}).get('kvps_list', [])
        entity_labels, link_labels = self._generate_labels(
            words,
            gt_kvps,
            word_bboxes,
            image_width,
            image_height
        )

        try:
            encoded = self.processor(
                images=image,
                text=words,
                boxes=bboxes_normalized,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_seq_length,
                truncation=True
            )
        except Exception as e:
            logger.warning(f"Failed to process {json_file}: {e}")
            return self._get_empty_example()

        word_ids = encoded.word_ids()

        entity_labels_token = self._align_labels_to_tokens(entity_labels, word_ids)
        link_labels_token = self._align_link_labels_to_tokens(link_labels, word_ids)

        item = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'bbox': encoded['bbox'].squeeze(0),
            'entity_labels': entity_labels_token,
            'link_labels': link_labels_token,
            'hash_name': data['hash_name']
        }

        if image is not None and 'pixel_values' in encoded:
            item['pixel_values'] = encoded['pixel_values'].squeeze(0)
        else:
            item['pixel_values'] = torch.zeros(3, 224, 224, dtype=torch.float32)

        return item

    def _parse_lmdx_text(self, lmdx_text: str, data: Dict) -> Tuple[List[str], List[List[int]]]:
        """
        Parse lmdx_text format to extract words and bounding boxes.

        Format: "word1 x1|y1|x2|y2\nword2 x1|y1|x2|y2"
        Coordinates are in pixel space (not quantized).
        """
        words = []
        bboxes = []

        lines = lmdx_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or '|' not in line:
                continue

            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                continue

            word_text = parts[0].strip()
            coords_str = parts[1].strip()

            try:
                coords = coords_str.split('|')
                if len(coords) != 4:
                    continue

                x1 = max(0, int(coords[0]))
                y1 = max(0, int(coords[1]))
                x2 = max(x1, int(coords[2]))
                y2 = max(y1, int(coords[3]))

                words.append(word_text)
                bboxes.append([x1, y1, x2, y2])

            except (ValueError, IndexError):
                logger.debug(f"Could not parse coordinates from line: {line}")
                continue

        return words, bboxes

    @staticmethod
    def _normalize_bboxes(
        bboxes: List[List[int]],
        image_width: int,
        image_height: int
    ) -> List[List[int]]:
        """
        Normalize bboxes from pixel coordinates to [0, 1000] scale.
        LayoutLMv3 expects normalized coordinates in range [0, 1000].
        """
        normalized = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            x1_norm = int((x1 / image_width * 1000)) if image_width > 0 else 0
            y1_norm = int((y1 / image_height * 1000)) if image_height > 0 else 0
            x2_norm = int((x2 / image_width * 1000)) if image_width > 0 else 1000
            y2_norm = int((y2 / image_height * 1000)) if image_height > 0 else 1000

            x1_norm = max(0, min(1000, x1_norm))
            y1_norm = max(0, min(1000, y1_norm))
            x2_norm = max(0, min(1000, x2_norm))
            y2_norm = max(0, min(1000, y2_norm))

            normalized.append([x1_norm, y1_norm, x2_norm, y2_norm])

        return normalized

    @staticmethod
    def _align_labels_to_tokens(
        word_labels: List[int],
        word_ids: List[Optional[int]]
    ) -> torch.Tensor:
        """
        Align word-level labels to token level using word_ids.
        Special tokens (CLS, SEP, PAD) have word_idx = None -> label 0.
        """
        token_labels = torch.zeros(len(word_ids), dtype=torch.long)

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx < len(word_labels):
                token_labels[token_idx] = word_labels[word_idx]

        return token_labels

    @staticmethod
    def _align_link_labels_to_tokens(
        word_link_labels: torch.Tensor,
        word_ids: List[Optional[int]]
    ) -> torch.Tensor:
        """
        Align word-level link labels to token level.
        Converts [max_seq, max_seq] word-level matrix to token-level.
        """
        max_len = word_link_labels.shape[0]
        token_link_labels = torch.zeros((max_len, max_len), dtype=torch.float32)

        for token_i, word_i in enumerate(word_ids):
            if word_i is None or token_i >= max_len:
                continue
            for token_j, word_j in enumerate(word_ids):
                if word_j is None or token_j >= max_len:
                    continue
                if word_i < max_len and word_j < max_len:
                    token_link_labels[token_i, token_j] = word_link_labels[word_i, word_j]

        return token_link_labels

    def _generate_labels(
        self,
        words: List[str],
        kvps: List[Dict],
        bboxes: List[List[int]],
        image_width: int,
        image_height: int
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Generate word-level entity labels and key-value relationship links.
        Entity labels: 0=Other, 1=Key, 2=Value
        Link labels: [seq_len x seq_len] binary adjacency matrix
        """
        entity_labels = [0] * len(words)
        link_labels = torch.zeros(
            (self.max_seq_length, self.max_seq_length),
            dtype=torch.float32
        )

        if not kvps or not bboxes:
            return entity_labels, link_labels

        for kvp in kvps:
            key_bbox = kvp.get("key", {}).get("bbox", None)
            val_bbox = kvp.get("value", {}).get("bbox", None)

            if not key_bbox or not val_bbox:
                continue

            key_indices = self._find_words_by_bbox_overlap(bboxes, key_bbox)
            val_indices = self._find_words_by_bbox_overlap(bboxes, val_bbox)

            for idx in key_indices:
                if idx < len(entity_labels):
                    entity_labels[idx] = 1
            for idx in val_indices:
                if idx < len(entity_labels):
                    entity_labels[idx] = 2

            for key_idx in key_indices:
                for val_idx in val_indices:
                    if key_idx < self.max_seq_length and val_idx < self.max_seq_length:
                        link_labels[key_idx, val_idx] = 1.0

        return entity_labels, link_labels

    @staticmethod
    def _find_words_by_bbox_overlap(
        word_bboxes: List[List[int]],
        target_bbox: List[int]
    ) -> List[int]:
        """Find word indices with >50% bbox overlap with target_bbox."""
        target_x1, target_y1, target_x2, target_y2 = target_bbox
        matching_indices = []

        for idx, bbox in enumerate(word_bboxes):
            wx1, wy1, wx2, wy2 = bbox

            inter_x1 = max(wx1, target_x1)
            inter_y1 = max(wy1, target_y1)
            inter_x2 = min(wx2, target_x2)
            inter_y2 = min(wy2, target_y2)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

                if target_area > 0 and intersection_area / target_area > 0.5:
                    matching_indices.append(idx)

        return matching_indices

    def _get_empty_example(self):
        """Return a zero-padded example for empty/invalid data."""
        return {
            'input_ids': torch.zeros(self.max_seq_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_seq_length, dtype=torch.long),
            'bbox': torch.zeros((self.max_seq_length, 4), dtype=torch.long),
            'entity_labels': torch.zeros(self.max_seq_length, dtype=torch.long),
            'link_labels': torch.zeros(
                (self.max_seq_length, self.max_seq_length),
                dtype=torch.float32
            ),
            'pixel_values': torch.zeros(3, 224, 224, dtype=torch.float32),
            'hash_name': 'empty'
        }


class PaddedBatchCollator:
    """Custom collator for variable-length LayoutLMv3 sequences."""

    def __call__(self, batch):
        """Collate batch into padded tensors."""
        if not batch:
            raise ValueError("Empty batch")

        keys = batch[0].keys()
        collated = {}

        for key in keys:
            if key == 'hash_name':
                collated[key] = [item[key] for item in batch]
            elif key in ['input_ids', 'attention_mask', 'entity_labels']:
                collated[key] = torch.stack([item[key] for item in batch])
            elif key == 'bbox':
                collated[key] = torch.stack([item[key] for item in batch])
            elif key == 'pixel_values':
                collated[key] = torch.stack([item[key] for item in batch])
            elif key == 'link_labels':
                collated[key] = torch.stack([item[key] for item in batch])

        return collated


def create_stage4_dataloaders(
    data_dir: str = "data/prepared",
    batch_size: int = 4,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    include_images: bool = False,
    image_base_dir: Optional[str] = None
) -> dict:
    """
    Create train/val/test dataloaders for Stage 4.

    Processor is loaded once here and shared across all three Dataset instances
    to avoid redundant cache reads and any risk of a second network call.
    """
    from torch.utils.data import random_split

    # Load processor ONCE from local cache. Fails fast with a helpful message
    # if pre-download was not done on the login node.
    processor = _load_processor()

    logger.info(f"Loading training data from {data_dir}/train/")
    train_dataset = LayoutLMv3PreparedDataset(
        data_dir=data_dir,
        split="train",
        processor=processor,
        include_images=include_images,
        image_base_dir=image_base_dir
    )

    num_train = len(train_dataset)
    num_val = int(num_train * val_fraction)
    num_train = num_train - num_val

    train_split, val_split = random_split(
        train_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Train: {len(train_split)} | Val: {len(val_split)}")

    logger.info(f"Loading test data from {data_dir}/test/")
    test_dataset = LayoutLMv3PreparedDataset(
        data_dir=data_dir,
        split="test",
        processor=processor,
        include_images=include_images,
        image_base_dir=image_base_dir
    )
    logger.info(f"Test: {len(test_dataset)}")

    collator = PaddedBatchCollator()

    dataloaders = {
        'train': DataLoader(
            train_split,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        ),
        'val': DataLoader(
            val_split,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )
    }

    return dataloaders


if __name__ == "__main__":
    logger.info("Testing LayoutLMv3PreparedDataset...")

    dataloaders = create_stage4_dataloaders(
        data_dir="../../data/prepared",
        batch_size=4,
        include_images=False
    )

    batch = next(iter(dataloaders['train']))
    logger.info("\u2713 Sample batch loaded:")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            logger.info(f"  {key}: {val.shape}")
        else:
            logger.info(f"  {key}: {type(val)}")
