"""
Dataset classes for KVP extraction with LayoutLMv3.

Converts KVP10k HuggingFace dataset to LayoutLMv3-compatible format.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import LayoutLMv3Processor
from PIL import Image
import io
from typing import List, Dict, Tuple, Optional
import config


class KVP10kDataset(Dataset):
    """
    PyTorch Dataset for KVP10k compatible with LayoutLMv3.
    
    Converts annotations to:
    - Entity labels: 0=Other, 1=Key, 2=Value
    - Link labels: Binary adjacency matrix for key-value pairs
    """
    
    def __init__(
        self,
        hf_dataset,
        processor: LayoutLMv3Processor,
        max_seq_length=512,
        include_images=False
    ):
        """
        Args:
            hf_dataset: HuggingFace KVP10k dataset
            processor: LayoutLMv3Processor for tokenization
            max_seq_length: Maximum sequence length
            include_images: Whether to include image features (slower, more memory)
        """
        self.dataset = hf_dataset
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.include_images = include_images
        
        print(f"Initialized KVP10k Dataset:")
        print(f"  Samples: {len(self.dataset)}")
        print(f"  Max seq length: {max_seq_length}")
        print(f"  Include images: {include_images}")
    
    def __len__(self):
        return len(self.dataset)
    
    def _extract_words_and_boxes(self, example):
        """
        Extract words and bounding boxes from annotations.
        
        Returns:
            words: List[str]
            boxes: List[List[int]] - [x1, y1, x2, y2] in pixel coordinates (0-1000)
            entity_labels: List[int] - 0=Other, 1=Key, 2=Value
            link_map: Dict[int, int] - Maps key_idx to value_idx
        """
        annotations = example.get('annotations', [])
        
        words = []
        boxes = []
        entity_labels = []
        link_map = {}
        
        # Build annotation lookup for linking
        ann_by_id = {}
        for ann in annotations:
            ann_id = ann.get('id')
            if ann_id:
                ann_by_id[ann_id] = ann
        
        # Process each annotation
        for idx, ann in enumerate(annotations):
            coords = ann.get('coordinates', [])
            text = ann.get('text', '').strip()
            
            if not coords or not text:
                continue
            
            # Compute bbox in [0, 1000] scale (LayoutLMv3 format)
            xs = [c['x'] for c in coords]
            ys = [c['y'] for c in coords]
            
            # Convert from [0, 1] normalized to [0, 1000] integer
            x1 = int(min(xs) * 1000)
            y1 = int(min(ys) * 1000)
            x2 = int(max(xs) * 1000)
            y2 = int(max(ys) * 1000)
            
            # Ensure valid bbox
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            bbox = [x1, y1, x2, y2]
            
            # Determine entity type
            linking = ann.get('attributes', {}).get('Linking', {}).get('value')
            
            if linking:
                # This is a key
                entity_label = 1
                
                # Check if linked value exists
                if linking in ann_by_id:
                    # Mark this for linking
                    link_map[len(words)] = linking  # Store for later resolution
            else:
                # Could be a value or unlinked text
                # For now, treat as "Other" (we'll mark as Value if it's linked)
                entity_label = 0
            
            words.append(text)
            boxes.append(bbox)
            entity_labels.append(entity_label)
        
        # Second pass: mark values that are linked
        for key_idx, value_id in list(link_map.items()):
            if value_id in ann_by_id:
                # Find the annotation index for this value
                for idx, ann in enumerate(annotations):
                    if ann.get('id') == value_id:
                        if idx < len(entity_labels):
                            entity_labels[idx] = 2  # Mark as Value
                        # Update link_map to point to word index instead of ID
                        # Find word index for this annotation
                        for word_idx in range(len(words)):
                            if word_idx == idx:  # Simplified - assumes 1:1 mapping
                                link_map[key_idx] = word_idx
                        break
        
        return words, boxes, entity_labels, link_map
    
    def _create_link_matrix(self, link_map, seq_length):
        """
        Create binary adjacency matrix for key-value links.
        
        Args:
            link_map: Dict[int, int] - Maps key_idx to value_idx
            seq_length: Total sequence length
        
        Returns:
            link_matrix: [seq_length, seq_length] - binary matrix
        """
        link_matrix = np.zeros((seq_length, seq_length), dtype=np.int64)
        
        for key_idx, value_idx in link_map.items():
            if key_idx < seq_length and value_idx < seq_length:
                link_matrix[key_idx, value_idx] = 1
        
        return link_matrix
    
    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Returns:
            dict with:
                - input_ids: [seq_len]
                - attention_mask: [seq_len]
                - bbox: [seq_len, 4]
                - entity_labels: [seq_len]
                - link_labels: [seq_len, seq_len]
                - pixel_values: [3, 224, 224] (if include_images)
        """
        example = self.dataset[idx]
        
        # Extract structured data
        words, boxes, entity_labels, link_map = self._extract_words_and_boxes(example)
        
        # Handle empty documents
        if not words:
            words = ["[EMPTY]"]
            boxes = [[0, 0, 1, 1]]
            entity_labels = [0]
            link_map = {}
        
        # Prepare image if needed
        image = None
        if self.include_images:
            try:
                image_bytes = example.get('image', {}).get('bytes')
                if image_bytes:
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                else:
                    # Create dummy white image if bytes not available
                    image = Image.new('RGB', (224, 224), color='white')
            except Exception as e:
                print(f"Warning: Could not load image for idx {idx}: {e}")
                # Create dummy white image
                image = Image.new('RGB', (224, 224), color='white')
        else:
            # LayoutLMv3 processor always needs an image, create dummy
            image = Image.new('RGB', (224, 224), color='white')
        
        # Encode with LayoutLMv3 processor
        encoding = self.processor(
            images=image,
            text=words,
            boxes=boxes,
            word_labels=entity_labels,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension (processor adds it)
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'bbox': encoding['bbox'].squeeze(0),
            'entity_labels': encoding['labels'].squeeze(0),
        }
        
        # Add image features if available
        if self.include_images and 'pixel_values' in encoding:
            item['pixel_values'] = encoding['pixel_values'].squeeze(0)
        
        # Create link matrix
        seq_len = item['input_ids'].size(0)
        link_matrix = self._create_link_matrix(link_map, seq_len)
        item['link_labels'] = torch.from_numpy(link_matrix)
        
        return item


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size=8,
    num_workers=0
):
    """
    Create training and validation dataloaders.
    
    Args:
        train_dataset: KVP10kDataset for training
        val_dataset: KVP10kDataset for validation
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    print(f"✓ Created dataloaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("KVP Dataset module ready")
