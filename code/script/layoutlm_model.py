"""
Stage 4: LayoutLMv3 + Linker Model for KVP Extraction.

Architecture:
1. LayoutLMv3: Encodes text, layout, and visual features
2. Entity Recognition: Classifies tokens as Key/Value/Other
3. Linker Module: Learns to pair keys with values based on:
   - Spatial relationships
   - Semantic similarity
   - Layout patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv3Processor, LayoutLMv3Model
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import config


@dataclass
class KVPExample:
    """Single training example for KVP extraction."""
    image_path: str
    words: List[str]
    bboxes: List[List[int]]  # [x1, y1, x2, y2] in pixel coordinates
    labels: List[int]  # 0=Other, 1=Key, 2=Value
    links: List[Tuple[int, int]]  # List of (key_idx, value_idx) pairs


class LayoutLMv3Encoder(nn.Module):
    """
    LayoutLMv3 encoder for document understanding.
    Encodes text + layout + visual features.
    """
    
    def __init__(self, model_name="microsoft/layoutlmv3-base", freeze_base=False):
        super().__init__()
        
        self.model = LayoutLMv3Model.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

        # Enable gradient checkpointing to reduce memory during backward pass.
        # Trades ~20% speed for ~40% memory saving on the encoder.
        self.model.gradient_checkpointing_enable()
        
        if freeze_base:
            # Freeze base model, only train task heads
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, bbox, pixel_values=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            bbox: [batch, seq_len, 4] - normalized [x1, y1, x2, y2] in [0, 1000]
            pixel_values: [batch, 3, 224, 224] - optional image features
        
        Returns:
            sequence_output: [batch, seq_len, hidden_size]
            pooled_output: [batch, hidden_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values
        )
        
        # LayoutLMv3 only returns last_hidden_state (no pooler_output)
        return outputs.last_hidden_state, outputs.last_hidden_state


class EntityClassifier(nn.Module):
    """
    Token classification head for Key/Value/Other detection.
    """
    
    def __init__(self, hidden_size, num_labels=3, dropout=0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, sequence_output):
        """
        Args:
            sequence_output: [batch, seq_len, hidden_size]
        
        Returns:
            logits: [batch, seq_len, num_labels]
        """
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class BiaffineLinker(nn.Module):
    """
    Biaffine attention mechanism for learning key-value links.
    
    For each (key, value) pair, computes a linking score based on:
    - Biaffine transformation of their hidden states
    - Spatial relationship features
    """
    
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        
        # Project keys and values to linking space
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        
        # Biaffine scoring
        self.biaffine = nn.Bilinear(hidden_size, hidden_size, 1)
        
        # Spatial feature encoding
        self.spatial_encoder = nn.Sequential(
            nn.Linear(8, 64),  # [dx, dy, dist, angle, h_align, v_align, area_ratio, aspect_ratio]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )
        
        # Combine biaffine + spatial
        self.final_scorer = nn.Sequential(
            nn.Linear(1 + 32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def compute_spatial_features(self, key_bboxes, value_bboxes):
        """
        Compute spatial relationship features between key and value bboxes.
        
        Args:
            key_bboxes: [batch, num_keys, 4] - normalized [x1, y1, x2, y2]
            value_bboxes: [batch, num_values, 4]
        
        Returns:
            spatial_features: [batch, num_keys, num_values, 8]
        """
        batch_size = key_bboxes.size(0)
        num_keys = key_bboxes.size(1)
        num_values = value_bboxes.size(1)
        
        # Expand for pairwise computation
        # [batch, num_keys, 1, 4]
        keys = key_bboxes.unsqueeze(2)
        # [batch, 1, num_values, 4]
        values = value_bboxes.unsqueeze(1)
        
        # Compute centroids
        key_cx = (keys[..., 0] + keys[..., 2]) / 2
        key_cy = (keys[..., 1] + keys[..., 3]) / 2
        val_cx = (values[..., 0] + values[..., 2]) / 2
        val_cy = (values[..., 1] + values[..., 3]) / 2
        
        # Relative position
        dx = val_cx - key_cx
        dy = val_cy - key_cy
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        angle = torch.atan2(dy, dx)
        
        # Alignment features
        key_h = keys[..., 3] - keys[..., 1]
        val_h = values[..., 3] - values[..., 1]
        h_overlap = torch.clamp(
            torch.min(keys[..., 3], values[..., 3]) - torch.max(keys[..., 1], values[..., 1]),
            min=0
        )
        h_align = h_overlap / (torch.min(key_h, val_h) + 1e-8)
        
        key_w = keys[..., 2] - keys[..., 0]
        val_w = values[..., 2] - values[..., 0]
        v_overlap = torch.clamp(
            torch.min(keys[..., 2], values[..., 2]) - torch.max(keys[..., 0], values[..., 0]),
            min=0
        )
        v_align = v_overlap / (torch.min(key_w, val_w) + 1e-8)
        
        # Size features
        key_area = key_h * key_w
        val_area = val_h * val_w
        area_ratio = val_area / (key_area + 1e-8)
        aspect_ratio = (val_w / (val_h + 1e-8)) / (key_w / (key_h + 1e-8))
        
        # Stack features
        spatial_features = torch.stack([
            dx, dy, dist, angle, h_align, v_align, area_ratio, aspect_ratio
        ], dim=-1)
        
        return spatial_features
    
    def forward(self, sequence_output, entity_logits, bboxes, attention_mask):
        """
        Compute linking scores between all key-value pairs.
        
        Args:
            sequence_output: [batch, seq_len, hidden_size]
            entity_logits: [batch, seq_len, 3] - entity predictions
            bboxes: [batch, seq_len, 4] - normalized bboxes
            attention_mask: [batch, seq_len]
        
        Returns:
            link_scores: [batch, num_keys, num_values] - pairwise linking scores
            key_indices: [batch, num_keys] - indices of key tokens
            value_indices: [batch, num_values] - indices of value tokens
        """
        batch_size, seq_len, hidden_size = sequence_output.shape
        entity_preds = torch.argmax(entity_logits, dim=-1)  # [batch, seq_len]

        all_link_scores = []
        all_key_indices = []
        all_value_indices = []

        for b in range(batch_size):
            key_mask = (entity_preds[b] == 1) & (attention_mask[b] == 1)
            val_mask = (entity_preds[b] == 2) & (attention_mask[b] == 1)

            key_idx = key_mask.nonzero(as_tuple=True)[0]
            val_idx = val_mask.nonzero(as_tuple=True)[0]

            if len(key_idx) == 0 or len(val_idx) == 0:
                all_link_scores.append(None)
                all_key_indices.append(key_idx)
                all_value_indices.append(val_idx)
                continue

            key_reps = self.key_projection(
                self.dropout(sequence_output[b][key_idx])
            )  # [nk, hidden]
            val_reps = self.value_projection(
                self.dropout(sequence_output[b][val_idx])
            )  # [nv, hidden]

            nk, nv = len(key_idx), len(val_idx)

            # Biaffine scores
            k_exp = key_reps.unsqueeze(1).expand(nk, nv, hidden_size).reshape(-1, hidden_size)
            v_exp = val_reps.unsqueeze(0).expand(nk, nv, hidden_size).reshape(-1, hidden_size)
            biaffine_scores = self.biaffine(k_exp, v_exp).reshape(nk, nv, 1)

            # Spatial scores
            key_boxes = bboxes[b][key_idx].unsqueeze(0).float()  # [1, nk, 4]
            val_boxes = bboxes[b][val_idx].unsqueeze(0).float()  # [1, nv, 4]
            spatial_feats = self.compute_spatial_features(
                key_boxes, val_boxes
            ).squeeze(0)  # [nk, nv, 8]
            spatial_scores = self.spatial_encoder(spatial_feats)  # [nk, nv, 32]

            # Combine and score
            combined = torch.cat([biaffine_scores, spatial_scores], dim=-1)  # [nk, nv, 33]
            link_scores = self.final_scorer(combined).squeeze(-1)  # [nk, nv]

            all_link_scores.append(link_scores)
            all_key_indices.append(key_idx)
            all_value_indices.append(val_idx)

        return all_link_scores, all_key_indices, all_value_indices


class LayoutLMv3KVPModel(nn.Module):
    """
    Complete model for KVP extraction:
    1. LayoutLMv3 encoder
    2. Entity classifier (Key/Value/Other)
    3. Biaffine linker (pairs keys with values) - OPTIONAL for ablation
    """
    
    def __init__(
        self,
        layoutlmv3_model="microsoft/layoutlmv3-base",
        freeze_base=False,
        num_labels=3,
        dropout=0.1,
        use_linker=True  # NEW: Enable/disable linking module
    ):
        super().__init__()
        
        self.encoder = LayoutLMv3Encoder(layoutlmv3_model, freeze_base)
        self.entity_classifier = EntityClassifier(
            self.encoder.hidden_size,
            num_labels,
            dropout
        )
        
        # Only create linker if enabled (Stage 4b)
        self.use_linker = use_linker
        if use_linker:
            self.linker = BiaffineLinker(self.encoder.hidden_size, dropout)
        else:
            self.linker = None
        
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids,
        attention_mask,
        bbox,
        pixel_values=None,
        entity_labels=None,
        link_labels=None
    ):
        """
        Forward pass with optional training labels.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            bbox: [batch, seq_len, 4]
            pixel_values: [batch, 3, 224, 224] - optional
            entity_labels: [batch, seq_len] - optional, for training
            link_labels: [batch, seq_len, seq_len] - optional, binary adjacency matrix
        
        Returns:
            dict with:
                - entity_logits: [batch, seq_len, num_labels]
                - link_scores: [batch, num_keys, num_values] or None
                - loss: scalar (if labels provided)
        """
        # Encode document
        sequence_output, pooled_output = self.encoder(
            input_ids, attention_mask, bbox, pixel_values
        )
        
        # Classify entities
        entity_logits = self.entity_classifier(sequence_output)
        
        # LayoutLMv3 appends visual patch tokens (~197) to sequence_output
        # Truncate back to text-only sequence length to match attention_mask, entity_labels, etc.
        text_seq_len = input_ids.shape[1]  # e.g., 512
        entity_logits = entity_logits[:, :text_seq_len, :]  # [batch, text_seq_len, num_labels]
        sequence_output_text = sequence_output[:, :text_seq_len, :]  # [batch, text_seq_len, hidden]
        
        # Compute linking scores (only if linker enabled)
        link_scores = None
        key_indices = None
        value_indices = None
        
        if self.use_linker and self.linker is not None:
            link_scores, key_indices, value_indices = self.linker(
                sequence_output_text, entity_logits, bbox, attention_mask
            )
        
        # Compute loss if labels provided
        loss = None
        entity_loss = None
        link_loss = None
        
        if entity_labels is not None:
            # Entity classification loss
            loss_fct = nn.CrossEntropyLoss()
            
            # Compute loss only on non-padding tokens (attention_mask == 1)
            # Use reshape() instead of view() to handle non-contiguous tensors from truncation
            active_loss = attention_mask.reshape(-1) == 1
            active_logits = entity_logits.reshape(-1, self.num_labels)[active_loss]
            active_labels = entity_labels.reshape(-1)[active_loss]
            
            entity_loss = loss_fct(active_logits, active_labels)
            
            loss = entity_loss
            
            # Add linking loss if linker enabled and labels provided
            if self.use_linker and link_labels is not None and link_scores is not None:
                link_loss_total = 0.0
                link_count = 0
                for b, scores in enumerate(link_scores):
                    if scores is None:
                        continue
                    k_idx = key_indices[b]
                    v_idx = value_indices[b]
                    if len(k_idx) == 0 or len(v_idx) == 0:
                        continue
                    # Extract GT linking sub-matrix [nk, nv]
                    gt_links = link_labels[b][k_idx][:, v_idx].float()
                    link_loss_total += F.binary_cross_entropy_with_logits(
                        scores, gt_links
                    )
                    link_count += 1
                if link_count > 0:
                    link_loss = link_loss_total / link_count
                    loss = entity_loss + link_loss
                else:
                    link_loss = None
        
        return {
            'entity_logits': entity_logits,
            'link_scores': link_scores,
            'key_indices': key_indices,
            'value_indices': value_indices,
            'loss': loss,
            'entity_loss': entity_loss,
            'link_loss': link_loss,
            'sequence_output': sequence_output
        }
    
    def predict_kvp(
        self,
        input_ids,
        attention_mask,
        bbox,
        pixel_values=None,
        words=None,
        score_threshold=0.5
    ):
        """
        Predict KV pairs for inference.
        
        Args:
            All encoding inputs
            words: List[str] - original words for output
            score_threshold: Minimum linking score
        
        Returns:
            List of predicted KV pairs with scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids, attention_mask, bbox, pixel_values
            )
            
            entity_logits = outputs['entity_logits']
            link_scores = outputs['link_scores']
            key_indices = outputs.get('key_indices')
            value_indices = outputs.get('value_indices')
            
            predictions = []
            batch_size = input_ids.size(0)

            for b in range(batch_size):
                entity_preds_b = torch.argmax(entity_logits[b], dim=-1)
                batch_preds = []

                if self.use_linker and link_scores is not None and link_scores[b] is not None:
                    scores = link_scores[b]
                    k_idx = key_indices[b]
                    v_idx = value_indices[b]

                    best_val_positions = torch.argmax(scores, dim=1)
                    best_scores = torch.sigmoid(
                        scores[range(len(k_idx)), best_val_positions]
                    )

                    for i, ki in enumerate(k_idx):
                        if best_scores[i].item() < score_threshold:
                            continue
                        vi = v_idx[best_val_positions[i]]
                        key_word = words[b][ki] if words else str(ki.item())
                        val_word = words[b][vi] if words else str(vi.item())
                        batch_preds.append({
                            "key": key_word,
                            "value": val_word,
                            "key_bbox": bbox[b][ki].tolist(),
                            "value_bbox": bbox[b][vi].tolist(),
                            "link_score": best_scores[i].item()
                        })
                else:
                    key_positions = (entity_preds_b == 1).nonzero(as_tuple=True)[0]

                    for ki in key_positions:
                        key_word = words[b][ki] if words else str(ki.item())
                        batch_preds.append({
                            "key": key_word,
                            "value": None,
                            "key_bbox": bbox[b][ki].tolist(),
                            "value_bbox": None,
                            "link_score": None
                        })

                predictions.append(batch_preds)

            return predictions


def create_model(
    model_name="microsoft/layoutlmv3-base",
    freeze_base=False,
    use_linker=True,
    device=None
):
    """
    Factory function to create and initialize the model.
    """
    if device is None:
        device = config.DEVICE
    
    model = LayoutLMv3KVPModel(
        layoutlmv3_model=model_name,
        freeze_base=freeze_base,
        num_labels=3,
        dropout=0.1,
        use_linker=use_linker
    )
    
    model = model.to(device)
    
    print(f"✓ Created LayoutLMv3 KVP Model")
    print(f"  Encoder: {model_name}")
    print(f"  Freeze base: {freeze_base}")
    print(f"  Use linker: {use_linker} ({'Stage 4b - With Linker' if use_linker else 'Stage 4a - No Linker'})")
    print(f"  Gradient checkpointing: enabled")
    print(f"  Device: {device}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


if __name__ == "__main__":
    print("LayoutLMv3 KVP Model module ready")
    model = create_model(freeze_base=True)
    print("\n✓ Model initialized successfully!")
