"""
Stage 4: LayoutLMv3 + Linker Model for KVP Extraction.

Architecture:
1. LayoutLMv3: Encodes text, layout, and visual features
2. Entity Recognition: Classifies tokens as Key/Value/Other
3. Linker Module: Learns to pair keys with values based on:
   - Spatial relationships
   - Semantic similarity
   - Layout patterns

Memory optimisations (Stage 4b OOM fix):
- Biaffine linker uses chunked scoring (LINKER_CHUNK_SIZE=8) instead of
  materialising the full [nk * nv, hidden] expanded tensor.
- Hard cap of MAX_LINK_CANDIDATES=128 keys and 128 values per page,
  selected by highest entity confidence. Guarantees worst-case linker
  matrix is 128x128 pairs regardless of epoch noisiness.
- gradient_checkpointing_enable() is NOT called: LayoutLMv3 does not
  support the global HuggingFace gradient-checkpointing API reliably.
  The chunked linker + candidate cap are sufficient to stay within 40GB.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv3Processor, LayoutLMv3Model
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import config

# Chunk size for biaffine linker pairwise scoring.
# Each chunk processes this many key rows at a time.
# Lower = less peak memory. 8 is safe for A100 40GB.
LINKER_CHUNK_SIZE = int(os.environ.get("LINKER_CHUNK_SIZE", "8"))

# Hard cap on key and value candidate tokens fed to the linker per page.
# Tokens are ranked by entity-class softmax confidence and the top-N kept.
# Worst-case linker matrix: MAX x MAX pairs. 128 x 128 = 16,384.
MAX_LINK_CANDIDATES = int(os.environ.get("LINKER_MAX_CANDIDATES", "128"))


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

    Note: gradient_checkpointing_enable() is NOT called here.
    LayoutLMv3 does not reliably support the global HuggingFace
    gradient-checkpointing API. Memory is managed via the chunked
    linker and the MAX_LINK_CANDIDATES cap instead.
    """

    def __init__(self, model_name="microsoft/layoutlmv3-base", freeze_base=False):
        super().__init__()

        self.model = LayoutLMv3Model.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

        if freeze_base:
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
            pooled_output:   [batch, hidden_size]  (same tensor, for API compat)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values
        )
        return outputs.last_hidden_state, outputs.last_hidden_state


class EntityClassifier(nn.Module):
    """Token classification head for Key/Value/Other detection."""

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

    Memory-efficient implementation:
    1. Hard cap: only top-MAX_LINK_CANDIDATES key/value tokens by entity
       confidence are passed to the linker (eliminates noisy-epoch OOM).
    2. Chunked scoring: biaffine scores are computed LINKER_CHUNK_SIZE
       key rows at a time (eliminates [nk*nv, hidden] expansion OOM).
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size

        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)

        self.biaffine = nn.Bilinear(hidden_size, hidden_size, 1)

        self.spatial_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )

        self.final_scorer = nn.Sequential(
            nn.Linear(1 + 32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def _compute_spatial_features_pair(self, key_box, val_boxes):
        """
        Compute spatial features for one key against all values.

        Args:
            key_box:   [4]      - single key bbox
            val_boxes: [nv, 4] - all value bboxes
        Returns:
            [nv, 8]
        """
        nv = val_boxes.size(0)
        key_box = key_box.float()
        val_boxes = val_boxes.float()

        key_cx = (key_box[0] + key_box[2]) / 2
        key_cy = (key_box[1] + key_box[3]) / 2
        val_cx = (val_boxes[:, 0] + val_boxes[:, 2]) / 2
        val_cy = (val_boxes[:, 1] + val_boxes[:, 3]) / 2

        dx = val_cx - key_cx
        dy = val_cy - key_cy
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        angle = torch.atan2(dy, dx)

        key_h = key_box[3] - key_box[1]
        val_h = val_boxes[:, 3] - val_boxes[:, 1]
        h_overlap = torch.clamp(
            torch.min(key_box[3].expand(nv), val_boxes[:, 3])
            - torch.max(key_box[1].expand(nv), val_boxes[:, 1]),
            min=0
        )
        h_align = h_overlap / (torch.min(key_h.expand(nv), val_h) + 1e-8)

        key_w = key_box[2] - key_box[0]
        val_w = val_boxes[:, 2] - val_boxes[:, 0]
        v_overlap = torch.clamp(
            torch.min(key_box[2].expand(nv), val_boxes[:, 2])
            - torch.max(key_box[0].expand(nv), val_boxes[:, 0]),
            min=0
        )
        v_align = v_overlap / (torch.min(key_w.expand(nv), val_w) + 1e-8)

        key_area = key_h * key_w
        val_area = val_h * val_w
        area_ratio = val_area / (key_area + 1e-8)
        aspect_ratio = (val_w / (val_h + 1e-8)) / (key_w / (key_h + 1e-8))

        return torch.stack(
            [dx, dy, dist, angle, h_align, v_align, area_ratio, aspect_ratio],
            dim=-1
        )  # [nv, 8]

    def _score_chunk(self, key_chunk, val_reps, key_boxes_chunk, val_boxes):
        """
        Score one chunk of keys against all values.

        Args:
            key_chunk:        [chunk, hidden]
            val_reps:         [nv, hidden]
            key_boxes_chunk:  [chunk, 4]
            val_boxes:        [nv, 4]
        Returns:
            [chunk, nv]
        """
        chunk_size = key_chunk.size(0)
        nv = val_reps.size(0)

        # [chunk * nv, hidden] -- bounded by LINKER_CHUNK_SIZE * MAX_LINK_CANDIDATES
        k_exp = key_chunk.unsqueeze(1).expand(chunk_size, nv, -1).reshape(-1, self.hidden_size)
        v_exp = val_reps.unsqueeze(0).expand(chunk_size, nv, -1).reshape(-1, self.hidden_size)
        biaffine_scores = self.biaffine(k_exp, v_exp).reshape(chunk_size, nv, 1)

        spatial_list = []
        for i in range(chunk_size):
            sf = self._compute_spatial_features_pair(key_boxes_chunk[i], val_boxes)
            spatial_list.append(sf)
        spatial_feats = torch.stack(spatial_list, dim=0)  # [chunk, nv, 8]
        spatial_scores = self.spatial_encoder(spatial_feats)  # [chunk, nv, 32]

        combined = torch.cat([biaffine_scores, spatial_scores], dim=-1)  # [chunk, nv, 33]
        return self.final_scorer(combined).squeeze(-1)  # [chunk, nv]

    def forward(self, sequence_output, entity_logits, bboxes, attention_mask):
        """
        Compute linking scores with hard candidate cap + chunked scoring.

        Args:
            sequence_output: [batch, seq_len, hidden_size]
            entity_logits:   [batch, seq_len, 3]
            bboxes:          [batch, seq_len, 4]
            attention_mask:  [batch, seq_len]

        Returns:
            link_scores:   list[Tensor|None]  shape per item: [nk, nv]
            key_indices:   list[Tensor]  (indices into original seq, capped)
            value_indices: list[Tensor]
        """
        batch_size = sequence_output.shape[0]
        # Softmax probabilities for confidence-based ranking
        entity_probs = torch.softmax(entity_logits, dim=-1)  # [batch, seq_len, 3]
        entity_preds = torch.argmax(entity_logits, dim=-1)   # [batch, seq_len]

        all_link_scores = []
        all_key_indices = []
        all_value_indices = []

        for b in range(batch_size):
            key_mask = (entity_preds[b] == 1) & (attention_mask[b] == 1)
            val_mask = (entity_preds[b] == 2) & (attention_mask[b] == 1)

            key_idx = key_mask.nonzero(as_tuple=True)[0]
            val_idx = val_mask.nonzero(as_tuple=True)[0]

            # ── Hard cap: keep top-MAX_LINK_CANDIDATES by entity confidence ──
            if len(key_idx) > MAX_LINK_CANDIDATES:
                key_conf = entity_probs[b][key_idx, 1]  # KEY class prob
                top_k = torch.topk(key_conf, MAX_LINK_CANDIDATES).indices
                key_idx = key_idx[top_k]

            if len(val_idx) > MAX_LINK_CANDIDATES:
                val_conf = entity_probs[b][val_idx, 2]  # VALUE class prob
                top_k = torch.topk(val_conf, MAX_LINK_CANDIDATES).indices
                val_idx = val_idx[top_k]

            if len(key_idx) == 0 or len(val_idx) == 0:
                all_link_scores.append(None)
                all_key_indices.append(key_idx)
                all_value_indices.append(val_idx)
                continue

            key_reps = self.key_projection(
                self.dropout(sequence_output[b][key_idx])
            )  # [nk, hidden]  nk <= MAX_LINK_CANDIDATES
            val_reps = self.value_projection(
                self.dropout(sequence_output[b][val_idx])
            )  # [nv, hidden]  nv <= MAX_LINK_CANDIDATES

            nk = len(key_idx)
            key_boxes = bboxes[b][key_idx]  # [nk, 4]
            val_boxes = bboxes[b][val_idx]  # [nv, 4]

            # Chunked scoring: LINKER_CHUNK_SIZE keys at a time
            score_rows = []
            for start in range(0, nk, LINKER_CHUNK_SIZE):
                end = min(start + LINKER_CHUNK_SIZE, nk)
                chunk_scores = self._score_chunk(
                    key_reps[start:end],
                    val_reps,
                    key_boxes[start:end],
                    val_boxes
                )  # [chunk, nv]
                score_rows.append(chunk_scores)

            link_scores = torch.cat(score_rows, dim=0)  # [nk, nv]

            all_link_scores.append(link_scores)
            all_key_indices.append(key_idx)
            all_value_indices.append(val_idx)

        return all_link_scores, all_key_indices, all_value_indices


class LayoutLMv3KVPModel(nn.Module):
    """
    Complete model for KVP extraction:
    1. LayoutLMv3 encoder
    2. Entity classifier (Key/Value/Other)
    3. Biaffine linker (chunked + capped) - optional for ablation
    """

    def __init__(
        self,
        layoutlmv3_model="microsoft/layoutlmv3-base",
        freeze_base=False,
        num_labels=3,
        dropout=0.1,
        use_linker=True
    ):
        super().__init__()

        self.encoder = LayoutLMv3Encoder(layoutlmv3_model, freeze_base)
        self.entity_classifier = EntityClassifier(
            self.encoder.hidden_size, num_labels, dropout
        )

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
        sequence_output, _ = self.encoder(
            input_ids, attention_mask, bbox, pixel_values
        )

        entity_logits = self.entity_classifier(sequence_output)

        text_seq_len = input_ids.shape[1]
        entity_logits = entity_logits[:, :text_seq_len, :]
        sequence_output_text = sequence_output[:, :text_seq_len, :]

        link_scores = key_indices = value_indices = None

        if self.use_linker and self.linker is not None:
            link_scores, key_indices, value_indices = self.linker(
                sequence_output_text, entity_logits, bbox, attention_mask
            )

        loss = entity_loss = link_loss = None

        if entity_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.reshape(-1) == 1
            active_logits = entity_logits.reshape(-1, self.num_labels)[active_loss]
            active_labels = entity_labels.reshape(-1)[active_loss]
            entity_loss = loss_fct(active_logits, active_labels)
            loss = entity_loss

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
                    gt_links = link_labels[b][k_idx][:, v_idx].float()
                    link_loss_total += F.binary_cross_entropy_with_logits(
                        scores, gt_links
                    )
                    link_count += 1
                if link_count > 0:
                    link_loss = link_loss_total / link_count
                    loss = entity_loss + link_loss

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
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, bbox, pixel_values)

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
    """Factory function to create and initialize the model."""
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

    print(f"\u2713 Created LayoutLMv3 KVP Model")
    print(f"  Encoder: {model_name}")
    print(f"  Freeze base: {freeze_base}")
    print(f"  Use linker: {use_linker} ({'Stage 4b - With Linker' if use_linker else 'Stage 4a - No Linker'})")
    print(f"  Gradient checkpointing: NOT used (LayoutLMv3 unsupported)")
    print(f"  Linker chunk size: {LINKER_CHUNK_SIZE}")
    print(f"  Max link candidates: {MAX_LINK_CANDIDATES}")
    print(f"  Device: {device}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model


if __name__ == "__main__":
    print("LayoutLMv3 KVP Model module ready")
    model = create_model(freeze_base=True)
    print("\n\u2713 Model initialized successfully!")
