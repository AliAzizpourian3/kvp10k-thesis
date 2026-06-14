"""
Stage 4b V2: LayoutLMv3 + Span-Level Biaffine Linker for KVP Extraction.

Key change from V1 (layoutlm_model.py):
  V1 scores individual key TOKENS against individual value TOKENS, producing a
  very large sparse matrix (~450:1 neg:pos ratio) that the linker struggles to
  learn from.

  V2 first groups contiguous same-label tokens into SPANS (e.g. "Invoice Number"
  = 3 key tokens → 1 key span), mean-pools their hidden states, and scores
  SPAN pairs with the same biaffine + spatial architecture. This reduces the
  matrix from ~100×100 tokens to ~10×10 spans, dramatically improving the
  positive ratio and matching the approach used by SPADE (Hwang et al., 2021)
  and BROS (Hong et al., 2022).

Architecture:
  1. LayoutLMv3 encoder (unchanged)
  2. Entity classifier (unchanged)
  3. Span grouper: contiguous same-label tokens → spans
  4. Span-level BiaffineLinker: biaffine + spatial scorer on span representations

Everything else (encoder, entity classifier, create_model, loss) is identical
to V1 so that training scripts need minimal changes.
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
LINKER_CHUNK_SIZE = int(os.environ.get("LINKER_CHUNK_SIZE", "8"))

# Hard cap on key and value spans fed to the linker per page.
MAX_LINK_SPANS = int(os.environ.get("LINKER_MAX_SPANS", "64"))

# Clamp link logits to this range before BCE loss.
LINK_LOGIT_CLAMP = 10.0


@dataclass
class KVPExample:
    """Single training example for KVP extraction."""
    image_path: str
    words: List[str]
    bboxes: List[List[int]]
    labels: List[int]
    links: List[Tuple[int, int]]


class LayoutLMv3Encoder(nn.Module):
    """LayoutLMv3 encoder for document understanding."""

    def __init__(self, model_name="microsoft/layoutlmv3-base", freeze_base=False):
        super().__init__()
        self.model = LayoutLMv3Model.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, bbox, pixel_values=None):
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
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


# ---------------------------------------------------------------------------
# Span grouping utilities
# ---------------------------------------------------------------------------

def group_contiguous_spans(entity_preds, attention_mask, bbox, label_id):
    """Group contiguous tokens with the same entity label into spans.

    Args:
        entity_preds: [seq_len] predicted entity labels (0/1/2)
        attention_mask: [seq_len] attention mask
        bbox: [seq_len, 4] bounding boxes
        label_id: which label to group (1=Key, 2=Value)

    Returns:
        spans: list of (start_idx, end_idx) token index ranges (inclusive)
    """
    spans = []
    seq_len = entity_preds.size(0)
    in_span = False
    start = 0

    # Exclude only truly degenerate bboxes: CLS/SEP/PAD have [0,0,0,0].
    # Previous bug: (x1 < x2) & (y1 < y2) also killed valid tokens whose
    # bboxes had x1==x2 or y1==y2 after LayoutLMv3 normalization, removing
    # ~50% of predicted key/value tokens.
    bbox_valid = ~((bbox[:, 0] == 0) & (bbox[:, 1] == 0) &
                   (bbox[:, 2] == 0) & (bbox[:, 3] == 0))

    for i in range(seq_len):
        is_target = (entity_preds[i] == label_id) and \
                    (attention_mask[i] == 1) and \
                    bbox_valid[i]
        if is_target:
            if not in_span:
                start = i
                in_span = True
        else:
            if in_span:
                spans.append((start, i - 1))
                in_span = False
    if in_span:
        spans.append((start, seq_len - 1))

    return spans


def compute_span_representations(sequence_output, spans, bbox):
    """Compute mean-pooled representation and union bbox for each span.

    Args:
        sequence_output: [seq_len, hidden] encoder hidden states
        spans: list of (start, end) inclusive index tuples
        bbox: [seq_len, 4] bounding boxes

    Returns:
        span_reps: [num_spans, hidden] mean-pooled representations
        span_bboxes: [num_spans, 4] union bounding boxes
        span_token_ranges: list of (start, end) for mapping back
    """
    if not spans:
        hidden = sequence_output.size(-1)
        device = sequence_output.device
        return (torch.zeros(0, hidden, device=device),
                torch.zeros(0, 4, device=device),
                [])

    reps = []
    bboxes_out = []
    for start, end in spans:
        # Mean-pool hidden states across span tokens
        span_hidden = sequence_output[start:end + 1]  # [span_len, hidden]
        reps.append(span_hidden.mean(dim=0))

        # Union bounding box
        span_boxes = bbox[start:end + 1].float()
        x1 = span_boxes[:, 0].min()
        y1 = span_boxes[:, 1].min()
        x2 = span_boxes[:, 2].max()
        y2 = span_boxes[:, 3].max()
        bboxes_out.append(torch.stack([x1, y1, x2, y2]))

    span_reps = torch.stack(reps, dim=0)        # [num_spans, hidden]
    span_bboxes = torch.stack(bboxes_out, dim=0)  # [num_spans, 4]
    return span_reps, span_bboxes, spans


# ---------------------------------------------------------------------------
# Span-level BiaffineLinker
# ---------------------------------------------------------------------------

class SpanBiaffineLinker(nn.Module):
    """
    Span-level linker: scores (key_span, value_span) pairs using
    biaffine attention + spatial features.

    Identical scorer architecture to V1 BiaffineLinker, but operates on
    span representations (mean-pooled contiguous tokens) instead of
    individual tokens. This reduces the scoring matrix from ~100x100 to ~10x10.
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.scale = hidden_size ** 0.5

        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)

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
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.zeros_(self.key_projection.bias)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.zeros_(self.value_projection.bias)
        for module in list(self.spatial_encoder) + list(self.final_scorer):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _compute_spatial_features_pair(self, key_box, val_boxes):
        """Compute 8 spatial features between one key bbox and multiple value bboxes."""
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
        area_ratio = torch.clamp(area_ratio, max=50.0)
        key_aspect = key_w / (key_h + 1e-8)
        val_aspect = val_w / (val_h + 1e-8)
        aspect_ratio = val_aspect / (key_aspect + 1e-8)

        return torch.stack(
            [dx, dy, dist, angle, h_align, v_align, area_ratio, aspect_ratio],
            dim=-1
        )  # [nv, 8]

    def _score_chunk(self, key_chunk, val_reps, key_boxes_chunk, val_boxes):
        """Score a chunk of key spans against all value spans."""
        chunk_size = key_chunk.size(0)
        nv = val_reps.size(0)

        k_exp = key_chunk.unsqueeze(1).expand(chunk_size, nv, -1)
        v_exp = val_reps.unsqueeze(0).expand(chunk_size, nv, -1)
        dot_scores = (k_exp * v_exp).sum(dim=-1, keepdim=True) / self.scale
        dot_scores = torch.clamp(dot_scores, min=-20.0, max=20.0)

        spatial_list = []
        for i in range(chunk_size):
            sf = self._compute_spatial_features_pair(key_boxes_chunk[i], val_boxes)
            spatial_list.append(sf)
        spatial_feats = torch.stack(spatial_list, dim=0)
        spatial_scores = self.spatial_encoder(spatial_feats)

        combined = torch.cat([dot_scores, spatial_scores], dim=-1)
        return self.final_scorer(combined).squeeze(-1)

    def forward(self, sequence_output, entity_logits, bboxes, attention_mask,
                entity_labels=None):
        """Score span pairs for a batch.

        Args:
            entity_labels: optional [batch, seq_len] GT entity labels.
                When provided (training), spans are derived from GT labels
                instead of predicted labels (teacher forcing).

        Returns:
            all_link_scores: list of [num_key_spans, num_val_spans] or None
            all_key_spans: list of [(start, end), ...] span token ranges
            all_value_spans: list of [(start, end), ...] span token ranges
        """
        batch_size = sequence_output.shape[0]
        entity_probs = torch.softmax(entity_logits, dim=-1)

        # Teacher forcing: use GT labels during training, predicted during inference
        if entity_labels is not None:
            entity_preds = entity_labels
        else:
            entity_preds = torch.argmax(entity_logits, dim=-1)

        all_link_scores = []
        all_key_spans = []
        all_value_spans = []

        for b in range(batch_size):
            # Group contiguous tokens into spans
            key_spans = group_contiguous_spans(
                entity_preds[b], attention_mask[b], bboxes[b], label_id=1
            )
            val_spans = group_contiguous_spans(
                entity_preds[b], attention_mask[b], bboxes[b], label_id=2
            )

            # Cap number of spans
            if len(key_spans) > MAX_LINK_SPANS:
                # Keep spans with highest mean entity confidence
                key_confs = []
                for s, e in key_spans:
                    key_confs.append(entity_probs[b, s:e + 1, 1].mean().item())
                top_indices = sorted(range(len(key_spans)),
                                     key=lambda i: key_confs[i], reverse=True)[:MAX_LINK_SPANS]
                key_spans = [key_spans[i] for i in sorted(top_indices)]

            if len(val_spans) > MAX_LINK_SPANS:
                val_confs = []
                for s, e in val_spans:
                    val_confs.append(entity_probs[b, s:e + 1, 2].mean().item())
                top_indices = sorted(range(len(val_spans)),
                                     key=lambda i: val_confs[i], reverse=True)[:MAX_LINK_SPANS]
                val_spans = [val_spans[i] for i in sorted(top_indices)]

            if len(key_spans) == 0 or len(val_spans) == 0:
                all_link_scores.append(None)
                all_key_spans.append(key_spans)
                all_value_spans.append(val_spans)
                continue

            # Compute span representations
            key_reps, key_boxes, _ = compute_span_representations(
                sequence_output[b], key_spans, bboxes[b]
            )
            val_reps, val_boxes, _ = compute_span_representations(
                sequence_output[b], val_spans, bboxes[b]
            )

            # Project
            key_reps = self.key_projection(self.dropout(key_reps))
            val_reps = self.value_projection(self.dropout(val_reps))

            # Chunked scoring
            nk = len(key_spans)
            score_rows = []
            for start in range(0, nk, LINKER_CHUNK_SIZE):
                end = min(start + LINKER_CHUNK_SIZE, nk)
                chunk_scores = self._score_chunk(
                    key_reps[start:end], val_reps,
                    key_boxes[start:end], val_boxes
                )
                score_rows.append(chunk_scores)

            link_scores = torch.cat(score_rows, dim=0)  # [nk_spans, nv_spans]

            all_link_scores.append(link_scores)
            all_key_spans.append(key_spans)
            all_value_spans.append(val_spans)

        return all_link_scores, all_key_spans, all_value_spans


def collapse_link_labels_to_spans(link_labels, key_spans, val_spans):
    """Collapse token-level link labels to span-level.

    A span pair (key_span_i, val_span_j) is positive if ANY token pair
    (k_token, v_token) within those spans has link_labels[k, v] == 1.

    Args:
        link_labels: [seq_len, seq_len] token-level binary adjacency matrix
        key_spans: list of (start, end) inclusive token ranges for key spans
        val_spans: list of (start, end) inclusive token ranges for value spans

    Returns:
        span_labels: [num_key_spans, num_val_spans] binary tensor
    """
    nk = len(key_spans)
    nv = len(val_spans)
    span_labels = torch.zeros(nk, nv, dtype=torch.float32,
                              device=link_labels.device)

    for i, (ks, ke) in enumerate(key_spans):
        for j, (vs, ve) in enumerate(val_spans):
            # Check if any token pair in the span cross-product is positive
            block = link_labels[ks:ke + 1, vs:ve + 1]
            if block.max() > 0.5:
                span_labels[i, j] = 1.0

    return span_labels


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LayoutLMv3KVPModelV2(nn.Module):
    """
    V2 KVP extraction model with span-level linking.

    Identical to V1 except the linker operates on spans instead of tokens.
    All weights are compatible with V1 checkpoints for the encoder and
    entity classifier (the linker weights have the same shapes since the
    hidden_size and spatial feature dimensions are unchanged).
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
            self.linker = SpanBiaffineLinker(self.encoder.hidden_size, dropout)
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

        link_scores = key_spans = value_spans = None

        if self.use_linker and self.linker is not None:
            # Teacher forcing: pass GT entity labels during training
            # so the linker learns from correct key/value positions.
            # At inference (entity_labels=None), uses predicted entities.
            link_scores, key_spans, value_spans = self.linker(
                sequence_output_text, entity_logits, bbox, attention_mask,
                entity_labels=entity_labels
            )

        loss = entity_loss = link_loss = None

        if entity_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.reshape(-1) == 1
            active_logits = entity_logits.reshape(-1, self.num_labels)[active_loss]
            active_labels = entity_labels.reshape(-1)[active_loss]
            entity_loss = loss_fct(active_logits, active_labels)
            loss = entity_loss

        # Link loss: computed regardless of whether entity_labels is provided.
        # This supports V3 training (linker-only, using predicted entities).
        if self.use_linker and link_labels is not None and link_scores is not None:
            link_loss_total = 0.0
            link_count = 0
            for b, scores in enumerate(link_scores):
                if scores is None:
                    continue
                k_spans = key_spans[b]
                v_spans = value_spans[b]
                if len(k_spans) == 0 or len(v_spans) == 0:
                    continue

                # Collapse token-level GT to span-level GT
                span_gt = collapse_link_labels_to_spans(
                    link_labels[b], k_spans, v_spans
                )

                # Flatten for BCE
                scores_flat = scores.reshape(-1)
                gt_flat = span_gt.reshape(-1).to(scores_flat.device)

                # Mask only valid binary labels
                valid_mask = (gt_flat >= 0.0) & (gt_flat <= 1.0)
                if not valid_mask.any():
                    continue

                scores_valid = scores_flat[valid_mask]
                gt_valid = gt_flat[valid_mask]

                scores_clamped = torch.clamp(
                    scores_valid, min=-LINK_LOGIT_CLAMP, max=LINK_LOGIT_CLAMP
                )

                # Class-imbalance fix
                n_pos = gt_valid.sum().clamp(min=1.0)
                n_neg = (1.0 - gt_valid).sum().clamp(min=1.0)
                pw = torch.clamp(n_neg / n_pos, max=50.0)
                link_loss_total += F.binary_cross_entropy_with_logits(
                    scores_clamped, gt_valid,
                    pos_weight=pw.to(scores_clamped.device)
                )
                link_count += 1

            if link_count > 0:
                link_loss = link_loss_total / link_count
                if loss is not None:
                    loss = loss + link_loss
                else:
                    loss = link_loss

        return {
            'entity_logits': entity_logits,
            'link_scores': link_scores,
            'key_indices': key_spans,      # V2: spans, not token indices
            'value_indices': value_spans,   # V2: spans, not token indices
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
        word_ids_list=None,
        score_threshold=0.5
    ):
        """Predict key-value pairs at span level.

        Args:
            words: list of word strings per batch item
            word_ids_list: list of word_ids (from tokenizer) per batch item,
                           mapping token indices to word indices
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, bbox, pixel_values)

            entity_logits = outputs['entity_logits']
            link_scores = outputs['link_scores']
            key_spans = outputs.get('key_indices')
            value_spans = outputs.get('value_indices')

            predictions = []
            batch_size = input_ids.size(0)

            for b in range(batch_size):
                batch_preds = []

                if (self.use_linker and link_scores is not None
                        and link_scores[b] is not None):
                    scores = link_scores[b]  # [nk_spans, nv_spans]
                    k_spans = key_spans[b]
                    v_spans = value_spans[b]

                    # For each key span, find best value span
                    best_val_pos = torch.argmax(scores, dim=1)
                    best_scores = torch.sigmoid(
                        scores[range(len(k_spans)), best_val_pos]
                    )

                    for i, (ks, ke) in enumerate(k_spans):
                        if best_scores[i].item() < score_threshold:
                            continue
                        vs, ve = v_spans[best_val_pos[i]]

                        # Reconstruct span text from token→word mapping
                        key_text = self._span_to_text(ks, ke, words, word_ids_list, b)
                        val_text = self._span_to_text(vs, ve, words, word_ids_list, b)

                        key_bbox = bbox[b][ks:ke + 1]
                        val_bbox = bbox[b][vs:ve + 1]

                        batch_preds.append({
                            "key": key_text,
                            "value": val_text,
                            "key_bbox": [
                                key_bbox[:, 0].min().item(),
                                key_bbox[:, 1].min().item(),
                                key_bbox[:, 2].max().item(),
                                key_bbox[:, 3].max().item()
                            ],
                            "value_bbox": [
                                val_bbox[:, 0].min().item(),
                                val_bbox[:, 1].min().item(),
                                val_bbox[:, 2].max().item(),
                                val_bbox[:, 3].max().item()
                            ],
                            "link_score": best_scores[i].item()
                        })
                else:
                    entity_preds_b = torch.argmax(entity_logits[b], dim=-1)
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

    def _span_to_text(self, start, end, words, word_ids_list, batch_idx):
        """Convert a token span back to text using word_ids mapping."""
        if words is None or word_ids_list is None:
            return f"span[{start}:{end}]"

        b_words = words[batch_idx] if isinstance(words[0], list) else words
        b_word_ids = word_ids_list[batch_idx] if word_ids_list else None

        if b_word_ids is None:
            return f"span[{start}:{end}]"

        # Collect unique word indices spanned by these tokens
        seen_word_ids = []
        for t in range(start, end + 1):
            if t < len(b_word_ids) and b_word_ids[t] is not None:
                wid = b_word_ids[t]
                if wid not in seen_word_ids:
                    seen_word_ids.append(wid)

        text_parts = []
        for wid in seen_word_ids:
            if wid < len(b_words):
                text_parts.append(b_words[wid])

        return " ".join(text_parts) if text_parts else f"span[{start}:{end}]"


def create_model(
    model_name="microsoft/layoutlmv3-base",
    freeze_base=False,
    use_linker=True,
    device=None
):
    """Factory function to create and initialize the V2 model."""
    if device is None:
        device = config.DEVICE

    model = LayoutLMv3KVPModelV2(
        layoutlmv3_model=model_name,
        freeze_base=freeze_base,
        num_labels=3,
        dropout=0.1,
        use_linker=use_linker
    )

    model = model.to(device)

    print(f"\u2713 Created LayoutLMv3 KVP Model V2 (span-level linker)")
    print(f"  Encoder: {model_name}")
    print(f"  Freeze base: {freeze_base}")
    print(f"  Use linker: {use_linker} ({'Stage 4b V2 - Span Linker' if use_linker else 'Stage 4a - No Linker'})")
    print(f"  Linker type: span-level biaffine (mean-pool spans)")
    print(f"  Link logit clamp: +/-{LINK_LOGIT_CLAMP}")
    print(f"  Linker chunk size: {LINKER_CHUNK_SIZE}")
    print(f"  Max link spans: {MAX_LINK_SPANS}")
    print(f"  Device: {device}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model


if __name__ == "__main__":
    print("LayoutLMv3 KVP Model V2 (span-level linker) module ready")
    model = create_model(freeze_base=True)
    print("\n\u2713 Model V2 initialized successfully!")
