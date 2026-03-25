# KVP10k LayoutLMv3 Stage 4: Entity + Linking

![Stage 4 Architecture](https://img.shields.io/badge/Stage-4-blue) ![Status](https://img.shields.io/badge/Status-Training-yellow) ![Fix](https://img.shields.io/badge/Latest-LayoutLMv3_Patch_Truncation-brightgreen)

This repository contains the **Stage 4** implementation of the KVP10k key-value pair extraction pipeline using LayoutLMv3 with an optional biaffine relation linker.

## 🎯 Overview

**Two-stage architecture:**
- **Stage 4a** (`train_stage4a.py`): Entity classification baseline
  - LayoutLMv3 encoder + Entity classifier
  - Detects Keys and Values token-level
  
- **Stage 4b** (`train_stage4b.py`): Entity + Relation linking with λ sweep
  - LayoutLMv3 encoder + Entity classifier + Biaffine linker
  - Learns to pair keys with values
  - Lambda sweep: λ ∈ {0.5, 1.0, 2.0} for linker loss weight

## 📋 Key Features

✅ **Gradient Accumulation**: Batch size 4 × accumulation 8 = effective batch 32  
✅ **LayoutLMv3 Patch Token Handling**: Correct truncation of vision patch tokens  
✅ **Lambda Weight Application**: Configurable link loss weighting  
✅ **Word-to-Token Alignment**: Proper label mapping for subword tokenization  
✅ **Unified F1 Metrics**: Key/Value-only F1 computation (no weighting)  

## 🔧 Model Architecture

```
Input: [batch, seq_len, 4]  (text_ids, attention_mask, bbox, pixel_values)
  ↓
LayoutLMv3Encoder
  ├─ Text embedding + layout + visual features
  └─ Output: [batch, 512+197_patches, hidden] → truncate to [batch, 512, hidden]
  ↓
EntityClassifier
  └─ Output: [batch, 512, 3]  (Other/Key/Value)
  ↓
BiaffineLinker (optional for Stage 4b)
  ├─ Extract key/value positions
  ├─ Compute spatial + semantic scores
  └─ Output: [batch, num_keys, num_values]
  ↓
Loss = entity_loss + λ * link_loss
```

## 🐛 Important Fix: LayoutLMv3 Visual Patch Tokens

**Problem**: LayoutLMv3 encoder appends ~197 visual patch tokens from its ViT backbone:
- `sequence_output` shape: `[batch, 709, hidden]` ← 512 text + 197 visual
- `attention_mask` shape: `[batch, 512]` ← text tokens only
- Causes shape mismatch in loss computation

**Solution**: Truncate sequence_output to text-only before loss/linker:
```python
text_seq_len = input_ids.shape[1]  # 512
entity_logits = entity_logits[:, :text_seq_len, :]
sequence_output = sequence_output[:, :text_seq_len, :]
```

This is the standard approach in official LayoutLMv3 fine-tuning code.

## 📊 Data Format

**Input**: Prepared JSON files from Stage 3
```json
{
  "hash_name": "doc_12345",
  "lmdx_text": "Company Name Co Ltd 100|200|150|220\nInvoice 200|250|250|270\n...",
  "image_width": 2000,
  "image_height": 2800,
  "gt_kvps": {
    "kvps_list": [
      {
        "key": "Company Name",
        "key_bbox": [100, 200, 150, 220],
        "value": "ACME Corp",
        "value_bbox": [100, 300, 180, 320]
      }
    ]
  }
}
```

**Label alignment**:
1. Words → token-level via `word_ids()` mapping
2. Entity labels: 0=Other, 1=Key, 2=Value
3. Link labels: [seq_len, seq_len] binary adjacency matrix

## 🚀 Training

### Stage 4a (Entity Only)
```bash
python train_stage4a.py \
    --data_dir data/prepared \
    --output_dir data/outputs/stage4a \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_epochs 10 \
    --early_stopping_patience 3 \
    --val_fraction 0.1 \
    --include_images
```

### Stage 4b (Entity + Linker)
```bash
# Lambda 0.5
python train_stage4b.py \
    --data_dir data/prepared \
    --linker_loss_weight 0.5 \
    --... (other args)

# Lambda 1.0
python train_stage4b.py \
    --data_dir data/prepared \
    --linker_loss_weight 1.0 \
    --... (other args)

# Lambda 2.0
python train_stage4b.py \
    --data_dir data/prepared \
    --linker_loss_weight 2.0 \
    --... (other args)
```

## 📈 Training Status (Latest)

| Job | Stage | Lambda | Status | Started |
|-----|-------|--------|--------|---------|
| 1557830 | 4a | — | PENDING | Mar 25, 2026 |
| 1557831 | 4b | 0.5 | PENDING | Mar 25, 2026 |
| 1557832 | 4b | 1.0 | PENDING | Mar 25, 2026 |
| 1557833 | 4b | 2.0 | PENDING | Mar 25, 2026 |

**Hardware**: A100 GPU (40GB VRAM)  
**Batch Config**: Batch=4, Accumulation=8 → Effective batch=32  
**Expected Duration**: ~12 hours (10 epochs × ~1.2hr epoch-1)

## 🔗 Previous Fixes (Stages 4a/4b)

### Bug #1: Loss Key Mismatch
- **Issue**: `train_stage4b.py` accessed non-existent `outputs['entity_loss']`
- **Fix**: Model now returns separate `entity_loss` and `link_loss`

### Bug #2: Gradient Accumulation Ignored
- **Issue**: Argument parsed but never applied in training loop
- **Fix**: Implemented loss scaling (÷ accumulation_steps) + conditional optimizer.step()

### Bug #3: Link Label Alignment
- **Issue**: Word-level supervision vs token-level indices mismatch
- **Fix**: Added `_align_link_labels_to_tokens()` using `word_ids()` mapping

### Bug #4: F1 Metric Inconsistency
- **Issue**: Stage 4a manual F1 vs Stage 4b sklearn weighted F1
- **Fix**: Unified both to manual Key/Value-only F1

### Bug #5: Lambda Weight Ignored
- **Issue**: Model hardcoded `loss = entity_loss + link_loss` (λ=1.0)
- **Fix**: Trainer applies λ explicitly: `loss = entity_loss + λ * link_loss`

### Bug #6 (Latest): Visual Patch Token Offset
- **Issue**: LayoutLMv3 visual tokens cause sequence length mismatch
- **Fix**: Truncate to text_seq_len after entity_classifier

## 📂 Project Structure

```
code/script/
├── layoutlm_model.py           # Core model (encoder + classifier + linker)
├── stage4_kvp_dataset.py       # Data loading + label alignment
├── train_stage4a.py            # Stage 4a trainer (entity only)
├── train_stage4b.py            # Stage 4b trainer (entity + linker)
├── config.py                   # Configuration
├── metrics.py                  # F1 computation
└── utils.py                    # Utilities

logs/
├── stage4a.sbatch              # SLURM script for Stage 4a
├── stage4b_lambda05.sbatch     # SLURM script for Stage 4b λ=0.5
├── stage4b_lambda10.sbatch     # SLURM script for Stage 4b λ=1.0
└── stage4b_lambda20.sbatch     # SLURM script for Stage 4b λ=2.0

COMMANDS_CHEATSHEET.md          # Quick reference for monitoring/debugging
```

## 💾 Model Checkpoints

Saved to `data/outputs/{stage4a,stage4b_lambda05,stage4b_lambda10,stage4b_lambda20}/`:
```
├── checkpoint-1/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── training_args.bin
├── best_model/
│   └── pytorch_model.bin
├── training_history.json       # Loss curves + F1 scores
└── evaluation.json             # Test set metrics
```

## 🎓 Notes on Gradient Accumulation

When `len(train_loader) % gradient_accumulation_steps ≠ 0`, final batches are dropped:
- Train loader: ~1348 batches
- Accumulation: 8
- Dropped per epoch: 4 batches (~0.3% of data)
- **Expected behavior**, not a bug — standard in most frameworks

Loss curves may show minor irregularities at epoch boundaries due to this.

## 📝 References

- **LayoutLMv3**: [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base)
- **Biaffine Attention**: Zhang et al., 2016 ([arxiv:1611.02902](https://arxiv.org/abs/1611.02902))
- **KVP10k Dataset**: ICDAR 2024

## 📧 Contact

Questions about this code? Ask Claude Sonnet 4.6 with a link to this repo!

---

**Latest Commit**: LayoutLMv3 patch token truncation fix (Jobs 1557830-1557833)  
**Date**: March 25, 2026
