# KVP10k: Key-Value Pair Extraction Pipeline

![Stages](https://img.shields.io/badge/Stages-0--4-blue) ![Dataset](https://img.shields.io/badge/Dataset-KVP10k%20ICDAR%202024-orange) ![Status](https://img.shields.io/badge/Stage%204b-In%20Progress-yellow)

Thesis project implementing a full key-value pair (KVP) extraction pipeline on the [KVP10k dataset](https://huggingface.co/datasets/ibm/KVP10k) (ICDAR 2024). The pipeline progresses through five stages: evaluation protocol → dataset ingestion → layout analysis → baselines → LayoutLMv3 fine-tuning with biaffine linking.


## Pipeline Overview

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | `config.py` + `main.py` | Evaluation protocol (NED, IoU thresholds) |
| 1 | `data_loader.py` | Dataset ingestion from HuggingFace |
| 2 | `features.py` + `visualization.py` | Layout clustering & data audit |
| 3 | `prepare_data.py` + `baselines.py` + `mistral_baseline.py` | Data preparation + baselines |
| 4a | `train_stage4a.py` | LayoutLMv3 entity classifier (ablation) |
| 4b | `train_stage4b.py` | LayoutLMv3 + biaffine linker, λ sweep |

---

## Stage 0: Evaluation Protocol

Defined in `config.py`, validated via `main.py`.

- **Text matching**: NED (Normalised Edit Distance) < 0.2
- **Location matching**: IoU > 0.3
- **Overall metric**: F1 score (balanced precision + recall)
- **Protocol**: IBM-compatible pair-level matching — a predicted KVP is a true positive if both its key text/bbox and value text/bbox match a ground-truth pair

---

## Stage 1: Dataset Ingestion

`data_loader.py` loads KVP10k from HuggingFace (`ibm/KVP10k`, cached locally at `hf_cache/`):

- 10,000 document pages — 9,405 train / 595 test (after filtering annotated pages)
- Each page: PDF URL + word annotations + KVP ground truth
- Annotation format: bounding polygon coordinates per word, structured KVP pairs

---

## Stage 2: Layout Clustering & Data Audit

`features.py` extracts 13 layout features per document page:

| Feature | Description |
|---------|-------------|
| `n_boxes` | Number of annotated bounding boxes |
| `total_area` | Total area covered by boxes |
| `mean_area`, `std_area` | Box area statistics |
| `mean_width`, `mean_height` | Average box dimensions |
| `mean_aspect_ratio` | Mean width/height ratio |
| `mean_cx`, `mean_cy` | Centroid of layout |
| `density` | Area coverage fraction |
| `v_spread`, `h_spread` | Vertical/horizontal extent |
| `mean_spacing` | Average inter-box spacing |

K-means clustering with optimal k selected by silhouette score. Results saved to `data/outputs/stage2/`.

---

## Stage 3: Data Preparation & Baselines

### Data Preparation (`prepare_data.py`)

Converts raw KVP10k pages into prepared JSON files for Stage 4 training:

1. Download PDF from HuggingFace image URL
2. Render page at 300 DPI with PyMuPDF (native text extraction, no Tesseract dependency)
3. Fuse extracted words with annotation bounding boxes (word-match threshold = 0.6)
4. Produce LMDX-format text and ground-truth KVP labels

Output format (`data/prepared/{train,test}/{hash_name}.json`):
```json
{
  "hash_name": "abc123...",
  "lmdx_text": "Company Name 100|200|150|220\nACME Corp 100|300|180|320\n...",
  "image_width": 2000,
  "image_height": 2800,
  "gt_kvps": {
    "kvps_list": [
      {"key": "Company Name", "key_bbox": [100, 200, 150, 220],
       "value": "ACME Corp",  "value_bbox": [100, 300, 180, 320]}
    ]
  }
}
```

Dataset sizes after preparation: **5,389 train** / **581 test**.

### Nearest-Neighbour Baseline (`baselines.py`)

Rule-based: pair each key with the spatially closest value (Euclidean centroid distance, max 0.3 normalised units). No learning required.

### Mistral-7B LoRA Baseline (`mistral_baseline.py`)

Faithful reproduction of the IBM ICDAR 2024 baseline:

| Parameter | Value |
|-----------|-------|
| Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Quantisation | 4-bit QLoRA (NF4) |
| LoRA rank | r=4, α=4, dropout=0.05 |
| Learning rate | 5×10⁻⁴ (AdamW) |
| Epochs | 8 |
| Batch / accum | 1 / 4 (effective 4) |
| Max length | 8192 tokens |
| Prompt format | LMDX (`<Document>…</Document><Task>…`) |

**Results on 581 test documents:**

| Metric | Text-only | Text + BBox |
|--------|-----------|-------------|
| Precision | 0.627 | 0.497 |
| Recall | 0.785 | 0.622 |
| **F1** | **0.697** | **0.552** |

---

## Stage 4: LayoutLMv3 Fine-tuning

### Architecture

```
Input: (input_ids, attention_mask, bbox, pixel_values)
  ↓
LayoutLMv3Encoder (microsoft/layoutlmv3-base, 125M params)
  ├─ Text + layout + visual features  
  └─ Output: [batch, 709, hidden] → truncate to [batch, 512, hidden]
  ↓
EntityClassifier
  └─ Output: [batch, 512, 3]  (0=Other, 1=Key, 2=Value)
  ↓
BiaffineLinker  ← Stage 4b only
  ├─ Filter tokens: key_mask & value_mask (bbox_valid applied)
  ├─ Spatial encoder: 9-dim spatial features → 64-dim MLP
  ├─ Dot-product scorer: key_proj · val_proj + spatial_bias
  └─ Output: [num_keys, num_values] logits
  ↓
Loss = CrossEntropy(entity) + λ × BCE(link, pos_weight=n_neg/n_pos)
```

The `pos_weight` in the link BCE loss corrects a ~450:1 negative-to-positive class imbalance in link labels (capped at 50 to prevent collapse).

### Stage 4a: Entity-Only Baseline

LayoutLMv3 encoder + entity classifier, no linker. Serves as an ablation.

```bash
sbatch logs/stage4a.sbatch
```

**Best validation entity F1: 0.8436**

### Stage 4b: Entity + Linker (λ Sweep)

Adds the BiaffineLinker with a configurable loss weight λ ∈ {0.5, 1.0, 2.0}.

```bash
sbatch logs/stage4b_lambda05.sbatch   # λ=0.5
sbatch logs/stage4b_lambda10.sbatch   # λ=1.0
sbatch logs/stage4b_lambda20.sbatch   # λ=2.0
```

**Hyperparameters (all Stage 4b runs):**

| Parameter | Value |
|-----------|-------|
| Encoder | `microsoft/layoutlmv3-base` |
| Precision | fp32 (bf16 disabled) |
| Learning rate | 2×10⁻⁵ |
| Batch size | 1 |
| Gradient accumulation | 8 (effective batch 8) |
| Epochs (max) | 20 |
| Early stopping patience | 3 epochs |
| GPU | A100 40GB |

**Stage 4b entity F1 results (λ sweep, best epoch):**

| Run | λ | Best Entity F1 | Best Epoch | Stopped |
|-----|---|----------------|------------|---------|
| Canary B | 1.0 | 0.8463 | 6 | 9 |
| λ=0.5 | 0.5 | **0.8488** | 6 | 9 |
| λ=1.0 | 1.0 | 0.8447 | 6 | 9 |
| λ=2.0 | 2.0 | 0.8480 | 9 | 9 |

**Link F1 evaluation (Canary B best_model, pos_weight pending):**

| Metric | Value |
|--------|-------|
| Entity F1 | 0.835 |
| Entity Precision | 0.816 |
| Entity Recall | 0.856 |
| Link F1 | 0.0 (linker predicted 0/3776 GT pairs — pos_weight fix in testing) |

### Evaluation

```bash
# Evaluate a trained checkpoint (entity F1 + link F1)
sbatch slurm/submit_eval.sh data/outputs/stage4b_canary_B

# Results saved to:
cat data/outputs/stage4b_canary_B/eval_results.json
```

`evaluate_stage4b.py` computes token-level entity F1 and word-level pair-matching link F1 (requires word-level F1 ≥ 0.5 for a match to handle subword tokenisation boundaries).

---

## Project Structure

```
kvp10k_thesis/
├── code/script/
│   ├── config.py                   # Stage 0: evaluation protocol constants
│   ├── main.py                     # Pipeline orchestration (stages 0–2)
│   ├── data_loader.py              # Stage 1: HuggingFace dataset loading
│   ├── features.py                 # Stage 2: layout feature extraction
│   ├── visualization.py            # Stage 2: cluster plots
│   ├── prepare_data.py             # Stage 3: PDF → prepared JSON
│   ├── baselines.py                # Stage 3: nearest-neighbour baseline
│   ├── mistral_baseline.py         # Stage 3: Mistral-7B LoRA baseline
│   ├── evaluate_mistral.py         # Stage 3: inference + evaluation
│   ├── analyze_results.py          # Stage 3: results analysis vs ground truth
│   ├── analyze_stage3_errors.py    # Stage 3: per-cluster error breakdown
│   ├── visualize_baseline.py       # Stage 3: baseline result visualisation
│   ├── kvp_dataset.py              # Stage 4 (prototype): original dataset class
│   ├── train_kvp.py                # Stage 4 (prototype): original training script
│   ├── layoutlm_model.py           # Stage 4: LayoutLMv3 + BiaffineLinker
│   ├── stage4_kvp_dataset.py       # Stage 4: data loading + label alignment
│   ├── train_stage4a.py            # Stage 4a: entity-only trainer
│   ├── train_stage4b.py            # Stage 4b: entity + linker trainer
│   ├── evaluate_stage4b.py         # Stage 4b: entity + link F1 evaluation
│   ├── metrics.py                  # F1 computation utilities
│   ├── utils.py                    # Shared utilities
│   └── KVP10k_poc.ipynb            # Exploratory notebook (proof of concept)
├── logs/
│   ├── gpu_check.sbatch            # SLURM: GPU availability check
│   ├── stage3_prepare_data.sbatch  # SLURM: Stage 3 data preparation
│   ├── stage3_mistral.sbatch       # SLURM: Stage 3 Mistral-7B fine-tuning
│   ├── stage4a.sbatch              # SLURM: Stage 4a train
│   ├── stage4b_lambda05.sbatch     # SLURM: Stage 4b λ=0.5
│   ├── stage4b_lambda10.sbatch     # SLURM: Stage 4b λ=1.0
│   └── stage4b_lambda20.sbatch     # SLURM: Stage 4b λ=2.0
├── slurm/
│   ├── submit_stage4b_canary_A.sh  # SLURM: canary A run (lr=5e-5)
│   ├── submit_stage4b_canary_B.sh  # SLURM: canary B run (lr=2e-5)
│   └── submit_eval.sh              # SLURM: evaluation job
├── data/
│   ├── prepared/{train,test}/      # 5389 train + 581 test prepared JSONs
│   └── outputs/                    # Training checkpoints + eval results
├── hf_cache/                       # Offline HuggingFace model/dataset cache
├── COMMANDS_CHEATSHEET.md          # Quick reference for common commands
└── env/kvp10k_env/                 # Python virtualenv
```

---

## Key Implementation Notes

**LayoutLMv3 visual patch truncation**: The encoder appends 197 ViT patch tokens; sequence output is truncated to `input_ids.shape[1]` (512) before the entity classifier and linker.

**BiaffineLinker NaN fixes**: Three patches were required to eliminate `link_loss=NaN`:
1. `bbox_valid` filter — excludes CLS/SEP/PAD tokens (zero bboxes) from key/value candidate sets
2. Aspect-ratio denominator guard — `+1e-8` to prevent divide-by-zero for zero-height boxes
3. Dot-score clamp ±20 — prevents MLP explosion in early training

**Link label quality**: 56.8% of raw link labels were junk (non-KVP annotation types). Fixed in `stage4_kvp_dataset.py` — link labels only generated for `type=="kvp"` entries with non-empty key/value text and valid bboxes.

**Class imbalance**: Link label matrices are ~450:1 negative-to-positive. `pos_weight = n_neg/n_pos` (capped at 50) passed to `F.binary_cross_entropy_with_logits` to prevent trivial all-zero predictions.

---

## References

- **KVP10k / IBM baseline**: [ICDAR 2024](https://huggingface.co/datasets/ibm/KVP10k)
- **LayoutLMv3**: Huang et al., 2022 — [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base)
- **Biaffine relation extraction**: Dozat & Manning, 2017
