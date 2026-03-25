# Implementation Summary

## What Has Been Done So Far

This project originally had a working Stage 2 pipeline and working Stage 3 CPU heuristics, but the Mistral baseline was not actually runnable in a faithful way.

The main work completed so far was to fix Stage 3 so that it matches IBM's KVP10k Mistral baseline much more closely and can run on the TinyGPU cluster.

## Stage Status

### Stage 2
- Completed earlier.
- Layout clustering was already run successfully.

### Stage 3 Heuristic Baselines
- Completed earlier.
- Existing heuristic baseline outputs were produced.
- Important caveat: the old Stage 3 ground truth generation is incomplete and should not be treated as the final reference for Mistral evaluation.

### Stage 3 Mistral Baseline
- Reworked substantially.
- This is the current active focus.

## What Was Fixed In Stage 3

### 1. Data Preparation Pipeline Was Added
New file: `prepare_data.py`

This script now does the missing preparation work that the original Mistral path depended on but did not implement correctly:
- downloads PDFs from the dataset URLs
- extracts word-level text with PyMuPDF
- fuses extracted words with annotation boxes
- creates KVP ground truth in IBM-style structure
- creates LMDX-style prompts for Mistral training
- saves per-page prepared JSON files

### 2. Mistral Baseline Was Rewritten
Updated file: `mistral_baseline.py`

The original implementation used the wrong assumptions about the dataset and wrong training settings. It has been rewritten to use:
- IBM-style prompt format
- IBM-style target format
- LoRA settings aligned with the IBM repo
- long-context training setup
- prepared JSON files instead of nonexistent raw dataset fields
- PEFT checkpoint loading for prediction

### 3. Evaluation Script Was Added
New file: `evaluate_mistral.py`

This evaluates Mistral predictions against the prepared Stage 3 ground truth.

### 4. Pipeline Integration Was Fixed
Updated file: `main.py`

The old Mistral integration incorrectly mixed train/test usage. The integration was corrected so the Mistral path points to prepared data instead of using the broken earlier assumptions.

### 5. Cluster Execution Workflow Was Fixed
Updated files:
- `logs/stage3_prepare_data.sbatch`
- `logs/stage3_mistral.sbatch`

Important cluster findings:
- TinyGPU compute nodes do not have internet access.
- Therefore PDF download and data preparation must run on the login node.
- GPU training must run after data and model files are already cached locally.
- The Mistral model was downloaded into the Hugging Face cache so GPU jobs can run offline.

## Current Real Workflow

Stage 3 now works as a 2-phase process.

### Phase 1: Prepare Data On Login Node
Run `prepare_data.py` on the login node because it needs internet access for PDF download.

Outputs:
- `data/prepared/train/*.json`
- `data/prepared/test/*.json`

### Phase 2: Train And Evaluate On A100
Run `logs/stage3_mistral.sbatch` after prepared data exists.

This job will:
- train the LoRA adapters
- generate predictions on the prepared test split
- evaluate the predictions

## Current Progress Snapshot

### Test Preparation
- Finished.
- Unique test pages processed: 1051
- Successfully prepared: 581
- Success rate: 55.3%
- Pages with valid KVPs among prepared test pages: 532

This success rate is limited mostly by broken or inaccessible source PDF URLs, not by the script itself.

### Train Preparation
- Completed.
- Unique train pages: 9656
- Successfully prepared: 5389
- Success rate: 55.8%
- Pages with valid KVPs: 4985

### Stage 3 Mistral Baseline: Training & Evaluation
- **Status: COMPLETE**
- Model: Mistral-7B-Instruct-v0.2 with LoRA (4-bit QLoRA)
- Training pages: 4,985 (with non-empty KVP supervision)
- Evaluation on: 581 test pages

**Results (Stage-0 Protocol: NED=0.2, IoU=0.3):**
- Text-only F1: **0.6711** (Precision 0.6037, Recall 0.7554)
- Text+Bbox F1: **0.5997** (Precision 0.5394, Recall 0.6750)

**Results (Default: NED=0.5, IoU=0.5):**
- Text-only F1: 0.6969 (Precision 0.6269, Recall 0.7845)
- Text+Bbox F1: 0.5523 (Precision 0.4968, Recall 0.6217)

**Error Analysis by Layout Density:**
- **Cluster 0 (Sparse layouts, n=116)**:
  - Correct: 14.4%, Hallucinated: 52.9%, Missed: 41.1%
  - Prediction count: 889 KVPs vs 224 GT (3.97× over-prediction)
- **Cluster 1 (Dense layouts, n=465)**:
  - Correct: 48.7%, Hallucinated: 40.5%, Missed: 31.2%
  - Prediction count: 7,852 KVPs vs 8,161 GT (0.96× under-prediction)
- **Key Finding**: Mistral achieves 3.4× higher correctness on dense layouts. Sparse layouts suffer from hallucination (model over-generates spurious KVP hypotheses). This reflects Mistral's pure language-model bias and lack of spatial reasoning.

**Evaluation artifacts:**
- `data/outputs/stage3_mistral/evaluation.json` (default thresholds)
- `data/outputs/stage3_mistral/evaluation_stage0_ned02_iou03.json` (Stage-0 protocol)
- `data/outputs/stage3_mistral/error_analysis/error_details.json` (per-document error categorization)
- `data/outputs/stage3_mistral/error_analysis/cluster_error_summary.json` (cluster-level aggregates)

## Why The Earlier Mistral Attempt Was Cancelled

The earlier job was cancelled because the previous code had several root problems:
- wrong dataset fields
- wrong prompt and response format
- wrong hyperparameters
- no real data preparation pipeline
- wrong checkpoint loading assumptions

Running that version would not have produced a meaningful reproduction.

## Files Most Relevant Right Now


## Compact File Map

### Core Pipeline
- `main.py`: pipeline entry and stage orchestration
- `config.py`: shared configuration
- `data_loader.py`: dataset loading helpers

### Stage 2
- `features.py`: layout features and clustering analysis
- `visualization.py`: plotting utilities

### Stage 3
- `baselines.py`: heuristic baselines
- `prepare_data.py`: prepared-data generation from PDFs and annotations
- `mistral_baseline.py`: LoRA training and prediction
- `evaluate_mistral.py`: evaluation against prepared test ground truth

### Stage 4
- `layoutlm_model.py`: LayoutLM-based model code
- `train_kvp.py`: Stage 4 training logic
- `kvp_dataset.py`: Stage 4 dataset adapter

### Outputs Used Now
- `data/prepared/train/`
- `data/prepared/test/`
- `data/outputs/stage3_mistral/`

## Immediate Next Steps

Stage 3 is complete. Comprehensive error analysis has been performed. Thesis Chapter 7 (Results) now includes:
- Dataset coverage explanation (581 / 1,051 unique pages)
- Three evaluation result tables (default thresholds, Stage-0 protocol, comparison)
- Paper baseline comparison (KVP10k Table 1: text-only F1=0.643, text+location F1=0.612)
- **Per-cluster error analysis** showing Mistral's 3.4× performance gap between dense and sparse layouts, with detailed hallucination breakdown
- Motivating narrative for Stage 4: spatial-aware models (LayoutLMv3) should address sparse-layout hallucination and spatial localization weaknesses

**Pre-Stage-4 Tasks** (in priority order):
1. ✅ **Task 1: Per-cluster error breakdown** — COMPLETE
   - Delivered per-cluster results and LaTeX table in Chapter 7.
2. ✅ **Task 2: Error taxonomy** — COMPLETE
   - Error categorization (correct/hallucinated/missed) embedded in per-cluster analysis.
3. ⏳ **Task 3: Write Chapter 2 (Literature Review)**
   - Scope: LayoutLM family (LayoutLMv1, v2, v3), generative IE, KVP10k paper, FUNSD/CORD datasets.
4. ⏳ **Task 4: Fix BibTeX references**
   - Add entries for: KVP10k, LayoutLMv3, Mistral-7B, LoRA/QLoRA, PyMuPDF.

**Stage 4 Roadmap**: LayoutLMv3 + Learned Linker for spatial-aware KVP extraction.

---

## Stage 4: Discriminative KVP Extraction — Implementation Complete

### Critical Blocker Identified and Fixed

**Root Problem**: The dataset adapter had a critical silent failure:
- Expected data format: `words[]`, `bboxes[]`, `image_path`
- Actual data format: `lmdx_text` (string with embedded coordinates), `image_width/height`, `gt_kvps` (dict)
- Result: Processor crashed when `images=None`, exception caught silently, returning empty examples with all-zero labels
- Outcome: Training appeared to work but no training signal (all entity and link labels were zeros)

### What Was Fixed In Stage 4

#### 1. Dataset Adapter Rewrite
File: `stage4_kvp_dataset.py` (330+ lines, complete rewrite)

**New Methods Implemented:**
- `_parse_lmdx_text()`: Parses lmdx_text format to extract words and pixel-level bounding boxes
- `_normalize_bboxes()`: Converts pixel coordinates to [0, 1000] scale (LayoutLMv3 standard)
- `_find_words_by_bbox_overlap()`: Matches words to KVP bounding boxes via spatial intersection (>50% overlap threshold)
- `_generate_labels()`: Creates entity labels (0=Other, 1=Key, 2=Value) and link_labels (adjacency matrix)
- `_align_labels_to_tokens()`: Maps word-level labels to subword token level for loss computation
- `_get_empty_example()`: Returns placeholder examples for edge cases

**Critical Runtime Fix:**
- Problem: `LayoutLMv3Processor` requires `images` parameter even with `apply_ocr=False`
- Solution: Create dummy white 224×224 images when real images unavailable
- Applied in: `__getitem__()` method (lines 110-113)

**API Improvements:**
- `create_stage4_dataloaders()` now returns dict: `{'train': DataLoader, 'val': DataLoader, 'test': DataLoader}`
- Updated calling code in `train_stage4a.py` (lines 343-351) and `train_stage4b.py` (lines 270-280)

#### 2. Configuration Updates
File: `stage4_kvp_dataset.py` (initialization)
- LayoutLMv3Processor configured with `apply_ocr=False` (avoids OCR on pre-extracted text)
- Max sequence length: 512 tokens
- PaddedBatchCollator handles variable-length sequences + link_labels matrix

#### 3. Supporting Files
Updated files:
- `train_stage4a.py`: Dict unpacking for dataloader access
- `train_stage4b.py`: Dict unpacking for dataloader access

### Data Processing Pipeline

**Input Format** (from KVP10k prepared data):
```json
{
  "hash_name": "00040fbaaab7ff89...",
  "image_width": 2550,
  "image_height": 3300,
  "lmdx_text": "Francis Marion University 33|6|69|8\nPosting Date: 11/02/2018 39|21|63|23\n...",
  "gt_kvps": {
    "kvps_list": [
      {"type": "kvp", "key": {"text": "Posting Date:", "bbox": [39, 21, 52, 23]}, 
       "value": {"text": "11/02/2018", "bbox": [53, 21, 63, 23]}},
      ...
    ]
  }
}
```

**Processing Steps:**
1. Parse lmdx_text → extract 19 words with pixel coordinates (example: [33, 6, 69, 8])
2. Normalize coordinates → [0, 1000] scale (example: [33/2550, 6/3300, 69/2550, 8/3300] × 1000)
3. Match words to KVP bboxes via overlap → generate entity labels
4. Create link_labels adjacency matrix for key-value relationships
5. Tokenize with LayoutLMv3Processor (subword tokenization)
6. Align word-level labels to token level
7. Batch with padding to 512 tokens

**Output Format** (per sample):
```python
{
  'input_ids': tensor([512])           # Tokenized text
  'attention_mask': tensor([512])      # Attention mask
  'bbox': tensor([512, 4])             # Normalized bboxes per token
  'entity_labels': tensor([512])       # 0/1/2 for Other/Key/Value
  'link_labels': tensor([512, 512])    # Key-value relationship matrix
  'pixel_values': tensor([3, 224, 224]) # Image tensor
  'hash_name': str                     # Document ID
}
```

### Verification Results ✅

**Multi-sample Testing** (Consistency Validation):

| Sample | Entity Labels | Link Labels | Status |
|--------|---------------|-------------|--------|
| 0 | 67 tokens (Key+Val) | 6 pairs | ✅ |
| 1 | 285 tokens | 23 pairs | ✅ |
| 2 | 0 tokens | 0 pairs | ⚠️ (edge case) |
| 3 | 42 tokens | 3 pairs | ✅ |
| 4 | 232 tokens | 11 pairs | ✅ |

**Key Validation Metrics:**
- ✅ Words parsed from lmdx_text correctly (19 words in test doc)
- ✅ Bbox normalization correct (pixel → [0,1000] scale verified)
- ✅ Bbox matching working (found 100% overlap for test KVP)
- ✅ Label generation functional (6 entity tokens found in direct test)
- ✅ Full pipeline end-to-end (no exceptions, proper tensor shapes)

**Syntax Verification:**
```bash
$ python3 -m py_compile stage4_kvp_dataset.py train_stage4a.py train_stage4b.py
✓ All files compile successfully
```

### Training Jobs Submitted

**Job IDs and Status:**

| Job | Type | Lambda | Status | Scheduled Start |
|-----|------|--------|--------|-----------------|
| 1557127 | Stage 4a | N/A | PENDING | 2026-03-25 18:21 |
| 1557128 | Stage 4b | 0.5 | PENDING | (after 4a) |
| 1557129 | Stage 4b | 1.0 | PENDING | (after 4a) |
| 1557130 | Stage 4b | 2.0 | PENDING | (after 4a) |

**Old duplicates cancelled**: 1557109-1557116 (avoided conflicting duplicate submissions)

### Expected Training Signals (When Jobs Run)

| Epoch | Metric | Expected Range | Interpretation |
|-------|--------|-----------------|-----------------|
| 1 | Entity loss | 1.0 - 1.1 → 0.6 - 0.7 | Should drop (random 3-class → learning) |
| 2 | Entity loss | < 1.0 | Decreasing loss confirms labels working |
| 3 | Entity loss | < 0.4 | Good convergence on entity task |

**Success Criteria:**
- If loss stays > 1.0 after epoch 2 → labels are still wrong
- If loss drops < 0.4 by epoch 3 → training working correctly ✅

### Files and Artifacts

**Core Implementation:**
- `code/script/stage4_kvp_dataset.py` (330 lines, complete)
- `code/script/train_stage4a.py` (Stage 4a training)
- `code/script/train_stage4b.py` (Stage 4b training with λ sweep)

**Documentation:**
- `STAGE4_DATASET_FIX.md` (high-level summary)
- `STAGE4_READY_FOR_TRAINING.md` (complete implementation overview)
- `TECHNICAL_LMDX_SPECIFICATION.md` (deep technical reference)
- `STAGE4_METHODOLOGY_DRAFT.tex` (LaTeX section for thesis Chapter 4)

**Data Sources:**
- Training: `data/prepared/train/` (5,389 prepared documents)
- Validation/Test: `data/prepared/val/`, `data/prepared/test/`

### Key Learnings

1. **LayoutLMv3Processor requires images even with apply_ocr=False**: Must provide dummy images when real images unavailable, not skip the parameter
2. **Silent failure mode in exception handling**: Processor crash caught by broad exception handler, returned empty example with all-zero labels
3. **Bbox-based matching > text-based matching**: Spatial overlap is more reliable for multi-word phrases in heterogeneous layouts
4. **Word-to-token alignment critical**: Subword tokenization requires explicit mapping from word-level labels to token level via word_ids()

### Immediate Next Steps

1. ⏳ **Monitor training logs** (when job starts running):
   - Check `logs/kvp_stage4a-1557127.out` for entity loss trajectory
   - Confirm entity loss drops below 1.0 by epoch 2
   
2. ⏳ **Later: Write thesis Chapter 7 (Results)**:
   - Placeholder structure with empty tables (ready now)
   - Fill numbers when training completes

3. ✅ **Now: Write thesis Chapter 4 (Methodology)**:
   - Stage 4 methodology section is complete (see `STAGE4_METHODOLOGY_DRAFT.tex`)
   - Ready to integrate into LaTeX thesis

---

## Stage 4 Status Summary

```
Stage 1: Document Classification ✅ Complete
  ↓
Stage 2: OCR Text Extraction + Layout Clustering ✅ Complete
  ↓
Stage 3: Text Encoding (Mistral Baseline) ✅ Complete
  ↓
Stage 4: KVP Extraction (LayoutLMv3 + Learned Linker)
  ├─ 4a: Entity Classification 🟢 Ready
  │        (Jobs queued: 1557127)
  ├─ 4b: Entity + Relation Extraction 🟢 Ready
  │        (Jobs queued: 1557128, 1557129, 1557130)
  └─ Status: Queued and waiting for GPU allocation
```
