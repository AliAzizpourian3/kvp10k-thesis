# KVP10k Thesis — Single Source-of-Truth Fact Sheet

Purpose: the one authoritative reference for every number, name, and setting used
while writing/polishing the thesis. Prefer this file over prose in any chapter,
and over the repo `README.md` / `measurements.json`, which contain **obsolete
pooled scores** and must not be cited.

Legend: ✅ = verified against a code/data artifact in this repo · ⚠️ = prose-only,
estimated, or not cleanly recoverable (state cautiously in the thesis).

Verification date: 2026-08-30.

---

## 1. Dataset sizes and usable pages

Source: `LaTeX_Thesis/DATA_PREPARATION_NOTE.md`, `prepare_data.py`, prepared JSONs.

| Quantity | Value | Status |
|---|---|---|
| KVP10k total unique pages | 10,707 (train 9,656 + test 1,051) | ✅ |
| Raw rows (duplicated annotator copies) | train 48,280 · test 5,255 | ✅ |
| Prepared/usable **train** pages | 5,389 of 9,656 (55.8%) | ✅ |
| — train pages with **zero** KVPs after prep | 404 (⇒ 4,985 with KVPs) | ✅ |
| Prepared/usable **test** pages | 581 of 1,051 (55.3%) | ✅ |
| — test pages with **non-empty** KVP supervision | 532 | ✅ |

Key wording rule: raw "rows" are **duplicated annotations of the same page**, not
unique documents. Report unique pages, not rows. Main loss source = broken /
inaccessible source PDFs, not random filtering. ✅

**Stage 2 page-level annotation-geometry clustering:** raw training rows are grouped by `hash_name`,
and the annotator copy with the most usable polygon boxes is retained per page.
Of 9,656 unique training pages, 9,124 have coordinate-bearing copies and 532 are
excluded as geometry unavailable. Clustering uses 12 distinct features: the
historical `density` column was removed because it exactly duplicated
`total_area`. Features are standardized with `StandardScaler`; K-means uses
seed 42 and `n_init=10`. A sweep over $k=2,\ldots,10$ selects $k=2$ by the
prespecified silhouette criterion (0.22214; modest separation). Cluster 0 has
5,617 pages (higher box count, broad spread); Cluster 1 has 3,507 pages (lower
box count, larger boxes, compact spread). PCA PC1/PC2 explain 30.15%/22.68%
(52.83% total). These are annotation-geometry regimes, not document genres or
OCR-density classes. ✅

**Corrected test assignment:** apply the frozen training scaler/K-means model
after the same richest-copy selection. All 1,051 unique test pages: Cluster 0 =
588, Cluster 1 = 396, geometry unavailable = 67. Prepared 581-page evaluation
subset: 319 / 213 / 49. Unavailable pages are not projected from zero vectors. ✅

**Chapter 4 spatial audit:** the notebook selected the first 200 training rows
with non-empty annotation lists, representing 70 unique page hashes and 69
coordinate-bearing rows. The subset contains 3,595 annotations, of which 755
(21.0%) carry linking values. Recomputed linked-center distances are mean
0.1204, median 0.0951, and 90th percentile 0.2393 in normalized coordinates. ✅

**Final discriminative train/validation manifests:** the seed-42 split of the
5,389 prepared training pages is fixed as **4,851 train / 538 validation**.
The separate test subset contains 581 pages and is not exposed to the training
loaders. ✅

---

## 2. Model / version names (canonical)

| Name | Definition | Status |
|---|---|---|
| Reconstructed Mistral-7B reference system (historically Stage 3) | Local Mistral-7B QLoRA system based on the published KVP10k generative baseline; not an exact reproduction | ✅ |
| Entity-classification initialisation | Final bbox-scale-fixed LayoutLMv3 encoder/classifier pretraining; epoch 10 selected by validation class-aware `KEY`/`VALUE` micro F1 0.828735; no test access | ✅ |
| V1 | LayoutLMv3 + **token-level** projected scaled-dot-product linker with spatial features (`layoutlm_model.py`) | ✅ |
| V2 | LayoutLMv3 + **span-level** projected scaled-dot-product linker with spatial features (`layoutlm_model_v2.py`) | ✅ |
| V3 | Linker-only diagnostic variant, not a headline result | ✅ |
| **V4** | **Final LayoutLMv3-based span-level discriminative model (V4)**: selected entity encoder/classifier, fresh relation linker, corrected 0-to-1000 model-input boxes, representative relation supervision, and validation pair-F1 checkpoint selection | ✅ |
| V4 parameters | 126,513,684 (approximately 125M) | ✅ |

V4 keeps the LayoutLMv3-base encoder, token classifier, and span-level
projected scaled-dot-product linker with spatial features. It combines several
data, supervision, training, and evaluation corrections. Do not attribute its
result to one correction.

---

## 3. Official benchmark and entity results

Official pair evaluator: `kvp10k_official_eval.py` /
`evaluate_kvp10k_benchmark.py`. Pair precision and recall are page-macro values;
F1 is their harmonic mean. TP/FP/FN columns are raw totals and do not reconstruct
the macro values. The evaluation set contains 581 prepared pages. ✅

### 3a. V4 final principal result

Sources: `data/outputs/stage4_bboxscale_fix_v1/final_joint_v4/checkpoint-15/validation_result.json`,
`data/outputs/stage4_bboxscale_fix_v1/final_joint_v4/training_completion.json`,
and `data/outputs/stage4_bboxscale_fix_v1/final_test_checkpoint15_regular/evaluation_kvp10k_official.json`. ✅

Checkpoint 15 was selected only by validation Regular text+location macro F1 at
relation-probability threshold 0.5. Its selection value is 0.3314636757. Early
stopping uses patience 5, with stopping permitted from epoch 10. After selection
was complete, checkpoint 15 was evaluated once on the 581-page test subset.
Epoch 10 is the separate entity-pretraining checkpoint. ✅

| Split/category | Text F1 | Location F1 | Text+location F1 |
|---|---:|---:|---:|
| Validation Regular | 0.363264 | 0.483804 | **0.331464** |
| Test Regular | **0.365022** | **0.490903** | **0.329114** |
| Test All, direct output | 0.234279 | 0.333767 | **0.214139** |

Complete checkpoint-15 direct-output test metrics: ✅

| Category | Mode | Precision | Recall | F1 | TP | FP | FN | Docs scored |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Regular | Text | 0.386330 | 0.345941 | **0.365022** | 1,256 | 1,924 | 2,911 | 483 |
| Regular | Location | 0.525138 | 0.460858 | **0.490903** | 1,852 | 1,328 | 2,315 | 483 |
| Regular | Text+location | 0.347247 | 0.312781 | **0.329114** | 1,134 | 2,046 | 3,033 | 483 |
| All, direct | Text | 0.350747 | 0.175877 | **0.234279** | 1,256 | 1,933 | 7,129 | 532 |
| All, direct | Location | 0.476770 | 0.256756 | **0.333767** | 1,852 | 1,337 | 6,533 | 532 |
| All, direct | Text+location | 0.315264 | 0.162133 | **0.214139** | 1,134 | 2,055 | 7,251 | 532 |

The checkpoint-15 decoder emits Regular linked pairs only. Direct Unkeyed and
Unvalued precision, recall, and F1 are zero. No checkpoint-15 recovery artifact
exists. Do not report the older recovery values with this lineage. ✅

Checkpoint-15 entity metrics are available on the 538-page validation split
only. The micro and macro metrics include `KEY` and `VALUE` and exclude `O`. No
checkpoint-15 test-entity or class-confusion artifact exists. ✅

| Aggregation/class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Micro | 0.790778 | 0.835812 | **0.812671** |
| Macro | 0.772092 | 0.819999 | **0.795279** |
| KEY | 0.721224 | 0.778437 | **0.748739** |
| VALUE | 0.822959 | 0.861561 | **0.841818** |

### 3b. Reconstructed Mistral-7B reference system

Final clean inference uses `data/outputs/stage3_mistral_clean/checkpoint` and
completed all 581 prepared test pages with greedy decoding (`do_sample=false`),
maximum input length 8,192, and maximum 2,048 new tokens. It preserves 581 raw
responses and 581 parsed prediction files. Official result artifact:
`data/outputs/stage3_mistral_clean_final_inference/evaluation_kvp10k_official.json`. ✅

| Category | Mode | Precision | Recall | F1 | TP | FP | FN | Docs scored |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Regular | Text | 0.754281 | 0.760279 | **0.757268** | 3,083 | 761 | 1,084 | 483 |
| Regular | Location | 0.696042 | 0.692743 | **0.694388** | 2,813 | 1,031 | 1,354 | 483 |
| Regular | Text+location | 0.657907 | 0.656550 | **0.657228** | 2,689 | 1,155 | 1,478 | 483 |
| Unkeyed | Text | 0.685771 | 0.677001 | **0.681358** | 1,533 | 811 | 944 | 472 |
| Unkeyed | Location | 0.697061 | 0.689384 | **0.693201** | 1,557 | 787 | 920 | 472 |
| Unkeyed | Text+location | 0.639285 | 0.631216 | **0.635225** | 1,383 | 961 | 1,094 | 472 |
| Unvalued | Text | 0.728452 | 0.758498 | **0.743171** | 1,238 | 510 | 503 | 181 |
| Unvalued | Location | 0.685290 | 0.713265 | **0.698998** | 1,118 | 630 | 623 | 181 |
| Unvalued | Text+location | 0.658019 | 0.685051 | **0.671263** | 1,039 | 709 | 702 | 181 |
| All | Text | 0.738012 | 0.748150 | **0.743046** | 5,854 | 2,371 | 2,531 | 532 |
| All | Location | 0.711174 | 0.714601 | **0.712884** | 5,488 | 2,737 | 2,897 | 532 |
| All | Text+location | 0.666878 | 0.671203 | **0.669034** | 5,111 | 3,114 | 3,274 | 532 |

Final headline F1 rounded to three decimals: Regular 0.757/0.694/0.657;
Unkeyed 0.681/0.693/0.635; Unvalued 0.743/0.699/0.671; All
0.743/0.713/0.669 (text/location/text+location). Do not mix these values with
the superseded `stage3_mistral_final_inference` result or its cluster table. ✅

Final Regular cluster values are deterministic aggregates of saved per-page
metrics from the clean Mistral and checkpoint-15 V4 artifacts under
`data/outputs/stage2/test_cluster_map.json`. No model inference was repeated. ✅

| Stratum | System | Assigned | Scored | Text F1 | Location F1 | Text+location F1 |
|---|---|---:|---:|---:|---:|---:|
| Cluster 0 | Mistral | 319 | 304 | 0.783329 | 0.702728 | **0.673180** |
| Cluster 0 | V4 | 319 | 304 | 0.312482 | 0.488625 | **0.279445** |
| Cluster 1 | Mistral | 213 | 179 | 0.712149 | 0.679495 | **0.629350** |
| Cluster 1 | V4 | 213 | 179 | 0.452555 | 0.492224 | **0.412074** |
| Geometry unavailable | Both | 49 | 0 | Not scored | Not scored | Not scored |

The clusters are annotation-geometry strata with weak separation. The
superseded unmatched-entity diagnostic is excluded because it uses another
threshold, another unit, and the superseded Mistral lineage.

### 3c. Published paper (Naparstek et al., KVP10k): comparison only

| Category | Mode | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Regular | Text | 0.678 | 0.641 | 0.659 |
| Regular | Location | 0.670 | 0.631 | 0.650 |
| Regular | Text+location | 0.627 | 0.595 | 0.611 |
| Unkeyed | Text | 0.584 | 0.620 | 0.601 |
| Unkeyed | Location | 0.635 | 0.672 | 0.653 |
| Unkeyed | Text+location | 0.568 | 0.601 | 0.584 |
| Unvalued | Text | 0.617 | 0.586 | 0.601 |
| Unvalued | Location | 0.634 | 0.604 | 0.618 |
| Unvalued | Text+location | 0.603 | 0.573 | 0.588 |
| All | Text | 0.645 | 0.640 | 0.643 |
| All | Location | 0.665 | 0.657 | 0.661 |
| All | Text+location | 0.615 | 0.608 | 0.612 |

The paper's Table 1 reports page-macro precision, recall, and F1 for Text,
Location, and Text+location in all four categories. It does not publish
individual per-page scores. Direct comparison remains limited by the different
local data and implementation conditions described below. ✅

### 3d. Entity-classification initialisation and legacy diagnostics

The final bbox-scale-fixed entity-classification phase selected checkpoint 10
by validation class-aware `KEY`/`VALUE` micro F1 0.828735. Test data was not
accessed. The selected encoder and classifier initialise V4, which creates a
fresh relation linker. The earlier pre-fix entity-only checkpoint and its
invalid metric remain excluded from the final result lineage. ✅

The corrected label-pipeline audit uses the first 20 pages of the fixed
seed-42 validation subset. It contains 79 ground-truth regular pairs, 53
recovered representative word-level links, 453 positive token-pair cells after
subword expansion, and 44 positive span-pair labels after span collapse. These
counts are diagnostic pipeline units, not official page-macro benchmark
results. ✅

The V2 model diagnostic uses 200 pages from the same held-out validation
split. The historical strict bounding-box predicate removes 1,658 of 3,490
predicted KEY tokens (47.5%) and 6,135 of 12,649 predicted VALUE tokens
(48.5%). Only 81 pages retain both strict key and value spans. The 1,481
candidate-pair logits have median -6.93 and mean -7.59; 36 candidate pairs
and 20 best-per-key predictions have sigmoid probability at least 0.5. ✅

---

## 4. Training settings actually used

### V4 final run ✅

- Final implementation: `stage4_bboxscale_fix_dataset.py`,
  `train_stage4_bboxscale_entity_pretrain.py`, and
  `train_stage4_bboxscale_final_joint_v4.py`.
- Prepared data and fixed seed-42 manifests: 4,851 train / 538 validation / 581
  test. The training loaders do not expose test data.
- Prepared 0-to-100 boxes are mapped to the LayoutLMv3 0-to-1000 input grid.
  Exported prediction boxes remain on the prepared 0-to-100 grid.
- Constant blank visual input; no document-specific image information.
- Entity initialisation: checkpoint 10, selected by validation class-aware
  `KEY`/`VALUE` micro F1 0.828735. Its encoder and classifier initialise V4. The
  relation linker starts with fresh weights.
- Joint V4: fp32, batch size 1, gradient accumulation 8, AdamW, learning rate
  2e-5, weight decay 0.01, relation-loss weight 5.0, and seed 42.
- Maximum 30 epochs; early-stopping patience 5, with stopping permitted from
  epoch 10; relation-probability threshold 0.5.
- Scheduler steps: `ceil(number_of_batches / 8) × epochs`; 607 optimiser updates
  per epoch, 18,210 maximum updates, and 500 warm-up updates.
- Selection and early stopping use validation Regular text+location macro pair
  F1 only. Checkpoint 15 is selected at optimiser step 9,105 with validation F1
  0.3314636757.
- Training completes 20 epochs and 12,140 optimiser steps. Full-state resumption
  restores the completed epoch-15 state before epochs 16--20.
- Selected checkpoint weights SHA-256:
  `ac494637a12cd12a43953501d649e748d4530fd1400ce0209391fe6d948fd8b0`.
- After selection was complete, checkpoint 15 was evaluated once on the 581-page
  test subset with the dedicated evaluation-only path. The test artifact records
  `test_ground_truth_used_by_model=false`.
- Each checkpoint stores model weights and the complete training state, including
  optimiser, scheduler, completed epoch, global optimiser step, best validation
  value, early-stopping counter, run configuration, history, and random states.

### Mistral (`mistral_baseline.py`) ✅

- Local model: Mistral-7B with 4-bit NF4 QLoRA (`load_in_4bit=True`).
- Settings shared with IBM's released configuration: LoRA rank 4, alpha 4,
  dropout 0.05, no bias; learning rate 5e-4; 8 epochs; batch 1; gradient
  accumulation 4; maximum length 8,192; seed 0.
- The released KVP10k repository contains the baseline prompt in
  `utils/prompt_utils.py` and these settings in `config/base.yaml` and
  `config/kvp.yaml`.
- Local differences: PyMuPDF native text instead of Tesseract OCR; reduced
  prepared data coverage; an adapted prompt/output contract; Hugging Face LoRA
  target names; 4-bit NF4 QLoRA instead of the released bfloat16 LoRA load;
  Hugging Face Trainer and paged 8-bit AdamW instead of PyTorch Lightning and
  non-8-bit AdamW.
- The local objective uses response-only supervision: document and instruction
  tokens are retained as conditioning context but masked from the loss, while
  the structured KVP response remains supervised. This avoids spending part of
  the optimisation signal on reproducing input tokens that are not themselves
  the extraction target. IBM's released implementation masks only padding
  tokens, so its loss also covers the prompt. No ablation isolates the effect
  of this difference.
- Final type-aware inference and parsing use
  `stage3_mistral_final_inference.py`; `mistral_baseline.py` delegates to
  the same parser.
- The Mistral loader scans 5,389 prepared training pages and skips 404 pages
  with empty targets. Training therefore uses 4,985 pages for a fixed eight
  epochs. No validation or test metric selects a checkpoint. The test split is
  accessed only afterward for final prediction and benchmark evaluation.

---

## 5. Evaluation protocol (exact)

Source: `kvp10k_official_eval.py` (mirrors IBM
`benchmark/metrics_calculator.py`). ✅
- Matching: **prediction-order greedy, one-to-one**; a prediction matches a GT only
  if **KVP type is equal** (kvp / unkeyed / unvalued).
- Text: Normalized Edit Distance, **strict `NED < 0.2`**.
- Location: **`IoU ≥ 0.3` (inclusive)** — matches released IBM code.
  ⚠️ Paper prose says IoU **> 0.3**; we follow the executable code and **disclose**
  the discrepancy.
- Malformed/missing bbox ⇒ **fail the location match** (do not raise). ✅
- Aggregation: **macro** precision/recall = mean over documents with **non-empty
  filtered GT** for that category (empty-GT docs excluded per category — see
  "docs scored" columns). ✅
- F1 = harmonic mean of macro P and macro R. ✅
- Modes reported per category: `text_only`, `location_only`, `text_location`. ✅
- Parity: IBM's **unmodified** `MetricsCalculator` reproduced these values exactly
  (Mistral text-only P/R/F1 and V4 Regular all modes). ✅
- Legacy `evaluate_stage4b.py` / `measurements.json` = **pooled entity-level
  diagnostic**, NOT the macro benchmark. Do not cite as headline. ✅

---

## 6. Baseline-comparison limitations (state explicitly)

Our Mistral system is a **reconstructed reference system, not an exact
reproduction** of Naparstek et al. ⚠️
Confounds that make our Regular text/text+location F1 of 0.757/0.657 versus the
paper's 0.659/0.611 only partially comparable:
- Text source: **PyMuPDF native PDF text extraction**, not OCR and not the paper's exact pipeline.
- **Coverage**: preparation retained 5,389 of 9,656 training pages (55.8%). The
  Stage 3 loader then skipped 404 empty-target pages, so Mistral training used
  4,985 pages, or 51.6% of the official training split.
- The local and released configurations share rank 4, alpha 4, dropout 0.05,
  learning rate 5e-4, seed 0, maximum length 8,192, eight epochs, batch 1, and
  gradient accumulation 4.
- The released prompt is public. The local prompt retains its structure but
  changes the detailed bounding-box output instruction and target
  serialization.
- The released code loads the base model in bfloat16. The local model uses
  4-bit NF4 QLoRA, different LoRA target-module names, Hugging Face Trainer,
  paged 8-bit AdamW, and response-only loss instead of loss on all non-padding
  tokens.
- Paper provides category-level macro **precision, recall, and F1**, but not individual per-page scores.
- V4 vs Mistral: V4 is a 125M discriminative model trained without page images;
  Mistral is a 7B generative model — different paradigm and capacity.

---

## 7. Final consistency rules

1. V4 is the final LayoutLMv3-based span-level discriminative model.
2. The final entity-classification initialisation selects epoch 10 by validation
   class-aware `KEY`/`VALUE` micro F1 0.828735. It supplies the encoder and
   classifier to V4. V4 uses a fresh relation linker.
3. V4 uses constant blank visual input and no document-specific images.
4. V4 is selected by validation Regular text+location macro F1 at
   relation-probability threshold 0.5. Checkpoint 15 is selected and is evaluated
   on the test subset only after selection. Epoch 10 refers only to entity
   pretraining.
5. The checkpoint-15 direct model output contains Regular pairs only. No
   checkpoint-15 recovery artifact exists. Do not report older recovery values
   with the final lineage.
6. Checkpoint-15 class-aware entity values are validation metrics. No final
   checkpoint-15 test-entity or class-confusion artifact exists.
7. Location-only evaluation is less restrictive than joint text+location
   evaluation. Do not state that adding text reduces model performance.
8. V4 combines several corrections. Do not make a single-cause claim.
9. Legacy pooled metrics are diagnostics. Do not use them as headline benchmark
   results.
10. Follow the executable benchmark and disclose the paper-prose IoU > 0.3 versus
    released-code IoU ≥ 0.3 difference.
11. Use
    `data/outputs/stage3_mistral_clean_final_inference/evaluation_kvp10k_official.json`
    for every reconstructed-Mistral result. Do not mix its values with the
    superseded parser pass or its old cluster evaluation.
12. The same-page headline comparison is Mistral 0.757/0.694/0.657 versus V4
    0.365/0.491/0.329 for Regular text/location/text+location F1 on the 581-page
    prepared test subset.
