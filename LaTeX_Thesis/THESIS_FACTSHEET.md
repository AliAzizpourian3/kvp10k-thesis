# KVP10k Thesis — Single Source-of-Truth Fact Sheet

Purpose: the one authoritative reference for every number, name, and setting used
while writing/polishing the thesis. Prefer this file over prose in any chapter,
and over the repo `README.md` / `measurements.json`, which contain **obsolete
pooled scores** and must not be cited.

Legend: ✅ = verified against a code/data artifact in this repo · ⚠️ = prose-only,
estimated, or not cleanly recoverable (state cautiously in the thesis).

Verification date: 2026-08-18.

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

**V4 train/val split** (`stage4_kvp_dataset.create_stage4_dataloaders`):
`random_split` of the 5,389 prepared train pages, `val_fraction=0.1`,
`torch.Generator().manual_seed(42)` ⇒ **4,851 train / 538 val**. Test = 581 pages. ✅

---

## 2. Model / version names (canonical)

| Name | Definition | Status |
|---|---|---|
| Stage 3 (Mistral) | Re-implemented KVP10k generative baseline, Mistral-7B QLoRA | ✅ |
| Stage 4a | Methodological entity-pre-training phase; no reliable checkpoint or reportable numerical result survives | ✅ |
| V1 | LayoutLMv3 + **token-level** projected scaled-dot-product linker with spatial features (`layoutlm_model.py`) | ✅ |
| V2 | LayoutLMv3 + **span-level** projected scaled-dot-product linker with spatial features (`layoutlm_model_v2.py`) | ✅ |
| V3 | Linker-only diagnostic variant, not a headline result | ✅ |
| **V4** | **Final corrected LayoutLMv3 experiment**: corrected V2 data and supervision pipeline with full-state resumption, corrected metrics, and validation pair-F1 checkpoint selection | ✅ |
| V4 parameters | 126,513,684 (approximately 125M) | ✅ |

V4 keeps the LayoutLMv3-base encoder, token classifier, and span-level
projected scaled-dot-product linker with spatial features. It combines several
data, supervision, training, and evaluation corrections. Do not attribute its
result to one correction.

---

## 3. Official benchmark and entity results

Official pair evaluator: `kvp10k_official_eval.py` /
`evaluate_kvp10k_benchmark.py`. Pair values below are macro F1. The evaluation
set contains 581 prepared pages. ✅

### 3a. V4 final principal result

Selected checkpoint: epoch 10, chosen only by official validation Regular
text+location macro F1 at linker score threshold 0.5. Validation combined F1 =
0.3417870621. The selected checkpoint was evaluated on the test set after
selection. ✅

| Split/category | Text F1 | Location F1 | Text+location F1 |
|---|---:|---:|---:|
| Validation Regular | 0.399 | 0.445 | **0.342** |
| Test Regular | **0.392** | **0.446** | **0.345** |
| Test All, direct output | 0.258 | 0.304 | **0.229** |

Exact test Regular F1: text 0.3919126343; location 0.4462274760; combined
0.3453115284. The learned decoder directly emits Regular linked pairs. ✅

Corrected class-aware entity results on the test set: ✅

| Aggregation/class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Micro | 0.745948 | 0.844332 | **0.792097** |
| Macro | 0.724199 | 0.826742 | **0.771948** |
| KEY | 0.666908 | 0.782975 | **0.720296** |
| VALUE | 0.781489 | 0.870509 | **0.823601** |

Confusion counts: 803 KEY targets predicted as VALUE; 584 VALUE targets
predicted as KEY. The metric counts each KEY–VALUE confusion as a false
positive for the predicted class and a false negative for the target class.
The reported corrected micro and macro aggregates include KEY and VALUE and
exclude O. ✅

V4 post-processing recovery uses the existing unchanged method. It emits
unlinked VALUE spans as unkeyed items and unlinked KEY spans as unvalued items.
It does not change Regular predictions or Regular F1. ✅

| Recovered category | Text F1 | Location F1 | Text+location F1 |
|---|---:|---:|---:|
| Regular, unchanged | 0.392 | 0.446 | 0.345 |
| Unkeyed | 0.175 | 0.316 | **0.162** |
| Unvalued | 0.298 | 0.336 | **0.276** |
| All | 0.292 | 0.376 | **0.261** |

V4 Regular results by frozen annotation-geometry cluster: Cluster 0 has
319 assigned pages and 304 scored pages; text/location/combined F1 =
0.359/0.430/**0.305**. Cluster 1 has 213 assigned pages and 179 scored pages;
text/location/combined F1 = 0.444/0.468/**0.410**. The 49 geometry-unavailable
pages have no scorable Regular ground truth and are excluded. These clusters
are geometry regimes, not semantic document types. ✅

### 3b. Mistral baseline (re-implementation)

| Category | Text F1 | Location F1 | Text+location F1 | Docs scored |
|---|---:|---:|---:|---:|
| Regular | 0.598 | 0.547 | **0.521** | 483 |
| Unkeyed | 0.000 | 0.000 | 0.000 | 472 |
| Unvalued | 0.744 | 0.699 | 0.672 | 181 |
| All | 0.513 | 0.478 | 0.455 | 532 |

Exact Regular combined F1 = 0.52057. Mistral remains stronger than V4 for the
official comparable Regular combined result. Its corrected cluster combined F1
is 0.569 for Cluster 0 and 0.429 for Cluster 1. ✅

### 3c. Published paper (Naparstek et al., KVP10k): comparison only

| Category | Text F1 | Location F1 | Text+location F1 |
|---|---:|---:|---:|
| Regular | 0.659 | 0.650 | 0.611 |
| Unkeyed | 0.601 | 0.653 | 0.584 |
| Unvalued | 0.601 | 0.618 | 0.588 |
| All | 0.643 | 0.661 | 0.612 |

The paper supplies F1 but not per-page precision and recall. Compare only F1. ✅

### 3d. Stage 4a and legacy diagnostics

Stage 4a is a methodological entity-pre-training phase. No reliable Stage 4a
checkpoint survives, so it cannot be rescored with the corrected class-aware
metric and has no validated or reportable numerical result. The historical
metric did not correctly penalize KEY–VALUE confusion and is removed. ✅

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

- Trainer: `train_stage4b_v5.py`; LayoutLMv3-base + token classifier + span-level projected scaled-dot-product linker with spatial features.
- Prepared data and seed-42 split: 4,851 train / 538 validation / 581 test.
- Constant blank visual input; no document-specific image information.
- Warm start: Canary B, not an intermediate development checkpoint. Loaded 226 compatible entries:
  212 encoder, 2 classifier, and 12 linker entries.
- Canary B checkpoint SHA-256:
  `e86c1c76323b34e2a195d5097f12504f6c2c54a424b3ba0f5af4efb610e9aed1`.
- Batch size 1; gradient accumulation 8; effective batch size 8; AdamW;
  learning rate 2e-5; weight decay 0.01; linker-loss weight 5.0.
- Maximum 30 epochs; early-stopping patience 10; score threshold 0.5.
- Scheduler steps: `ceil(number_of_batches / 8) × epochs`; 607 optimizer updates
  per epoch, 18,210 maximum steps, and 500 warm-up optimizer steps.
- Selection and early stopping: official validation Regular text+location macro
  pair F1. Best epoch 10 at optimizer step 6,070.
- Completed 20 epochs and 12,140 optimizer steps; early stopping occurred after
  10 consecutive epochs without an improvement.
- The first job reached its 24-hour limit after epoch 15. The second job restored
  the full state and completed epochs 16–20.
- Approximate allocated GPU time: 31 h 19 min.
- Selected checkpoint SHA-256:
  `f4b61bf2db8833aa5b23bf49c9778fb154177b18954e6f8563a59cf80145a27c`.
- Each checkpoint saves model, AdamW optimizer, scheduler, gradient scaler,
  epoch, global optimizer step, best score, early-stopping counter, complete
  history, run configuration, and Python/NumPy/PyTorch/CUDA random states.
  `pytorch_model.bin` is also stored for evaluator compatibility.

### Mistral (`mistral_baseline.py`) ✅

- Mistral-7B, QLoRA (`load_in_4bit=True`).
- LoRA: rank 4, alpha 4, dropout 0.05, no bias; Hugging Face attention targets.
- AdamW, learning rate 5e-4, 8 epochs, batch 1, gradient accumulation 4,
  effective batch 4, maximum length 8,192, seed 0.

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

Our Mistral is a **re-implementation, not a reproduction** of Naparstek et al. ⚠️
Confounds that make the 0.598/0.521 vs paper 0.659/0.611 gap only partially comparable:
- Text source: **PyMuPDF native PDF text extraction**, not OCR and not the paper's exact pipeline.
- **Coverage**: trained on 55.8% of usable train pages (broken PDFs dropped).
- LoRA rank **r=4** here; the paper's exact rank/hyperparameters are **not confirmed**
  (thesis previously estimated ≥16). Do not claim identical hyperparameters. ⚠️
- Prompt formatting, seed, max_len, and epoch budget are our choices, not the paper's.
- Paper provides **F1 only** (no P/R) ⇒ compare F1, and only at the category level.
- V4 vs Mistral: V4 is a 125M discriminative model trained without page images;
  Mistral is a 7B generative model — different paradigm and capacity.

---

## 7. Final consistency rules

1. V4 is the final corrected LayoutLMv3 experiment.
2. Stage 4a has no reportable score because no reliable checkpoint survives and
   its historical entity metric was not class-aware.
3. V4 uses constant blank visual input and no document-specific images.
4. V4 is selected by official validation Regular text+location macro F1 at
   threshold 0.5; epoch 10 is selected and is tested after selection.
5. Direct model output contains Regular pairs. Unkeyed and unvalued outputs come
   from unchanged post-processing recovery.
6. Location-only evaluation is less restrictive than joint text+location
   evaluation. Do not state that adding text reduces model performance.
7. V4 combines several corrections. Do not make a single-cause claim.
8. Legacy pooled metrics are diagnostics. Do not use them as headline benchmark
   results.
9. Follow the executable benchmark and disclose the paper-prose IoU > 0.3 versus
   released-code IoU ≥ 0.3 difference.
