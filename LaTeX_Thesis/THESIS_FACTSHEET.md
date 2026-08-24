# KVP10k Thesis — Single Source-of-Truth Fact Sheet

Purpose: the one authoritative reference for every number, name, and setting used
while writing/polishing the thesis. Prefer this file over prose in any chapter,
and over the repo `README.md` / `measurements.json`, which contain **obsolete
pooled scores** and must not be cited.

Legend: ✅ = verified against a code/data artifact in this repo · ⚠️ = prose-only,
estimated, or not cleanly recoverable (state cautiously in the thesis).

Verification date: 2026-08-24.

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
`evaluate_kvp10k_benchmark.py`. Pair precision and recall are page-macro values;
F1 is their harmonic mean. TP/FP/FN columns are raw totals and do not reconstruct
the macro values. The evaluation set contains 581 prepared pages. ✅

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

Complete selected-checkpoint direct-output test metrics: ✅

| Category | Mode | Precision | Recall | F1 | TP | FP | FN | Docs scored |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Regular | Text | 0.424863 | 0.363705 | **0.391913** | 1,434 | 1,467 | 2,733 | 483 |
| Regular | Location | 0.484977 | 0.413212 | **0.446227** | 1,623 | 1,278 | 2,544 | 483 |
| Regular | Text+location | 0.371127 | 0.322854 | **0.345312** | 1,216 | 1,685 | 2,951 | 483 |
| All, direct | Text | 0.385731 | 0.193856 | **0.258033** | 1,434 | 1,478 | 6,951 | 532 |
| All, direct | Location | 0.440308 | 0.232513 | **0.304322** | 1,623 | 1,289 | 6,762 | 532 |
| All, direct | Text+location | 0.336944 | 0.172981 | **0.228602** | 1,216 | 1,696 | 7,169 | 532 |

Direct Unkeyed and Unvalued P/R/F1 are 0 because the learned decoder emits only
Regular pairs. They are supplied only by the unchanged recovery step below. ✅

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

| Category | Mode | Precision | Recall | F1 | TP | FP | FN | Docs scored |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Regular, unchanged | Text | 0.424863 | 0.363705 | **0.391913** | 1,434 | 1,467 | 2,733 | 483 |
| Regular, unchanged | Location | 0.484977 | 0.413212 | **0.446227** | 1,623 | 1,278 | 2,544 | 483 |
| Regular, unchanged | Text+location | 0.371127 | 0.322854 | **0.345312** | 1,216 | 1,685 | 2,951 | 483 |
| Unkeyed | Text | 0.189219 | 0.163551 | **0.175451** | 240 | 1,212 | 2,237 | 472 |
| Unkeyed | Location | 0.348060 | 0.290085 | **0.316439** | 409 | 1,043 | 2,068 | 472 |
| Unkeyed | Text+location | 0.172947 | 0.152761 | **0.162229** | 214 | 1,238 | 2,263 | 472 |
| Unvalued | Text | 0.298976 | 0.297524 | **0.298248** | 360 | 636 | 1,381 | 181 |
| Unvalued | Location | 0.351799 | 0.321433 | **0.335931** | 399 | 597 | 1,342 | 181 |
| Unvalued | Text+location | 0.278177 | 0.273611 | **0.275875** | 319 | 677 | 1,422 | 181 |
| All | Text | 0.323127 | 0.266804 | **0.292277** | 2,034 | 3,612 | 6,351 | 532 |
| All | Location | 0.416247 | 0.343449 | **0.376360** | 2,431 | 3,215 | 5,954 | 532 |
| All | Text+location | 0.286822 | 0.239297 | **0.260913** | 1,749 | 3,897 | 6,636 | 532 |

V4 Regular results by frozen annotation-geometry cluster: Cluster 0 has
319 assigned pages and 304 scored pages; text/location/combined F1 =
0.359/0.430/**0.305**. Cluster 1 has 213 assigned pages and 179 scored pages;
text/location/combined F1 = 0.444/0.468/**0.410**. The 49 geometry-unavailable
pages have no scorable Regular ground truth and are excluded. These clusters
are geometry regimes, not semantic document types. ✅

### 3b. Mistral baseline (re-implementation)

Final inference uses `data/outputs/stage3_mistral/checkpoint` and completed all
581 prepared test pages with greedy decoding (`do_sample=false`), maximum input
length 8,192, and maximum 2,048 new tokens. It preserves 581 raw responses and
581 parsed prediction files. Official result artifact:
`data/outputs/stage3_mistral_final_inference/evaluation_kvp10k_official.json`. ✅

| Category | Mode | Precision | Recall | F1 | TP | FP | FN | Docs scored |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Regular | Text | 0.763149 | 0.775018 | **0.769038** | 3,116 | 850 | 1,051 | 483 |
| Regular | Location | 0.698525 | 0.700213 | **0.699368** | 2,851 | 1,115 | 1,316 | 483 |
| Regular | Text+location | 0.660577 | 0.663004 | **0.661788** | 2,727 | 1,239 | 1,440 | 483 |
| Unkeyed | Text | 0.705257 | 0.693157 | **0.699155** | 1,535 | 814 | 942 | 472 |
| Unkeyed | Location | 0.708039 | 0.692368 | **0.700116** | 1,559 | 790 | 918 | 472 |
| Unkeyed | Text+location | 0.652894 | 0.639909 | **0.646336** | 1,395 | 954 | 1,082 | 472 |
| Unvalued | Text | 0.739973 | 0.761732 | **0.750695** | 1,232 | 484 | 509 | 181 |
| Unvalued | Location | 0.694542 | 0.705239 | **0.699850** | 1,141 | 575 | 600 | 181 |
| Unvalued | Text+location | 0.663828 | 0.674710 | **0.669225** | 1,060 | 656 | 681 | 181 |
| All | Text | 0.747052 | 0.756358 | **0.751676** | 5,883 | 2,261 | 2,502 | 532 |
| All | Location | 0.715336 | 0.719899 | **0.717610** | 5,551 | 2,593 | 2,834 | 532 |
| All | Text+location | 0.670584 | 0.674865 | **0.672718** | 5,182 | 2,962 | 3,203 | 532 |

Final headline F1 rounded to three decimals: Regular 0.769/0.699/0.662;
Unkeyed 0.699/0.700/0.646; Unvalued 0.751/0.700/0.669; All
0.752/0.718/0.673 (text/location/text+location). The earlier parser pass and
its cluster results are superseded and must not be mixed with the final pass. ✅

Final official Regular results by frozen annotation-geometry cluster use
`data/outputs/stage3_mistral_final_inference/evaluation_kvp10k_official_per_cluster.json`. ✅

| Cluster | Mode | Precision | Recall | F1 | TP | FP | FN | Assigned | Scored |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cluster 0 | Text | 0.791479 | 0.780953 | **0.786181** | 2,483 | 519 | 869 | 319 | 304 |
| Cluster 0 | Location | 0.716654 | 0.708182 | **0.712393** | 2,261 | 741 | 1,091 | 319 | 304 |
| Cluster 0 | Text+location | 0.682427 | 0.674887 | **0.678636** | 2,177 | 825 | 1,175 | 319 | 304 |
| Cluster 1 | Text | 0.715034 | 0.764939 | **0.739145** | 633 | 331 | 182 | 213 | 179 |
| Cluster 1 | Location | 0.667735 | 0.686680 | **0.677075** | 590 | 374 | 225 | 213 | 179 |
| Cluster 1 | Text+location | 0.623468 | 0.642822 | **0.632997** | 550 | 414 | 265 | 213 | 179 |

Geometry unavailable has 49 assigned pages and no scorable Regular ground
truth. It is excluded from the Regular cluster averages. ✅

The legacy pooled unmatched-entity diagnostic on the final predictions gives
39.7% for Cluster 0 (3,505/8,834) and 52.5% for Cluster 1 (1,730/3,297).
It uses NED < 0.5, strict IoU > 0.5, no key–value role equality, and all
predicted entities as the denominator. It is not an official benchmark
metric. ✅

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
- Stage 3 training uses only the prepared training split for the fixed eight
  epochs; the held-out test split is accessed only afterward for final
  prediction and benchmark evaluation.

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
Confounds that make our Regular text/combined F1 of 0.769/0.662 versus the
paper's 0.659/0.611 only partially comparable:
- Text source: **PyMuPDF native PDF text extraction**, not OCR and not the paper's exact pipeline.
- **Coverage**: trained on 55.8% of usable train pages (broken PDFs dropped).
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
10. Use the final Mistral inference/evaluation artifact for every Mistral result.
    Do not mix its values with the superseded parser pass or its old cluster
    evaluation.
