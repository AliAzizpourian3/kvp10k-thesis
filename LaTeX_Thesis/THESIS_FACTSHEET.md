# KVP10k Thesis — Single Source-of-Truth Fact Sheet

Purpose: the one authoritative reference for every number, name, and setting used
while writing/polishing the thesis. Prefer this file over prose in any chapter,
and over the repo `README.md` / `measurements.json`, which contain **obsolete
pooled scores** and must not be cited.

Legend: ✅ = verified against a code/data artifact in this repo · ⚠️ = prose-only,
estimated, or not cleanly recoverable (state cautiously in the thesis).

Verification date: 2026-08-06.

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

**V4 train/val split** (`stage4_kvp_dataset.create_stage4_dataloaders`):
`random_split` of the 5,389 prepared train pages, `val_fraction=0.1`,
`torch.Generator().manual_seed(42)` ⇒ **4,851 train / 538 val**. Test = 581 pages. ✅

---

## 2. Model / version names (canonical)

| Name | Definition | Status |
|---|---|---|
| Stage 3 (Mistral) | Re-implemented KVP10k generative baseline, Mistral-7B QLoRA | ✅ |
| V1 | LayoutLMv3 + **token-level** biaffine linker (`layoutlm_model.py`) | ✅ |
| V2 | LayoutLMv3 + **span-level** biaffine linker (`layoutlm_model_v2.py`) | ✅ |
| V3 | Linker-only diagnostic variant (not a headline row) | ⚠️ keep as diagnostic |
| **V4** | **The corrected V2** run (three data-pipeline bugs fixed); the final/headline discriminative model | ✅ |
| V4 params | 126,513,684 (~125M) | ✅ |

V4 is **not** a new architecture — it is V2 with the pipeline fixes. Say so once
explicitly in the thesis.

---

## 3. Final official benchmark numbers (headline)

Evaluator: `kvp10k_official_eval.py` / `evaluate_kvp10k_benchmark.py`.
Artifacts:
`data/outputs/stage4b_v4/evaluation_kvp10k_official.json`,
`data/outputs/stage3_mistral/evaluation_kvp10k_official.json`,
`data/outputs/stage4b_v4/evaluation_kvp10k_official_alltypes.json`.
All values are **macro F1** (see §5). `documents_loaded = 581`. ✅

### 3a. V4 (headline = regular-only model)
| Category | Text F1 | Location F1 | Text+Location F1 | Docs scored |
|---|---|---|---|---|
| **Regular** | **0.334** | 0.456 | **0.309** | 483 (98 empty-GT excl.) |
| Unkeyed | 0.000 | 0.000 | 0.000 | 472 |
| Unvalued | 0.000 | 0.000 | 0.000 | 181 |
| All | 0.214 | 0.310 | 0.199 | 532 (49 excl.) |

Exact (for tables): Regular text P/R/F1 = 0.38195 / 0.29644 / 0.33380;
location F1 = 0.45603; combined P/R/F1 = 0.35184 / 0.27473 / 0.30854. ✅
All: text F1 0.21400, location 0.30996, combined 0.19908. ✅

Per-cluster (Regular): Dense (521 docs) text/combined 0.334 / 0.310;
Sparse (60 docs) text/combined 0.343 / 0.227. ✅ (small sparse subset — high variance)

Threshold sensitivity (Regular, fixed predictions): official combined 0.309;
relaxed location (IoU≥0.1) 0.321; relaxed text (NED<0.3) 0.321; text-only 0.334. ✅

### 3b. V4 with decode-time recovery (Analysis C ablation, no retraining)
`...official_alltypes.json`. Regular is **byte-identical** to 3a. ✅
| Category | Text F1 | Location F1 | Text+Location F1 |
|---|---|---|---|
| Regular (unchanged) | 0.334 | 0.456 | 0.309 |
| Unvalued (recovered) | 0.271 | 0.369 | 0.242 | ✅ |
| Unkeyed (recovered) | 0.178 | 0.325 | 0.165 | ✅ |
| **All (with recovery)** | **0.257** | **0.392** | **0.237** | ✅ |

Export counts: kvp = 2,601 (identical to headline), + 1,211 unvalued + 1,747 unkeyed. ✅

### 3c. Mistral baseline (re-implementation)
| Category | Text F1 | Location F1 | Text+Location F1 | Docs scored |
|---|---|---|---|---|
| Regular | 0.598 | 0.547 | 0.521 | 483 |
| Unkeyed | 0.000 | 0.000 | 0.000 | 472 |
| Unvalued | 0.744 | 0.699 | 0.672 | 181 |
| All | 0.513 | 0.478 | 0.455 | 532 |

Exact: Regular text P/R/F1 = 0.48761 / 0.77418 / 0.59835; location F1 0.54711;
combined 0.52057. All text F1 0.51312, location 0.47804, combined 0.45508.
Unvalued text F1 0.74401. Unkeyed = 0 (parser emits no unkeyed predictions). ✅

### 3d. Published paper (Naparstek et al., KVP10k) — comparison only, **F1 only** ✅
| Category | Text | Location | Text+Location |
|---|---|---|---|
| Regular | 0.659 | 0.650 | 0.611 |
| Unkeyed | 0.601 | 0.653 | 0.584 |
| Unvalued | 0.601 | 0.618 | 0.588 |
| All | 0.643 | 0.661 | 0.612 |

No per-page P/R available for the paper — compare F1 only. ✅

### 3e. Entity (token-classification) F1 — separate task
V4 full model entity F1 = **0.827**; entity-only Stage 4a = 0.847. ✅
Linking oracle (GT spans fed to linker; legacy pooled diagnostic, **not** macro):
text 0.365 / text+bbox 0.345 — perfect entities barely help ⇒ linker is the bottleneck. ✅

---

## 4. Training settings actually used

### V4 (`slurm/submit_stage4b_v4*.sh` + `train_stage4b_v2.py`) ✅
- Trainer: `train_stage4b_v2.py`; encoder LayoutLMv3-base + span biaffine linker.
- **Warm start:** `--pretrained_encoder data/outputs/stage4b_canary_B/best_model/pytorch_model.bin` (**not** from scratch). ⚠️ fix any "trained from scratch" prose.
- batch_size 1, gradient_accumulation_steps 8 (**effective batch 8**), lr 2e-5,
  linker_loss_weight 5.0, early_stopping_patience 10, val_fraction 0.1.
- **No `--include_images`** ⇒ trained **without page-image content** despite the multimodal backbone. The dataset supplied blank white placeholder pixel tensors, not actual page pixels (so `pixel_values` was not `None`). ⚠️ do not claim page-image features were used at train time.
- Best checkpoint selected by **validation entity F1**, not pair/link F1. ⚠️ state this.
- GPU: A100 (training); RTX 3080 used for later inference/diagnostics.
- **Epochs:** target **30** across initial + resume + resume-2 runs. Exact completed
  count is **not cleanly recoverable** (trainer restarts local epoch numbering on
  resume; `--resume_from_checkpoint` auto-picks "latest" via **lexicographic** sort,
  a known ordering caveat). ⚠️ report as "targeted 30 epochs across resumed jobs,"
  do not assert a single uninterrupted 30-epoch run.

### Mistral (`mistral_baseline.py`) ✅
- Mistral-7B, **QLoRA** (`load_in_4bit=True`).
- LoRA: r=4, alpha=4, dropout=0.05, bias=none, target_modules = HF attention names.
- Optimizer AdamW, lr 5e-4, 8 epochs, batch 1, grad_accum 4 (eff. 4), max_len 8192, seed 0.

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
- Text source: **PyMuPDF/OCR extraction**, not the paper's exact pipeline.
- **Coverage**: trained on 55.8% of usable train pages (broken PDFs dropped).
- LoRA rank **r=4** here; the paper's exact rank/hyperparameters are **not confirmed**
  (thesis previously estimated ≥16). Do not claim identical hyperparameters. ⚠️
- Prompt formatting, seed, max_len, and epoch budget are our choices, not the paper's.
- Paper provides **F1 only** (no P/R) ⇒ compare F1, and only at the category level.
- V4 vs Mistral: V4 is a 125M discriminative model trained without page images;
  Mistral is a 7B generative model — different paradigm and capacity.

---

## 7. Known prose contradictions to fix during polishing
1. "Trained from scratch" → **warm-started** from `stage4b_canary_B`. ⚠️
2. Any claim that V4 uses page images → it does **not** (no `--include_images`). ⚠️
3. "Best model at epoch 13" / single 30-epoch run → say "targeted 30 across resumed
   jobs; best by val entity F1." ⚠️
4. V4 described as a 4th architecture → it is the **corrected V2**. ⚠️
5. Old pooled numbers (0.342/0.309, 0.671/0.600, 0.339/0.306) anywhere except the
   explicitly-labeled legacy diagnostic → replace with §3 macro numbers. ✅ mostly done
6. "Follows the KVP10k protocol exactly" → follows the **executable** benchmark;
   disclose the IoU `>` vs `≥` discrepancy. ✅
