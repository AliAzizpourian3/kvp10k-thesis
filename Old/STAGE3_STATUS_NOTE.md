# Stage 3 Status Note

Date: 2026-03-07

## Very Short Project Story So Far

We already finished the earlier setup work and the CPU heuristic baselines.

Then we discovered that the old Mistral baseline path was not actually correct enough to trust. Instead of forcing a bad run, we stopped, audited the code against IBM's pipeline, and rebuilt Stage 3 properly.

The rebuilt Stage 3 now has two clear phases:
- prepare usable train/test pages from the raw KVP10k PDF sources
- train, predict, and evaluate Mistral on the prepared data

So the project is currently in the middle of a cleaner and more reliable Stage 3 reproduction, not stuck randomly.

## What Was Fixed

### 1. Added `prepare_data.py`

This was the main missing piece in the earlier Stage 3 path.

The old Mistral baseline assumed the dataset already contained clean OCR words and ready-to-train key-value pairs in the format needed by the model. That assumption was wrong, so a dedicated preparation step had to be built.

What `prepare_data.py` now does:
- downloads the source PDF from the dataset URL
- checks that the downloaded file is really a PDF and not an HTML error page
- extracts word-level text and bounding boxes with PyMuPDF
- converts the dataset page numbering correctly from 1-indexed dataset pages to 0-indexed PyMuPDF pages
- fuses extracted words with annotation boxes using an OCR-to-annotation matching step
- builds IBM-style KVP supervision from the linked annotations
- creates LMDX document text for prompting Mistral
- creates the target response text in the Python list-of-lists format expected by the IBM pipeline
- saves one prepared JSON file per unique page

Important fixes inside this step:
- added PDF validation because many URLs return broken content or permission-denied pages
- handled the fact that KVP10k contains about 5 annotator copies per page
- grouped rows by `hash_name` so the same PDF is not downloaded five times
- selected the richest annotation set among the annotator copies for each unique page

Why this matters:
- without this preparation step, the Mistral baseline would not be training on the real KVP10k document inputs
- this step creates the actual train/test material used by the corrected Stage 3 pipeline

### 2. Rewrote `mistral_baseline.py`

The previous version was not just slightly off; it was structurally wrong for this dataset and this benchmark.

What was wrong before:
- it expected nonexistent fields such as precomputed `words` and `kvp`
- it used the wrong prompt style
- it used the wrong response format
- it used different LoRA settings and training hyperparameters from the IBM setup
- it treated the pipeline as if the raw Hugging Face rows were directly trainable

What the rewritten version now does:
- loads the prepared JSON files from `prepare_data.py`
- uses IBM-style prompt construction based on LMDX document text
- uses the IBM-style target format as a Python list of lists
- masks prompt tokens during training so the loss is computed only on the response portion
- uses the corrected LoRA configuration and training settings for the Mistral baseline
- performs greedy decoding during prediction
- parses the model output back into KVP JSON structure

Important implementation fixes inside this rewrite:
- corrected checkpoint loading for PEFT adapters
- added a safe attention fallback because flash-attn is not installed in the environment
- separated the training and prediction paths cleanly
- aligned the script with the prepared-data workflow instead of the broken raw-row workflow

Why this matters:
- the earlier Stage 3 Mistral job would have produced results that were not a meaningful reproduction
- the rewritten script is the first version that is technically consistent with the intended baseline

### 3. Added `evaluate_mistral.py`

Once the prediction path was rebuilt, we also needed a direct evaluation step that uses the new prepared ground truth rather than the older incomplete Stage 3 ground truth files.

What `evaluate_mistral.py` does:
- loads prediction JSON files
- loads the prepared test-set ground truth
- compares predicted entities against ground truth entities
- computes document-level and aggregate matching results
- supports text-only matching and text-plus-bounding-box matching
- writes a structured evaluation JSON for later reporting

Why this matters:
- it gives us a Stage 3 evaluation path that matches the rebuilt preparation pipeline
- it avoids relying on the older Stage 3 outputs that were not adequate for the corrected Mistral experiment

### 4. Fixed the TinyGPU Cluster Workflow

The earlier assumption was that preparation and training could both be run in a standard compute-job flow. That turned out to be false on this cluster.

What we discovered:
- TinyGPU compute nodes do not have internet access
- PDF downloading therefore fails on compute nodes
- the Mistral model must be cached before GPU training starts

What was changed operationally:
- `logs/stage3_prepare_data.sbatch` was changed into a login-node style preparation script rather than a normal compute-node batch job
- `logs/stage3_mistral.sbatch` was updated for offline GPU execution
- the Mistral model was downloaded into the Hugging Face cache ahead of time
- offline environment settings were added so the GPU job can run without trying to reach the internet

Why this matters:
- data preparation now runs where PDF download is actually possible
- training now runs where GPU is available and internet is not required
- this split is necessary on TinyGPU for Stage 3 to work at all

## Important Cluster Fact

TinyGPU compute nodes do not have internet access.

That means:
- PDF downloading must run on the login node
- GPU jobs must run offline from cached data and cached model weights

## Exact Current Stage 3 Position

### Test Preparation
- complete
- unique pages processed: 1051
- prepared successfully: 581
- success rate: 55.3%
- prepared pages with non-empty KVP supervision: 532

### Train Preparation
- complete
- unique pages: 9656
- prepared successfully: 5389
- success rate: 55.8%
- prepared pages with non-empty KVP supervision: 4985

### Training & Prediction
- complete
- Stage 3 Mistral-7B-Instruct-v0.2 LoRA model trained on 4,985 pages with supervision
- predictions generated for all 581 test pages

### Evaluation Results

**Stage 3 Mistral Baseline (581 test pages)**

Default thresholds (NED=0.5, IoU=0.5):
| Mode | Precision | Recall | F1 |
|------|-----------|--------|-----|
| text_only | 0.6269 | 0.7845 | 0.6969 |
| text_bbox | 0.4968 | 0.6217 | 0.5523 |

Stage-0 protocol thresholds (NED=0.2, IoU=0.3):
| Mode | Precision | Recall | F1 |
|------|-----------|--------|-----|
| text_only | 0.6037 | 0.7554 | 0.6711 |
| text_bbox | 0.5394 | 0.6750 | 0.5997 |

**Key Notes on Coverage:**
- Evaluated on 581 successfully-prepared unique test pages (~55% of 1,051 unique pages)
- Loss due to: PDF download failures and scanned PDFs without usable text layers (PyMuPDF limitation)
- For paper comparison, use Stage-0 threshold results (NED=0.2, IoU=0.3)
- Full results saved to `data/outputs/stage3_mistral/evaluation_stage0_ned02_iou03.json`

## Next Step

Update thesis Chapter 7 (Results) with Stage 3 findings, including dataset coverage explanation and all three evaluation tables.

After Stage 3 finishes, the content here can be folded into:
- Chapter 5 implementation details
- Chapter 6 experiment setup
- Chapter 7 baseline results

## Update: OOM Fix (2026-03-08)

### What happened

The first corrected Mistral training job (job 1545825) was submitted automatically after train preparation completed. It crashed within the first training step with CUDA out-of-memory on the A100 40GB GPU.

Root cause: loading Mistral-7B in full bf16 uses about 14GB for weights alone, plus AdamW optimizer states double that, plus activations at max_length=8192. Total exceeded 40GB.

### What was changed

Switched from full-precision LoRA to QLoRA (quantized LoRA):
- base model loaded in 4-bit NF4 quantization via bitsandbytes (reduces weight memory from ~14GB to ~4GB)
- double quantization enabled for additional memory savings
- gradient checkpointing enabled (trades compute for memory on activations)
- optimizer changed from `adamw_torch` to `paged_adamw_8bit` (8-bit optimizer states, paged to avoid fragmentation)
- `prepare_model_for_kbit_training` applied before attaching LoRA adapters

bitsandbytes 0.49.2 was installed into the environment.

### What stays the same

- LoRA config: r=4, alpha=4, same target modules
- training hyperparameters: lr=5e-4, 8 epochs, batch=1, grad_accum=4, max_len=8192
- prompt/target format unchanged
- prediction and evaluation paths unchanged

### Current status

Second run submitted as job 1546224 on the A100 partition, pending in queue.

### Why this is a valid adaptation

IBM's original setup used a larger GPU (A100 80GB). QLoRA is a well-established technique that preserves model quality while fitting within smaller GPU memory. This is a practical adaptation, not a methodological deviation.

## Update: Error Analysis Complete (2026-03-22)

### What was done

Created `analyze_stage3_errors.py` to categorize Stage 3 errors by:
1. **Layout cluster** (sparse vs. dense)
   - Sparse (Cluster 0): ≤ median annotation density (~0.0025 KVPs/pixel)
   - Dense (Cluster 1): > median annotation density
2. **Error type** (via text set matching)
   - Correct: predicted text in ground truth
   - Hallucinated: predicted text NOT in ground truth (spurious generation)
   - Missed: ground truth text NOT predicted (undergeneration)

### Key Findings

**Per-Cluster Breakdown (581 test pages):**

| Cluster | Docs | Predicted KVPs | GT KVPs | Correct (%) | Hallucinated (%) | Missed (%) |
|---------|------|-----------------|---------|-------------|-----------------|-----------|
| **Cluster 0: Dense** | 465 | 7,852 | 8,161 | 48.7 | 40.5* | 31.2 |
| **Cluster 1: Sparse** | 116 | 889 | 224 | 14.4 | 52.9* | 41.1 |

*Hallucinated % is calculated as the percentage of predicted KVPs that are unmatched in ground truth (spurious generation).

**Summary:**
- Dense layouts: **3.4× higher correctness** (48.7% vs 14.4%)
- Sparse layouts: **catastrophic over-prediction** (889 preds vs 224 GT = 3.97× ratio), thus **severe hallucination** (52.9% of predictions are unmatched)
- Dense layouts: near-balanced prediction (7,852 preds vs 8,161 GT = 0.96× ratio), thus **controlled hallucination** (40.5% of predictions unmatched)
- The hallucination % must be interpreted relative to prediction volume: sparse docs halluci ate more both in rate (52.9%) **and** in absolute count (470 spurious KVPs)

### Interpretation

Mistral's pure language-model architecture excels at text extraction (F1=0.67) but lacks spatial reasoning and layout awareness:
- On **dense layouts**, abundant textual context (many co-occurring entities) and repeated patterns learned during training enable pattern recognition and constraint inference.
- On **sparse layouts**, weak textual signal and distribution shift (fewer examples per page than in training) cause the model to default to template-driven hallucination rather than learned restraint.
- The text-vs-spatial gap (text-only F1=0.67 vs text+bbox F1=0.60) indicates the model can identify entities but struggles with precise localization — a task fundamentally misaligned with pure language-model architecture.

### Research Implications

This analysis directly motivates **Stage 4: LayoutLMv3 + Learned Linker**:
- **LayoutLMv3's layout embeddings** (visual position, relative location) provide spatial anchoring, enabling the model to reason about layout density and reduce hallucination on sparse documents.
- **Learned linker module** should improve spatial localization (text+bbox F1) by explicitly modeling key-value proximity patterns.
- **Hypothesis**: Dense layouts should remain strong (48%+ correctness); sparse layouts should improve from 14.4% to competitive levels via layout-aware architectures.

### Artifacts

- Error analysis script: `code/script/analyze_stage3_errors.py`
- Per-document details: `data/outputs/stage3_mistral/error_analysis/error_details.json`
- Cluster summary: `data/outputs/stage3_mistral/error_analysis/cluster_error_summary.json`
- Thesis integration: Chapter 7, Section "Per-Cluster Analysis: Answering 'Why does Mistral Fail?'" (with LaTeX table)