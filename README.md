# KVP10k: Key-Value Pair Extraction Pipeline

![Stages](https://img.shields.io/badge/Stages-0--4-blue) ![Dataset](https://img.shields.io/badge/Dataset-KVP10k%20ICDAR%202024-orange) ![Status](https://img.shields.io/badge/Thesis%20Results-Final-green)

Thesis project for key-value pair (KVP) extraction on the [KVP10k dataset](https://huggingface.co/datasets/ibm/KVP10k) (ICDAR 2024). The final internal comparison covers a reconstructed Mistral baseline and the corrected LayoutLMv3 V4 model. [LaTeX_Thesis/THESIS_FACTSHEET.md](LaTeX_Thesis/THESIS_FACTSHEET.md) is the authoritative source for all reported values and settings.


## Pipeline Overview

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | `kvp10k_official_eval.py` + `evaluate_kvp10k_benchmark.py` | Released-benchmark-compatible evaluation |
| 1 | `data_loader.py` | Dataset ingestion from HuggingFace |
| 2 | `features.py` + `visualization.py` | Annotation-geometry clustering and audit |
| 3 | `prepare_data.py` + `mistral_baseline.py` + `stage3_mistral_final_inference.py` | Data preparation and reconstructed Mistral baseline |
| 4a | `train_stage4a.py` | Methodological entity-pre-training phase; no reportable result |
| V4 | `train_stage4b_v5.py` | Final corrected LayoutLMv3 and span-level relation linker |

---

## Stage 0: Evaluation Protocol

Implemented in `kvp10k_official_eval.py` and exposed by
`evaluate_kvp10k_benchmark.py`.

- **Text matching**: NED (Normalised Edit Distance) < 0.2
- **Location matching**: IoU >= 0.3, as implemented by the released IBM code
- **Matching**: prediction-order greedy, one-to-one, with equal KVP type
- **Aggregation**: page-macro precision and recall over pages with non-empty filtered ground truth
- **F1**: harmonic mean of macro precision and macro recall

---

## Stage 1: Dataset Ingestion

`data_loader.py` loads KVP10k from HuggingFace (`ibm/KVP10k`, cached locally at `hf_cache/`):

- 10,707 unique pages: 9,656 train and 1,051 test
- Prepared pages: 5,389 train and 581 test
- Each page: PDF URL + word annotations + KVP ground truth
- Annotation format: bounding polygon coordinates per word, structured KVP pairs

---

## Stage 2: Layout Clustering & Data Audit

`features.py` extracts 12 annotation-geometry features per document page:

| Feature | Description |
|---------|-------------|
| `n_boxes` | Number of annotated bounding boxes |
| `total_area` | Total area covered by boxes |
| `mean_area`, `std_area` | Box area statistics |
| `mean_width`, `mean_height` | Average box dimensions |
| `mean_aspect_ratio` | Mean width/height ratio |
| `mean_cx`, `mean_cy` | Centroid of layout |
| `v_spread`, `h_spread` | Vertical/horizontal extent |
| `mean_spacing` | Average inter-box spacing |

The historical `density` feature was removed because it duplicated
`total_area`. The frozen seed-42 K-means model selects k=2. On the prepared
581-page test subset, Cluster 0 has 319 assigned pages, Cluster 1 has 213, and
49 have unavailable geometry.

---

## Stage 3: Data Preparation & Baselines

### Data Preparation (`prepare_data.py`)

Converts raw KVP10k pages into prepared JSON files for Stage 4 training:

1. Download PDF from HuggingFace image URL
2. Render page at 300 DPI with PyMuPDF (native text extraction, no Tesseract dependency)
3. Fuse extracted words with annotation bounding boxes (word-match threshold = 0.6)
4. Produce LMDX-format text and ground-truth KVP labels

Each `data/prepared/{train,test}/{hash_name}.json` record contains the
quantised LMDX text, the complete training prompt, the list-of-lists target,
and typed `gt_kvps` entries. Ground-truth entries use the official `kvp`,
`unkeyed`, or `unvalued` type and nested text and bounding-box fields.

Dataset sizes after preparation: **5,389 train** / **581 test**.

### Nearest-Neighbour Baseline (`baselines.py`)

Rule-based: pair each key with the spatially closest value (Euclidean centroid distance, max 0.3 normalised units). No learning required.

### Reconstructed Mistral-7B Baseline

Stage 3 is a re-implementation, not an exact reproduction. The local and
released IBM configurations share the model, rank 4, alpha 4, dropout 0.05,
learning rate 5e-4, seed 0, maximum length 8192, eight epochs, batch size 1,
and gradient accumulation 4.

The local implementation uses PyMuPDF native text, reduced prepared coverage,
an adapted prompt/output contract, Hugging Face LoRA target names, 4-bit NF4
QLoRA, Hugging Face Trainer, paged 8-bit AdamW, and response-only loss. The
released IBM implementation uses Tesseract OCR, bfloat16 LoRA, PyTorch
Lightning, non-8-bit AdamW, and loss on all non-padding tokens.

Response-only supervision is a deliberate task-aligned choice: the document
and instruction are conditioning input, while the structured KVP response is
the prediction target. No controlled ablation isolates its effect on the final
score.

`stage3_mistral_final_inference.py` is the canonical final inference and
type-aware parser path. `mistral_baseline.py` delegates prediction parsing to
the same implementation. The parser maps Regular, Unkeyed, and Unvalued
list-of-lists entries to their benchmark types before evaluation.

Final page-macro F1 on the prepared 581-page test subset:

| Category | Text | Location | Text+location |
|----------|-----:|---------:|--------------:|
| Regular | 0.769 | 0.699 | 0.662 |
| Unkeyed | 0.699 | 0.700 | 0.646 |
| Unvalued | 0.751 | 0.700 | 0.669 |
| All | 0.752 | 0.718 | 0.673 |

Final Regular text+location F1 by frozen annotation-geometry cluster is 0.679
for Cluster 0 and 0.633 for Cluster 1.

Authoritative result files:

- `data/outputs/stage3_mistral_final_inference/evaluation_kvp10k_official.json`
- `data/outputs/stage3_mistral_final_inference/evaluation_kvp10k_official_per_cluster.json`
- `data/outputs/stage3_mistral_final_inference/unmatched_entity_audit/cluster_error_summary.json`

---

## Stage 4: LayoutLMv3 Fine-tuning

V4 is the final corrected LayoutLMv3 experiment. It uses:

- LayoutLMv3-base with a token classifier
- span-level projected scaled-dot-product linking with spatial features
- extracted text and bounding boxes
- a constant blank visual input, with no document-specific image information
- batch size 1, gradient accumulation 8, and learning rate 2e-5
- official validation Regular text+location macro F1 for checkpoint selection
- score threshold 0.5

The selected epoch-10 checkpoint produces these direct Regular test results:

| Mode | Precision | Recall | F1 |
|------|----------:|-------:|---:|
| Text | 0.425 | 0.364 | 0.392 |
| Location | 0.485 | 0.413 | 0.446 |
| Text+location | 0.371 | 0.323 | 0.345 |

Corrected class-aware entity F1 is 0.792 micro and 0.772 macro. The unchanged
post-processing recovery adds Unkeyed and Unvalued outputs and gives All
text+location F1 of 0.261. It does not change Regular predictions.

Regular text+location F1 by frozen cluster is 0.305 for Cluster 0 and 0.410 for
Cluster 1. The 49 geometry-unavailable pages have no scorable Regular ground
truth.

Stage 4a remains a methodological entity-pre-training phase. No reliable
checkpoint or class-aware numerical result survives, so it has no reportable
score.

### Evaluation

Re-evaluate saved predictions without training or inference:

```bash
# Final Stage 3
PYTHONPATH=code/script env/kvp10k_env/bin/python \
  code/script/evaluate_kvp10k_benchmark.py \
  --prediction_dir data/outputs/stage3_mistral_final_inference/predictions \
  --ground_truth_dir data/prepared/test \
  --output data/outputs/stage3_mistral_final_inference/evaluation_kvp10k_official.json

# Final Stage 3 by frozen annotation-geometry cluster
PYTHONPATH=code/script env/kvp10k_env/bin/python \
  code/script/evaluate_kvp10k_benchmark.py \
  --prediction_dir data/outputs/stage3_mistral_final_inference/predictions \
  --ground_truth_dir data/prepared/test \
  --cluster_map data/outputs/stage2/test_cluster_map.json \
  --output data/outputs/stage3_mistral_final_inference/evaluation_kvp10k_official_per_cluster.json

# Final V4
PYTHONPATH=code/script env/kvp10k_env/bin/python \
  code/script/evaluate_kvp10k_benchmark.py \
  --prediction_dir data/outputs/stage4b_v4/predictions \
  --ground_truth_dir data/prepared/test \
  --cluster_map data/outputs/stage2/test_cluster_map.json \
  --output data/outputs/stage4b_v4/evaluation_kvp10k_official.json
```

`evaluate_stage4b.py` retains legacy pooled diagnostic metrics. Headline
benchmark comparisons use `evaluate_kvp10k_benchmark.py`, which implements
type-aware page-macro evaluation. Do not use `measurements.json` or legacy
pooled scores as headline results.

---

## Key Files

| Component | File | Purpose |
|-----------|------|---------|
| Facts | `LaTeX_Thesis/THESIS_FACTSHEET.md` | Authoritative values and settings |
| Preparation | `code/script/prepare_data.py` | PDF text and annotations to prepared JSON |
| Stage 2 | `code/script/features.py` | Annotation-geometry features and clustering |
| Stage 3 training | `code/script/mistral_baseline.py` | Local Mistral QLoRA training |
| Stage 3 final inference | `code/script/stage3_mistral_final_inference.py` | Final inference, saved raw output, and type-aware parsing |
| Stage 3 parser tests | `code/script/test_stage3_mistral_final_inference.py` | CPU-only parser and entry-point parity tests |
| Official evaluation | `code/script/kvp10k_official_eval.py` | Benchmark-compatible matching and aggregation |
| Evaluation CLI | `code/script/evaluate_kvp10k_benchmark.py` | Evaluation of saved predictions |
| V1 model | `code/script/layoutlm_model.py` | Token-level projected relation scorer |
| V2/V4 model | `code/script/layoutlm_model_v2.py` | Span-level projected relation scorer |
| V4 training | `code/script/train_stage4b_v5.py` | Final corrected training and full-state resumption |

---

## Reproducibility Notes

- `data/outputs/`, prepared data, model checkpoints, raw responses, and full
  prediction directories are normally ignored because they are generated or
  environment-specific.
- The small authoritative final Stage 3 evaluation summaries are tracked even
  though their parent output directory is ignored.
- Exact GPU inference requires the corresponding local checkpoint. Saved
  evaluation summaries can be inspected without the checkpoint.
- Stage 4a and legacy pooled metrics are diagnostic only. Do not report them as
  final benchmark results.

---

## References

- **KVP10k / IBM baseline**: [ICDAR 2024](https://huggingface.co/datasets/ibm/KVP10k)
- **KVP10k released code and configuration**: [IBM/KVP10k](https://github.com/IBM/KVP10k)
- **LayoutLMv3**: Huang et al., 2022 — [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base)
- **Biaffine relation extraction**: Dozat & Manning, 2017
