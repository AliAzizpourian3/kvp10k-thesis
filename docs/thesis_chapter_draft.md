# Thesis Chapter Draft: Key-Value Pair Linking
# ==============================================

## Suggested Chapter: "Entity Linking for Key-Value Pair Extraction"

### 4.X.1  Problem Statement
- After entity classification (Stage 4a): tokens labeled as Key / Value / Other
- Remaining challenge: which key belongs to which value?
- This is a **graph construction** problem over document entities

### 4.X.2  Related Work on Entity Linking in Documents
- **SPADE** (Hwang et al., ACL Findings 2021, arXiv:2005.00642)
  - Segment-level dependency parsing with biaffine attention
  - Operates on ~20-50 OCR segments, NOT individual tokens
  - Two relation types: within-group links and cross-group (key→value) links
  - Key insight: treating it as segment-to-segment reduces the search space dramatically

- **BROS** (Hong et al., AAAI 2022, arXiv:2108.04539)
  - Initial-token linking: only the FIRST token of each entity span participates
  - SpanLM pretraining with area-masking for layout awareness
  - Reduces ~200 tokens → ~30 initial-token candidates

- **KVP10k** (Friedman et al., ICDAR 2024, arXiv:2405.00505)
  - Dataset: 10,707 documents, diverse templates
  - Baseline: Mistral-7B with LMDX prompting
  - Regular KVP text-only F1 = 0.659 (Table 1)

### 4.X.3  Approach V1: Token-Level Biaffine Linker
- **Architecture**: BiaffineLinker on individual tokens
  - Key/value projections (768 → 128 dim)
  - Spatial feature encoder: 8-d bbox features → 64 → 32 MLP
  - Combined scorer: dot-product + spatial = link logit
  - Chunked processing (chunk_size=8) for memory efficiency
- **Training**: BCE loss with pos_weight = min(n_neg/n_pos, 50)
- **Result**: Entity F1 ≈ 0.847, **Link F1 ≈ 0.008**
- **Failure analysis**:
  - ~100 key tokens × ~100 value tokens = 10,000-entry matrix per document
  - Only ~3-10 positive pairs → **~450:1 neg:pos class imbalance**
  - Even with pos_weight, token-level linking is too fine-grained
  - SPADE/BROS literature confirms: span/segment-level is the standard approach

### 4.X.4  Approach V2: Span-Level Biaffine Linker (SPADE-Inspired)
- **Key innovation**: group contiguous same-label tokens into spans BEFORE linking
- **Span grouping algorithm**:
  1. Run entity classifier → per-token key/value predictions
  2. Group contiguous same-label tokens into spans: [(start, end), ...]
  3. Filter degenerate bboxes (zero-area spans)
  4. Typically: ~100 tokens → ~10-20 spans
- **Span representation**: mean-pool hidden states + union bbox
- **Linking**: same biaffine scorer, but on ~15×15 span matrix (vs 100×100 tokens)
- **Expected improvement**: ~60:1 → ~5:1 neg:pos ratio → much better signal

### 4.X.5  Experimental Results

| Model | Entity F1 | Link F1 | Link P | Link R | Notes |
|-------|-----------|---------|--------|--------|-------|
| KVP10k baseline (LMDX+Mistral) | — | 0.659 | 0.678 | 0.641 | Official, text-only |
| V1 token-level (no pos_weight) | 0.846 | 0.000 | — | — | Zero predictions |
| V1 token-level (pos_weight, ep1) | 0.824 | 0.008 | 0.007 | 0.010 | Over-predicting |
| V1 token-level (pos_weight, best) | 0.847 | TBD | TBD | TBD | Full training |
| **V2 span-level** | **TBD** | **TBD** | **TBD** | **TBD** | **In progress** |

### 4.X.6  Discussion
- Token-level linking fails due to granularity mismatch with the task structure
- Span-level linking aligns with the natural document structure (multi-word fields)
- Our approach is inspired by SPADE but simplified:
  - SPADE trains a separate segmenter; we reuse entity classifier predictions
  - SPADE has two relation types; we have a single key→value relation
- pos_weight alone insufficient: reduces false negatives but doesn't fix the
  fundamental combinatorial explosion of the token-level search space

### 4.X.7  Figures to Include
1. **Architecture diagram**: LayoutLMv3 → Entity Classifier → Span Grouping → Biaffine Linker
2. **Span grouping visualization**: tokens with entity labels → merged spans
3. **Linking matrix comparison**: V1 (100×100 sparse) vs V2 (15×15 dense)
4. **Training curves**: entity loss + link loss over epochs for V1 and V2
5. **Qualitative examples**: predicted key-value pairs on sample documents

---

## Other Thesis Chapters (Brief Outline)

### Chapter 1: Introduction
- Document understanding and KVP extraction problem
- Motivation: automating data extraction from semi-structured documents

### Chapter 2: Background & Related Work
- LayoutLM family (v1, v2, v3) — multimodal document understanding
- OCR + layout integration
- SPADE, BROS, GCN-based approaches
- KVP10k dataset and LMDX prompting baseline

### Chapter 3: Dataset & Preprocessing
- KVP10k: 10,707 documents, 5,389 train / 581 test
- LMDX text format: word positions with bounding boxes
- Normalization and tokenization pipeline

### Chapter 4: Methodology
- Stage 4a: Entity classification (key/value/other)
- Stage 4b V1: Token-level biaffine linker (this chapter draft above)
- Stage 4b V2: Span-level biaffine linker

### Chapter 5: Experiments
- Training setup (A100, hyperparameters, early stopping)
- Lambda sweep (linker loss weight)
- Entity F1 and Link F1 evaluation
- Ablation: token-level vs span-level linking

### Chapter 6: Results & Discussion
- Quantitative comparison table
- Qualitative analysis of predictions
- Error analysis: common failure modes

### Chapter 7: Conclusion & Future Work
- Summary of contributions
- Future: image features, multi-page documents, few-shot learning
