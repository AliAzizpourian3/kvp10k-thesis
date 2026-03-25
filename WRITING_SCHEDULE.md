# Writing Schedule While Training Runs

## ✅ What We've Just Created (Ready to Integrate)

### 1. **STAGE4_METHODOLOGY_DRAFT.tex** — Stage 4 Chapter Complete
📄 **File**: `/home/woody/iwi5/iwi5413h/kvp10k_thesis/STAGE4_METHODOLOGY_DRAFT.tex`

**What's in it:**
- ✅ Overall architecture (LayoutLMv3 + Entity Classifier + Biaffine Linker)
- ✅ Why LayoutLMv3 specifically (4 detailed justifications with citations)
- ✅ Stage 4a methodology (entity classification, loss functions, label generation from ground truth)
- ✅ Stage 4b methodology (biaffine attention, relation loss, λ sweep justification)
- ✅ Data preprocessing pipeline (lmdx_text parsing, coordinate normalization, processor config)
- ✅ Token alignment procedure (word-to-token label mapping)
- ✅ Training configuration (optimizer, batch size, learning rate schedule)
- ✅ Inference and postprocessing (entity prediction, relation scoring, bbox inheritance)
- ✅ Ablation study design section

**How to use it:**
1. Copy the content
2. Insert into `LaTeX_Thesis/chapters/03_methodology.tex` after line 20 (after "Overview" section)
3. LaTeX will auto-generate \ref{} and citation links

**Estimated integration time:** 15 minutes

---

### 2. **CHAPTER7_RESULTS_TEMPLATE.tex** — Results Chapter Structure
📄 **File**: `/home/woody/iwi5/iwi5413h/kvp10k_thesis/CHAPTER7_RESULTS_TEMPLATE.tex`

**What's in it:**
- ✅ Complete section structure (Stage 3 overall, by cluster, error analysis)
- ✅ Table templates with placeholder \texttt{[TBD]} for numbers
- ✅ Interpretation narratives (written in past tense, ready to fill with numbers)
- ✅ Ablation analysis sections (ready for results)
- ✅ Summary section connecting to research questions

**How to use it:**
1. Copy sections corresponding to what's available (Stage 3 is complete; keep Stage 4 as template)
2. Replace \texttt{[TBD]} with actual numbers when training completes
3. Tables can be generated directly from evaluation JSON files

**Estimated integration time:** 30 minutes (now), 30 minutes (later to fill numbers)

**Estimated fill time** (when training done): 30 minutes

---

### 3. **IMPLEMENTATION_SUMMARY.md Updated** — Stage 4 Documentation
📄 **File**: `/home/woody/iwi5/iwi5413h/kvp10k_thesis/code/script/IMPLEMENTATION_SUMMARY.md`

**What was added:**
- ✅ Complete Stage 4 dataset fix explanation (root cause, solution, verification)
- ✅ Data processing pipeline walkthrough with JSON examples
- ✅ Verification results table (multi-sample consistency check)
- ✅ Job submission status and tracking
- ✅ Expected training signals and success criteria
- ✅ Key learnings and immediate next steps

**Status:** Already integrated into main implementation documentation

---

## 📝 Priority Writing Tasks (NO RESULTS NEEDED)

### **HIGH PRIORITY** — Do This Now (Next 2 hours)

#### Task 1: Integrate Stage 4 Methodology Section (30 min)
**Action**: Add `STAGE4_METHODOLOGY_DRAFT.tex` to `LaTeX_Thesis/chapters/03_methodology.tex`

**Why now**: 
- Completely writeable (architecture, not results)
- High value for thesis completeness
- No dependencies on training

**Effort**: Copy + paste + verify LaTeX compiles

---

#### Task 2: Complete/Polish Chapter 2 — LayoutLM Family Literature (1-2 hours)
**Current status**: Partially complete. Missing:
- ✅ LayoutLMv1, v2, v3 overview (mostly done)
- ⚠️ Biaffine attention in NLP (brief mention needed)
- ⚠️ Generative IE vs discriminative IE trade-offs (incomplete)
- ⚠️ KVP10k dataset details (exists but could expand)

**What to write**:
```
\subsection{Biaffine Attention for Relation Extraction}
Include: origin in dependency parsing (Dozat & Manning 2017), 
adaptation to IE, why chosen for Stage 4 linker
(2-3 paragraphs, ~400 words)

\subsection{Generative vs Discriminative Extraction}
Include: trade-offs (hallucination vs grounding), 
comparison to Stage 3 motivation
(2 paragraphs, ~300 words)

\section{Summary: Research Gaps}
Already present but could expand with Stage 4-specific gaps
```

**Estimate**: 1-2 hours (mostly filling existing TODO stubs)

---

#### Task 3: Create Results Chapter Skeleton (45 min)
**Action**: Copy `CHAPTER7_RESULTS_TEMPLATE.tex` into `LaTeX_Thesis/chapters/07_results.tex`

**Why now**: 
- Structure is ready
- Interpretation narrative is written
- Just need numbers later

**Effort**: Copy + verify tables render + set up \ref{} labels

---

### **MEDIUM PRIORITY** — Do When Training Starts (3-4 hours)

#### Task 4: Detailed Error Analysis Writeup for Chapter 8 (Discussion)
**What to write**:
- Interpret Stage 3 hallucination patterns by cluster
- Hypothesize about why sparse layouts fail (lack of global spatial context?)
- Discuss why discriminative approaches might help
- Compare to related work (PICK, other graph-based models)

**When**: After Stage 3 results are finalized (already have them!)
**Estimate**: 2-3 hours

---

#### Task 5: Robustness & Limitations Section (Discussion, Ch. 8)
**What to write** (Template):
- Data format assumptions (lmdx_text, KVP10k structure)
- Model size limitations (Mistral-7B, LayoutLMv3-base vs large)
- Generalization concerns (business documents only, English-centric)
- Computational requirements

**When**: Can write now (design-level discussion)
**Estimate**: 1 hour

---

### **LOWER PRIORITY** — Wait for Results (Post-Training)

#### Task 6: Fill Numbers in Chapter 7 Tables (30 min/table)
**When**: After training completes and evaluation runs
**Effort**: Extract from evaluation JSON + paste into LaTeX tables

---

#### Task 7: Write Detailed Per-Cluster Comparison (1-2 hours)
**When**: After ablation results are available
**Content**: Detailed error analysis figure captions, interpretations of λ sweep

---

## 📊 Current File Status

### Ready to Use
```
✅ STAGE4_METHODOLOGY_DRAFT.tex       → Insert into 03_methodology.tex
✅ CHAPTER7_RESULTS_TEMPLATE.tex      → Insert into 07_results.tex (update later)
✅ IMPLEMENTATION_SUMMARY.md          → Complete, reference as needed
✅ STAGE4_DATASET_FIX.md             → Already created
✅ STAGE4_READY_FOR_TRAINING.md      → Already created
```

### Need Minor Additions
```
⏳ LaTeX_Thesis/chapters/02_literature_review.tex
   - Add: Biaffine attention section (300 words)
   - Add: Generative vs discriminative section (300 words)
   - Expand: KVP10k dataset details (200 words)
   - Effort: 1-2 hours
```

### Will Update When Training Completes
```
🟡 LaTeX_Thesis/chapters/07_results.tex
   - Fill: All \texttt{[TBD]} placeholders with numbers
   - Update: Interpretation sections with actual findings
   - Effort: 45 minutes per stage (Stage 3 done, Stage 4 pending)
```

---

## ⏱️ Recommended Schedule

### **Tonight (Right Now) — 3 hours**
1. (30 min) Integrate Stage 4 Methodology → compile LaTeX to verify
2. (90 min) Polish Chapter 2 — LayoutLM family + Biaffine attention
3. (45 min) Create Results chapter skeleton → set up table formatting

### **Tomorrow Morning — Check Job Status → 1 hour**
- Jobs should be starting around 18:21 UTC today
- Check logs for entity loss trajectory (should drop by epoch 2)
- Start monitoring evaluation outputs

### **While Training Runs — 2-3 hours**
1. Write full error analysis section (Discussion chapter)
2. Write robustness & limitations section
3. Polish introduction and related work connections

### **After Training Completes — 1-2 hours**
1. Fill Chapter 7 result tables
2. Verify all references and citations
3. Compile full thesis PDF
4. Final polish

---

## 🎯 What You'll Have by End of Week

If you follow this plan and training completes in 24 hours:

```
✅ Chapter 1: Introduction
✅ Chapter 2: Literature Review (COMPLETE — with biaffine + generative/discriminative)
✅ Chapter 3: Methodology (COMPLETE — with full Stage 4 section)
✅ Chapter 4: Dataset (existing)
✅ Chapter 5: Implementation (existing)
✅ Chapter 6: Experiments (existing)
✅ Chapter 7: Results (DRAFT — numbers filled when training done)
✅ Chapter 8: Discussion (DRAFT — error analysis + robustness)
✅ Chapter 9: Conclusion (to be written)

= Full first draft of thesis, ready for advisor review
```

---

## 📋 Files to Reference While Writing

| Document | Purpose | Location |
|----------|---------|----------|
| Literature Review (complete) | Citation style, thesis voice | `LaTeX_Thesis/chapters/02_literature_review.tex` |
| Methodology (existing) | Structure, formatting | `LaTeX_Thesis/chapters/03_methodology.tex` |
| BibTeX | All available citations | `LaTeX_Thesis/references.bib` |
| Implementation details | Stage 4 technical facts | `code/script/IMPLEMENTATION_SUMMARY.md` |
| KVP10k paper | Baseline metrics, dataset details | `KVP10k.pdf` |
| Dataset spec | lmdx_text format, coordinate system | `TECHNICAL_LMDX_SPECIFICATION.md` |

---

## ✨ Key Insight

**You can write 80% of the thesis RIGHT NOW without waiting for results.** 

The only things that depend on training completion:
- Exact numbers in result tables (30 minutes to fill)
- Detailed ablation interpretations (1-2 hours)
- Detailed error analysis narratives (already mostly writeable)

**Total value of writing now:** 
- 4-5 hours → ~40% complete thesis draft
- **Zero risk** of losing information or context
- **High benefit** of having solid draft before final results arrive

**Recommendation**: Write tonight. By tomorrow morning, you'll have a substantial, polished thesis draft that just needs numbers filled in.
