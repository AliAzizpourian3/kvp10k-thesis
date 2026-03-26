# KVP Extraction Thesis - LaTeX Files

This directory contains the LaTeX source files for the Master's thesis on Key-Value Pair Extraction from Business Documents.

## Structure

```
LaTeX_Thesis/
├── main.tex                    # Main document
├── references.bib              # Bibliography
├── chapters/                   # Chapter files
│   ├── 00_abstract.tex
│   ├── 00_acknowledgments.tex
│   ├── 01_introduction.tex
│   ├── 02_literature_review.tex
│   ├── 03_methodology.tex
│   ├── 04_dataset.tex
│   ├── 05_implementation.tex
│   ├── 06_experiments.tex
│   ├── 07_results.tex
│   ├── 08_discussion.tex
│   ├── 09_conclusion.tex
│   ├── appendix_a_code.tex
│   └── appendix_b_data.tex
└── figures/                    # Figures (to be added)
```

## Compiling

### On Overleaf
1. Upload all files to Overleaf
2. Set `main.tex` as the main document
3. Compile with XeLaTeX or pdfLaTeX

### Locally
```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk:
```bash
latexmk -pdf main.tex
```

## Current Status

### ✅ Complete Chapters
- **Abstract** - Template ready
- **Introduction** - Structure complete, content outlined
- **Dataset (Chapter 4)** - Fully written with results from Stage 2

### ⏳ In Progress / TODO
- **Literature Review (Chapter 2)** - Structure only, needs content
- **Methodology (Chapter 3)** - Structure only, needs implementation details
- **Implementation (Chapter 5)** - Partial, needs model details
- **Experiments (Chapter 6)** - Structure only, awaiting Stage 3-5 results
- **Results (Chapter 7)** - Structure only, awaiting experimental results
- **Discussion (Chapter 8)** - Structure only, awaiting analysis
- **Conclusion (Chapter 9)** - Template ready

### 📊 Figures Needed
- PCA visualization (from Stage 2)
- KV distance distribution (from Stage 2)
- Cluster distribution (from Stage 2)
- Model architecture diagram
- Attention visualizations
- Results tables and plots

## Update Strategy

As experiments progress:
1. **After Stage 3**: Update Chapter 6 (baselines)
2. **After Stage 4**: Update Chapter 3 & 5 (methodology & implementation)
3. **After Stage 5**: Update Chapter 7 (main results)
4. **After Stage 6**: Update Chapter 7 (robustness results)
5. **After Stage 7**: Update Chapter 8 (interpretability)
6. **After Stage 8**: Finalize all chapters

## Current Working Note

For the current operational status of Stage 3 before LaTeX integration, see:

- `STAGE3_STATUS_NOTE.md`

## Notes

- All `% TODO:` comments mark sections needing content
- Dataset chapter (04) already contains real results from analysis
- Keep updating as experiments progress
- Add figures to `figures/` directory as they're generated
