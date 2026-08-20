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
│   ├── 03_dataset.tex
│   ├── 04_methodology.tex
│   ├── 05_implementation.tex
│   ├── 06_experiments.tex
│   ├── 07_results.tex
│   ├── 07_discussion.tex
│   ├── 08_conclusion.tex
│   └── appendix_a_code.tex
└── figures/                    # Thesis figures
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

The repository contains a complete supervisor-review draft. V4 is the final
corrected LayoutLMv3 experiment. The
verified numerical source is `THESIS_FACTSHEET.md`; older root README files and
legacy pooled measurements are not authoritative for benchmark results.

The normal local build command is:

```bash
latexmk -pdf main.tex
```

## Notes

- All `% TODO:` comments mark sections needing content
- Dataset chapter (04) already contains real results from analysis
- Keep updating as experiments progress
- Add figures to `figures/` directory as they're generated
