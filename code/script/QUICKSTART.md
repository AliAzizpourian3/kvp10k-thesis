# Quick Start

This file is the shortest possible guide to the current project state.

## Where We Are

- Stage 0 exists and defines the evaluation protocol.
- Stage 1 exists and covers dataset ingestion.
- Stage 2 is done.
- Stage 3 heuristic baselines are done.
- Stage 3 Mistral is complete (prepared data, training, prediction, evaluation).
- Stage 4 is next (post-Stage-3).

Stage 0 and Stage 1 were not removed from the code. They were left out of the shortened doc version because the current operational bottleneck is Stage 3.

## Important Cluster Rule

Do not run Stage 3 preparation on a compute node.

Reason:
- compute nodes have no internet access
- PDF download only works from the login node

## Current Stage 3 Workflow

### 1. Prepare Data On Login Node
Run from `code/script/`:

```bash
nohup bash /home/woody/iwi5/iwi5413h/kvp10k_thesis/logs/stage3_prepare_data.sbatch > /home/woody/iwi5/iwi5413h/kvp10k_thesis/logs/prepare_data.out 2>&1 &
```

This creates:
- `data/prepared/train/*.json`
- `data/prepared/test/*.json`

### 2. Submit Mistral GPU Job After Preparation Finishes
Run from the project root:

```bash
cd /home/woody/iwi5/iwi5413h/kvp10k_thesis
sbatch logs/stage3_mistral.sbatch
```

This job does:
- training
- prediction
- evaluation

## Current Snapshot

### Test Preparation
- complete
- 581 prepared pages out of 1051 unique test pages

### Train Preparation
- complete
- 5389 prepared pages out of 9656 unique train pages

## Main Files To Know


## Documentation Layout

The markdown docs in `code/script/` have been reduced on purpose.

- `QUICKSTART.md`: short operational guide
- `IMPLEMENTATION_SUMMARY.md`: detailed status, rationale, and key file map

Older split-out docs were collapsed because they were repeating the same Stage 3 status in slightly different words.

## What To Do Next

Stage 3 is complete.

- Results are reported in `LaTeX_Thesis/chapters/07_results.tex` (including coverage explanation and the default vs Stage-0 threshold tables).
- Next work: proceed to Stage 4 (LayoutLMv3 baseline / linker), then robustness and ablations.
