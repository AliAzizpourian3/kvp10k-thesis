#!/usr/bin/env bash
# Stage 3 – Phase 1: Data Preparation
# Downloads PDFs, extracts text (PyMuPDF), fuses with annotations.
#
# IMPORTANT: TinyGPU compute nodes have NO internet access.
# This script must be run on the LOGIN NODE (which has internet).
#
# Usage (login node):
#   nohup bash logs/stage3_prepare_data.sbatch > logs/prepare_data.out 2>&1 &
#   # or in a tmux session:
#   bash logs/stage3_prepare_data.sbatch
#
set -euo pipefail
cd "$WORK/kvp10k_thesis/code/script"

export HF_HOME="$WORK/kvp10k_thesis/hf_cache"
export HF_DATASETS_CACHE="$WORK/kvp10k_thesis/hf_cache"
export TRANSFORMERS_CACHE="$WORK/kvp10k_thesis/hf_cache"
export KVP10K_HF_CACHE="$WORK/kvp10k_thesis/hf_cache"
export KVP10K_OUTPUT_DIR="$WORK/kvp10k_thesis/data/outputs"

python_bin="$WORK/kvp10k_thesis/env/kvp10k_env/bin/python"

echo "HOST: $(hostname)"
echo "DATE: $(date)"

echo "=== Test split (1,051 unique pages) ==="
"$python_bin" prepare_data.py --split test --workers 8 \
    --output_dir "$WORK/kvp10k_thesis/data/prepared"

echo ""
echo "=== Train split (~9,656 unique pages) ==="
"$python_bin" prepare_data.py --split train --workers 8 \
    --output_dir "$WORK/kvp10k_thesis/data/prepared"

echo ""
echo "=== Summary ==="
echo "Test samples:"  $(ls "$WORK/kvp10k_thesis/data/prepared/test/"*.json 2>/dev/null | wc -l)
echo "Train samples:" $(ls "$WORK/kvp10k_thesis/data/prepared/train/"*.json 2>/dev/null | wc -l)
echo "DATE: $(date)"
echo "DONE"
