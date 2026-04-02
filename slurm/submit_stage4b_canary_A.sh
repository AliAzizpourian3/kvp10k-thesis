#!/bin/bash
#SBATCH --job-name=kvp_4b_canary_A
#SBATCH --output=logs/kvp_stage4b_canary_A-%j.out
#SBATCH --error=logs/kvp_stage4b_canary_A-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$HOME/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/woody/iwi5/iwi5413h/kvp10k_thesis
PYTHON_BIN="/home/woody/iwi5/iwi5413h/kvp10k_thesis/env/kvp10k_env/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter not found at $PYTHON_BIN"
  exit 1
fi

echo "=== Stage 4b CANARY A (original settings: lr=5e-5, accum=32) ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:     $(date)"
echo "Python:   $PYTHON_BIN"
echo "HF cache: $HF_HOME"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"

if [[ ! -d "${HUGGINGFACE_HUB_CACHE}/models--microsoft--layoutlmv3-base" ]]; then
  echo "ERROR: LayoutLMv3 not found in HF cache at ${HUGGINGFACE_HUB_CACHE}"
  exit 1
fi
echo "HF cache check: OK"

# ── Find best Stage 4a checkpoint ──────────────────────────────────────
STAGE4A_DIR="data/outputs/stage4a"
STAGE4A_CKPT=$(find "${STAGE4A_DIR}" -name 'pytorch_model.bin' -o -name 'model.pt' 2>/dev/null | sort -rV | head -1)

if [[ -z "$STAGE4A_CKPT" ]]; then
  echo "ERROR: No Stage 4a checkpoint found in ${STAGE4A_DIR}"
  exit 1
fi
echo "Stage 4a checkpoint: $STAGE4A_CKPT"

"$PYTHON_BIN" code/script/train_stage4b.py \
  --data_dir            data/prepared \
  --output_dir          data/outputs/stage4b_canary_A \
  --batch_size          1 \
  --gradient_accumulation_steps 32 \
  --learning_rate       5e-5 \
  --num_epochs          20 \
  --early_stopping_patience 3 \
  --linker_loss_weight  1.0 \
  --pretrained_encoder  "$STAGE4A_CKPT"

echo "=== Stage 4b CANARY A DONE ==="
