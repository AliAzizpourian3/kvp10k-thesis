#!/bin/bash
#SBATCH --job-name=kvp_v3
#SBATCH --output=logs/kvp_v3-%j.out
#SBATCH --error=logs/kvp_v3-%j.err
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

# Use the V2+TF patience-7 best model as base
CHECKPOINT="data/outputs/stage4b_v2_tf_p7/best_model/pytorch_model.bin"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: Checkpoint not found at $CHECKPOINT"
  exit 1
fi

echo "=== Stage 4b V3: Linker-Only Training ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date)"
echo "Checkpoint: $CHECKPOINT"
echo "Strategy: Freeze encoder+entity_classifier, train linker with predicted entities"
echo "Key changes: lr=1e-4, linker_loss_weight=5.0, patience=10, batch=2, grad_accum=4"

"$PYTHON_BIN" code/script/train_stage4b_v3.py \
  --data_dir            data/prepared \
  --output_dir          data/outputs/stage4b_v3 \
  --checkpoint          "$CHECKPOINT" \
  --batch_size          2 \
  --gradient_accumulation_steps 4 \
  --learning_rate       1e-4 \
  --num_epochs          30 \
  --early_stopping_patience 10 \
  --linker_loss_weight  5.0

echo "=== DONE ==="
