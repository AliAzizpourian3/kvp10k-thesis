#!/bin/bash
#SBATCH --job-name=kvp_v2tf_p7
#SBATCH --output=logs/kvp_v2tf_p7-%j.out
#SBATCH --error=logs/kvp_v2tf_p7-%j.err
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

PRETRAINED_CKPT="data/outputs/stage4b_canary_B/best_model/pytorch_model.bin"

if [[ ! -f "$PRETRAINED_CKPT" ]]; then
  echo "ERROR: Pretrained checkpoint not found at $PRETRAINED_CKPT"
  exit 1
fi

echo "=== Stage 4b V2 + Teacher Forcing (patience=7) ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:     $(date)"
echo "Pretrained: $PRETRAINED_CKPT"
echo "Changes: teacher forcing + patience 7 + separate ent/link loss logging"

"$PYTHON_BIN" code/script/train_stage4b_v2.py \
  --data_dir            data/prepared \
  --output_dir          data/outputs/stage4b_v2_tf_p7 \
  --batch_size          1 \
  --gradient_accumulation_steps 8 \
  --learning_rate       2e-5 \
  --num_epochs          20 \
  --early_stopping_patience 7 \
  --linker_loss_weight  1.0 \
  --pretrained_encoder  "$PRETRAINED_CKPT"

echo "=== DONE ==="
