#!/bin/bash
#SBATCH --job-name=kvp_eval
#SBATCH --output=logs/kvp_eval-%j.out
#SBATCH --error=logs/kvp_eval-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$HOME/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/woody/iwi5/iwi5413h/kvp10k_thesis
PYTHON_BIN="/home/woody/iwi5/iwi5413h/kvp10k_thesis/env/kvp10k_env/bin/python"

CHECKPOINT_DIR="${1:-data/outputs/stage4b_canary_B}"

echo "=== Stage 4b Evaluation ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Date:     $(date)"

"$PYTHON_BIN" code/script/evaluate_stage4b.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --data_dir data/prepared \
  --batch_size 1 \
  --score_threshold 0.5

echo "=== Evaluation DONE ==="
