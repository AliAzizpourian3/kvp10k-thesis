#!/bin/bash
#SBATCH --job-name=kvp_eval_v3
#SBATCH --output=logs/kvp_eval_v3-%j.out
#SBATCH --error=logs/kvp_eval_v3-%j.err
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

echo "=== V3 Evaluation (linker-only model) ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date)"

for THRESH in 0.0 0.1 0.3 0.5; do
  echo ""
  echo "--- V3 best_model, threshold=$THRESH ---"
  "$PYTHON_BIN" code/script/evaluate_stage4b.py \
    --checkpoint_dir data/outputs/stage4b_v3 \
    --data_dir data/prepared \
    --batch_size 1 \
    --score_threshold "$THRESH" \
    --model_version v2
done

echo ""
echo "=== ALL DONE ==="
