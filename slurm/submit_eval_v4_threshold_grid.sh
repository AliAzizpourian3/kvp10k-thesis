#!/bin/bash
#SBATCH --job-name=kvp_eval_v4_grid
#SBATCH --output=logs/kvp_eval_v4_grid-%j.out
#SBATCH --error=logs/kvp_eval_v4_grid-%j.err
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

echo "=== V4 Eval — threshold grid completion ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date)"

# Complete the NED x IoU grid. Existing runs already cover (0.2,0.3) and (0.5,0.5).
# These two add the off-diagonal cells:
#   (0.5,0.3) = relax text only  -> text/OCR tolerance
#   (0.2,0.5) = tighten IoU only  -> localization sensitivity
# Each call runs inference once and reports BOTH text-only and text+bbox link F1.
for SETTING in "0.5 0.3" "0.2 0.5"; do
  read -r NED IOU <<< "$SETTING"
  echo ""
  echo "--- V4 (V2 model), NED<=${NED}, IoU>=${IOU}, score_thr=0.5 ---"
  "$PYTHON_BIN" code/script/evaluate_stage4b.py \
    --checkpoint_dir data/outputs/stage4b_v4 \
    --data_dir data/prepared \
    --batch_size 1 \
    --score_threshold 0.5 \
    --ned_thresh "$NED" \
    --iou_thresh "$IOU" \
    --model_version v2
done

echo ""
echo "=== ALL DONE $(date) ==="
