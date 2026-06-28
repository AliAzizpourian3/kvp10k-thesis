#!/bin/bash
#SBATCH --job-name=kvp_oracle_v4
#SBATCH --output=logs/kvp_oracle_v4-%j.out
#SBATCH --error=logs/kvp_oracle_v4-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
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

echo "=== Analysis B: ORACLE linking (GT entity spans) on TEST (rtx3080) ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date)"

# Oracle linking = upper bound on link F1 given perfect entity detection.
# Compare directly against the honest end-to-end numbers under both protocols.
for SETTING in "0.2 0.3" "0.5 0.5"; do
  read -r NED IOU <<< "$SETTING"
  echo ""
  echo "--- ORACLE TEST, NED<=${NED}, IoU>=${IOU} ---"
  "$PYTHON_BIN" code/script/evaluate_stage4b.py \
    --checkpoint_dir data/outputs/stage4b_v4 \
    --data_dir data/prepared \
    --batch_size 1 \
    --score_threshold 0.5 \
    --ned_thresh "$NED" \
    --iou_thresh "$IOU" \
    --model_version v2 \
    --oracle
done

echo ""
echo "=== ALL DONE $(date) ==="
