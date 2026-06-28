#!/bin/bash
#SBATCH --job-name=kvp_sweep_v4
#SBATCH --output=logs/kvp_sweep_v4-%j.out
#SBATCH --error=logs/kvp_sweep_v4-%j.err
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

echo "=== V4 link-score threshold sweep on held-out VAL slice (rtx3080) ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date)"

# Tune the decision threshold on the seed-42 10% val slice held out from linker
# training (no test leakage). Sweep under BOTH protocols.
for SETTING in "0.2 0.3" "0.5 0.5"; do
  read -r NED IOU <<< "$SETTING"
  echo ""
  echo "--- sweep VAL, NED<=${NED}, IoU>=${IOU} ---"
  "$PYTHON_BIN" code/script/sweep_link_threshold.py \
    --checkpoint_dir data/outputs/stage4b_v4 \
    --data_dir data/prepared \
    --split val \
    --model_version v2 \
    --batch_size 1 \
    --ned_thresh "$NED" \
    --iou_thresh "$IOU"
done

echo ""
echo "=== ALL DONE $(date) ==="
