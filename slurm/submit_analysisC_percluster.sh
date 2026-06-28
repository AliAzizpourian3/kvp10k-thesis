#!/bin/bash
#SBATCH --job-name=analysisC_percluster
#SBATCH --output=logs/analysisC_percluster-%j.out
#SBATCH --error=logs/analysisC_percluster-%j.err
#SBATCH --time=01:30:00
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

echo "=== Analysis C: V4 per-cluster evaluation (rtx3080) ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date)"

# Thesis protocol: NED<=0.2, IoU>=0.3
"$PYTHON_BIN" code/script/evaluate_stage4b_per_cluster.py \
  --checkpoint_dir data/outputs/stage4b_v4 \
  --data_dir data/prepared \
  --cluster_map data/outputs/stage2/test_cluster_map.json \
  --model_version v2 \
  --score_threshold 0.5 \
  --ned_thresh 0.2 \
  --iou_thresh 0.3

echo ""
echo "=== ALL DONE $(date) ==="
