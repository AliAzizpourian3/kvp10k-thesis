#!/bin/bash
#SBATCH --job-name=v4_alltypes
#SBATCH --output=logs/v4_alltypes-%j.out
#SBATCH --error=logs/v4_alltypes-%j.err
#SBATCH --time=02:00:00
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

echo "=== V4 all-type decode (regular + recovered unvalued/unkeyed) ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date)"

PYTHONPATH=code/script "$PYTHON_BIN" code/script/export_v4_all_types.py \
  --checkpoint_dir data/outputs/stage4b_v4 \
  --data_dir data/prepared \
  --model_version v2 \
  --score_threshold 0.5 \
  --batch_size 1 \
  --output_dir data/outputs/stage4b_v4/predictions_alltypes

PYTHONPATH=code/script "$PYTHON_BIN" code/script/evaluate_kvp10k_benchmark.py \
  --prediction_dir data/outputs/stage4b_v4/predictions_alltypes \
  --ground_truth_dir data/prepared/test \
  --cluster_map data/outputs/stage2/test_cluster_map.json \
  --output data/outputs/stage4b_v4/evaluation_kvp10k_official_alltypes.json

echo "=== DONE $(date) ==="
