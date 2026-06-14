#!/bin/bash
#SBATCH --job-name=kvp_diag
#SBATCH --output=logs/kvp_diag-%j.out
#SBATCH --error=logs/kvp_diag-%j.err
#SBATCH --time=00:30:00
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
PYTHON_BIN="env/kvp10k_env/bin/python"

echo "=== Link Score Diagnostic ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date)"

echo ""
echo "--- V2+TF (patience 3, best_model) ---"
"$PYTHON_BIN" code/script/diagnose_link_scores.py \
  --checkpoint_dir data/outputs/stage4b_v2_tf \
  --data_dir data/prepared \
  --max_samples 200

echo ""
echo "=== DONE ==="
