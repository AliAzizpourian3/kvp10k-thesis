#!/usr/bin/env bash
#SBATCH --job-name=kvp_stage4b_l05
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/woody/iwi5/iwi5413h/kvp10k_thesis/logs/%x-%j.out
#
# Stage 4b - Phase B1: LayoutLMv3 with Biaffine Linker (lambda=0.5)
# Stability-first defaults: lr=2e-5, accum=8, batch=1
#
# Usage:  sbatch logs/stage4b_lambda05.sbatch
#
set -euo pipefail
cd /home/woody/iwi5/iwi5413h/kvp10k_thesis

export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export KVP10K_HF_CACHE="$HOME/.cache/huggingface"
export KVP10K_OUTPUT_DIR="/home/woody/iwi5/iwi5413h/kvp10k_thesis/data/outputs"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="/home/woody/iwi5/iwi5413h/kvp10k_thesis/env/kvp10k_env/bin/python"
prepared_dir="/home/woody/iwi5/iwi5413h/kvp10k_thesis/data/prepared"
output_dir="/home/woody/iwi5/iwi5413h/kvp10k_thesis/data/outputs/stage4b_lambda05"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "Python: $PYTHON_BIN"
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found at $PYTHON_BIN"; exit 1; }

echo "=== GPU check ==="
nvidia-smi -L
"$PYTHON_BIN" -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== Verify prepared data ==="
echo "Train samples:" $(ls "$prepared_dir/train/"*.json 2>/dev/null | wc -l)
echo "Test samples:"  $(ls "$prepared_dir/test/"*.json  2>/dev/null | wc -l)

[ -d "$prepared_dir/train" ] || { echo "ERROR: Prepared data not found at $prepared_dir/train"; exit 1; }

STAGE4A_DIR="/home/woody/iwi5/iwi5413h/kvp10k_thesis/data/outputs/stage4a"
STAGE4A_CKPT=$(find "$STAGE4A_DIR" -name 'pytorch_model.bin' -o -name 'model.pt' 2>/dev/null | sort -rV | head -1)
[ -n "$STAGE4A_CKPT" ] || { echo "ERROR: No Stage 4a checkpoint found in $STAGE4A_DIR"; exit 1; }
echo "Stage 4a checkpoint: $STAGE4A_CKPT"

echo "=== Stage 4b Training (lambda=0.5) ==="
mkdir -p "$output_dir"

"$PYTHON_BIN" code/script/train_stage4b.py \
    --data_dir "$prepared_dir" \
    --output_dir "$output_dir" \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --early_stopping_patience 3 \
    --val_fraction 0.1 \
    --linker_loss_weight 0.5 \
    --pretrained_encoder "$STAGE4A_CKPT"

echo "=== Stage 4b (lambda=0.5) Training Complete ==="
echo "Results saved in: $output_dir"
