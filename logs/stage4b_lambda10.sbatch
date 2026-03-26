#!/usr/bin/env bash
#SBATCH --job-name=kvp_stage4b_l10
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/woody/iwi5/iwi5413h/kvp10k_thesis/logs/%x-%j.out
#
# Stage 4b – Phase B2: LayoutLMv3 with Biaffine Linker (λ=1.0)
# Key-value linking with linker loss weight = 1.0 (balanced baseline)
# Runs AFTER Stage 4b λ=0.5 completes (queued to run sequentially).
# GPU job on A100.
#
# Usage:  sbatch logs/stage4b_lambda10.sbatch
#
set -euo pipefail
cd "$WORK/kvp10k_thesis/code/script"

export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export KVP10K_HF_CACHE="$HOME/.cache/huggingface"
export KVP10K_OUTPUT_DIR="$WORK/kvp10k_thesis/data/outputs"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python_bin="$WORK/kvp10k_thesis/env/kvp10k_env/bin/python"
prepared_dir="$WORK/kvp10k_thesis/data/prepared"
output_dir="$WORK/kvp10k_thesis/data/outputs/stage4b_lambda10"

echo "HOST: $(hostname)"
echo "DATE: $(date)"

echo "=== GPU check ==="
nvidia-smi -L
"$python_bin" -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== Verify prepared data ==="
echo "Train samples:" $(ls "$prepared_dir/train/"*.json 2>/dev/null | wc -l)
echo "Test samples:"  $(ls "$prepared_dir/test/"*.json  2>/dev/null | wc -l)

if [ ! -d "$prepared_dir/train" ]; then
    echo "ERROR: Prepared data not found at $prepared_dir/train"
    exit 1
fi

echo "=== Stage 4b Training (λ=1.0) ==="
echo "Configuration:"
echo "  Model: LayoutLMv3 with Biaffine Linker"
echo "  Linker loss weight: 1.0"
echo "  Batch size: 4"
echo "  Gradient accumulation: 8 (effective batch: 32)"
echo "  Learning rate: 5e-5"
echo "  Epochs: 10"
echo "  Early stopping: patience=3 on validation F1"
echo "  Output: $output_dir"
echo ""

mkdir -p "$output_dir"

"$python_bin" train_stage4b.py \
    --data_dir "$prepared_dir" \
    --output_dir "$output_dir" \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_epochs 10 \
    --early_stopping_patience 3 \
    --val_fraction 0.1 \
    --linker_loss_weight 1.0 \
    --include_images \
    --pretrained_encoder "$WORK/kvp10k_thesis/data/outputs/stage4a/checkpoint-9/model.pt" \
    --resume_from_checkpoint

echo ""
echo "=== Stage 4b (λ=1.0) Training Complete ==="
echo "Results saved in: $output_dir"
