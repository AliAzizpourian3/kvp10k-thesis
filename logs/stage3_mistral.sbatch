#!/usr/bin/env bash
#SBATCH --job-name=kvp_stage3_mistral
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/woody/iwi5/iwi5413h/kvp10k_thesis/logs/%x-%j.out
#
# Stage 3 – Phase 2: Mistral-7B LoRA Training + Prediction
# Requires prepared data from Phase 1 (stage3_prepare_data.sbatch).
# GPU job on A100.
#
# Usage:  sbatch logs/stage3_mistral.sbatch
#
set -euo pipefail
cd "$WORK/kvp10k_thesis/code/script"

export HF_HOME="$WORK/kvp10k_thesis/hf_cache"
export HF_DATASETS_CACHE="$WORK/kvp10k_thesis/hf_cache"
export TRANSFORMERS_CACHE="$WORK/kvp10k_thesis/hf_cache"
export KVP10K_HF_CACHE="$WORK/kvp10k_thesis/hf_cache"
export KVP10K_OUTPUT_DIR="$WORK/kvp10k_thesis/data/outputs"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python_bin="$WORK/kvp10k_thesis/env/kvp10k_env/bin/python"
prepared_dir="$WORK/kvp10k_thesis/data/prepared"
output_dir="$WORK/kvp10k_thesis/data/outputs/stage3_mistral"

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
    echo "       Run stage3_prepare_data.sbatch first."
    exit 1
fi

echo "=== Phase 2a: Training ==="
# Auto-detect latest checkpoint for resume
latest_ckpt=$(ls -d "$output_dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
resume_flag=""
if [ -n "$latest_ckpt" ] && [ -f "$latest_ckpt/trainer_state.json" ]; then
    echo "Resuming from $latest_ckpt"
    resume_flag="--resume_from_checkpoint $latest_ckpt"
fi

"$python_bin" mistral_baseline.py train \
    --data_dir "$prepared_dir" \
    --output_dir "$output_dir" \
    $resume_flag

echo ""
echo "=== Phase 2b: Prediction ==="
"$python_bin" mistral_baseline.py predict \
    --checkpoint "$output_dir/checkpoint" \
    --data_dir "$prepared_dir" \
    --output_dir "$output_dir/predictions"

echo ""
echo "=== Phase 2c: Evaluation ==="
"$python_bin" evaluate_mistral.py \
    --pred_dir "$output_dir/predictions" \
    --gt_dir "$prepared_dir/test" \
    --output "$output_dir/evaluation.json"

echo ""
echo "=== Summary ==="
echo "Predictions:" $(ls "$output_dir/predictions/"*.json 2>/dev/null | wc -l)
echo "DATE: $(date)"
echo "DONE"
