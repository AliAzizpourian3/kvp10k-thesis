#!/bin/bash
#SBATCH --job-name=kvp_v5
#SBATCH --output=logs/kvp_v5-%j.out
#SBATCH --error=logs/kvp_v5-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$HOME/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/woody/iwi5/iwi5413h/kvp10k_thesis

PYTHON_BIN="/home/woody/iwi5/iwi5413h/kvp10k_thesis/env/kvp10k_env/bin/python"
OUTPUT_DIR="data/outputs/stage4b_v5"
WARM_START="data/outputs/stage4b_canary_B/best_model/pytorch_model.bin"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d data/prepared/train || ! -d data/prepared/test ]]; then
  echo "ERROR: Prepared train/test data not found" >&2
  exit 1
fi
if [[ ! -f "$WARM_START" ]]; then
  echo "ERROR: Canary B warm-start checkpoint not found: $WARM_START" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" && "$*" != *"--resume_from_checkpoint"* ]]; then
  echo "ERROR: Refusing to overwrite existing output: $OUTPUT_DIR" >&2
  echo "For full-state resume, pass --resume_from_checkpoint auto or an explicit checkpoint directory." >&2
  exit 1
fi

echo "=== Stage 4b V5: corrected official-pair checkpoint selection ==="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURMD_NODENAME}"
echo "GPU:          $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:         $(date --iso-8601=seconds)"
echo "Git commit:   $(git rev-parse HEAD)"
echo "Output:       $OUTPUT_DIR"
echo "Warm start:   $WARM_START"
echo "Warm SHA-256: $(sha256sum "$WARM_START" | cut -d' ' -f1)"
echo "Real images:  disabled (generated blank visual input only)"
echo "Extra args:   $*"

"$PYTHON_BIN" code/script/train_stage4b_v5.py \
  --data_dir                    data/prepared \
  --output_dir                  "$OUTPUT_DIR" \
  --batch_size                  1 \
  --gradient_accumulation_steps 8 \
  --learning_rate               2e-5 \
  --num_epochs                  30 \
  --early_stopping_patience     10 \
  --linker_loss_weight          5.0 \
  --seed                        42 \
  --val_fraction                0.1 \
  --warm_start                  "$WARM_START" \
  --device                      cuda \
  "$@"

echo "=== V5 job finished ==="
