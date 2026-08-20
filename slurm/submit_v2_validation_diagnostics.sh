#!/bin/bash
#SBATCH --job-name=v2_val_diag
#SBATCH --output=logs/v2_validation_diagnostics-%j.out
#SBATCH --error=logs/v2_validation_diagnostics-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$HOME/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/woody/iwi5/iwi5413h/kvp10k_thesis

PYTHON_BIN="env/kvp10k_env/bin/python"
CHECKPOINT_DIR="data/outputs/stage4b_v2_tf_p7"
OUTPUT_DIR="data/outputs/stage4b_v2_validation_diagnostics"
OUTPUT_JSON="$OUTPUT_DIR/diagnostics.json"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d data/prepared/train ]]; then
  echo "ERROR: Prepared training data not found" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_DIR/best_model/pytorch_model.bin" ]]; then
  echo "ERROR: V2 checkpoint not found" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: Refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 1
fi

echo "=== V2 validation diagnostics ==="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date:         $(date --iso-8601=seconds)"
echo "Git commit:   $(git rev-parse HEAD)"
echo "Split:        validation, seed 42, fraction 0.1"
echo "Checkpoint:   $CHECKPOINT_DIR"
echo "Output JSON:  $OUTPUT_JSON"
echo "Real images:  disabled"

"$PYTHON_BIN" code/script/diagnose_v2_validation.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --data_dir data/prepared \
  --output_json "$OUTPUT_JSON" \
  --max_model_samples 200 \
  --audit_samples 20 \
  --seed 42 \
  --val_fraction 0.1

echo "=== V2 validation diagnostics complete ==="
