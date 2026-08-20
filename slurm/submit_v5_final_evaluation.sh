#!/bin/bash
#SBATCH --job-name=v5_final_eval
#SBATCH --output=logs/v5_final_eval-%j.out
#SBATCH --error=logs/v5_final_eval-%j.err
#SBATCH --time=03:00:00
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
RUN_DIR="data/outputs/stage4b_v5"
OUTPUT_DIR="data/outputs/stage4b_v5_final_evaluation"
SELECTED_SHA="f4b61bf2db8833aa5b23bf49c9778fb154177b18954e6f8563a59cf80145a27c"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: refusing to overwrite $OUTPUT_DIR" >&2
  exit 1
fi
if [[ $(find "$RUN_DIR/final_test_predictions" -maxdepth 1 -name '*.json' | wc -l) -ne 581 ]]; then
  echo "ERROR: expected 581 direct V5 prediction files" >&2
  exit 1
fi

echo "=== V5 final evaluation-only analyses ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date --iso-8601=seconds)"
echo "Selected checkpoint SHA-256: $SELECTED_SHA"
echo "NED < 0.2 | IoU >= 0.3 | link threshold 0.5"

PYTHONPATH=code/script "$PYTHON_BIN" code/script/export_v5_recovery_analysis.py \
  --checkpoint_root "$RUN_DIR" \
  --expected_sha256 "$SELECTED_SHA" \
  --data_dir data/prepared \
  --cluster_map data/outputs/stage2/test_cluster_map.json \
  --prediction_dir "$OUTPUT_DIR/postprocessed_predictions" \
  --entity_output "$OUTPUT_DIR/entity_error_candidates.json" \
  --score_threshold 0.5 \
  --batch_size 1

PYTHONPATH=code/script "$PYTHON_BIN" code/script/evaluate_kvp10k_benchmark.py \
  --prediction_dir "$RUN_DIR/final_test_predictions" \
  --ground_truth_dir data/prepared/test \
  --cluster_map data/outputs/stage2/test_cluster_map.json \
  --ned_threshold 0.2 \
  --iou_threshold 0.3 \
  --output "$OUTPUT_DIR/direct_official.json"

PYTHONPATH=code/script "$PYTHON_BIN" code/script/evaluate_kvp10k_benchmark.py \
  --prediction_dir "$OUTPUT_DIR/postprocessed_predictions" \
  --ground_truth_dir data/prepared/test \
  --cluster_map data/outputs/stage2/test_cluster_map.json \
  --ned_threshold 0.2 \
  --iou_threshold 0.3 \
  --output "$OUTPUT_DIR/postprocessed_official.json"

PYTHONPATH=code/script "$PYTHON_BIN" code/script/analyze_v5_qualitative.py \
  --prediction_dir "$RUN_DIR/final_test_predictions" \
  --ground_truth_dir data/prepared/test \
  --cluster_map data/outputs/stage2/test_cluster_map.json \
  --entity_candidates "$OUTPUT_DIR/entity_error_candidates.json" \
  --output "$OUTPUT_DIR/qualitative_examples.json"

PYTHONPATH=code/script "$PYTHON_BIN" code/script/collect_v5_final_run_info.py \
  --run_dir "$RUN_DIR" \
  --job_ids 1779774 1781811 \
  --output "$OUTPUT_DIR/final_run_info.json"

echo "=== V5 final evaluation complete: $(date --iso-8601=seconds) ==="
