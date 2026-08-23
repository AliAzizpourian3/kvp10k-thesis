#!/usr/bin/env bash
#SBATCH --job-name=stage3_final_full
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/woody/iwi5/iwi5413h/kvp10k_thesis/logs/%x-%j.out
#SBATCH --error=/home/woody/iwi5/iwi5413h/kvp10k_thesis/logs/%x-%j.err

set -euo pipefail

repository_dir="/home/woody/iwi5/iwi5413h/kvp10k_thesis"
python_bin="$repository_dir/env/kvp10k_env/bin/python"
checkpoint_dir="$repository_dir/data/outputs/stage3_mistral/checkpoint"
prepared_dir="$repository_dir/data/prepared"
output_dir="$repository_dir/data/outputs/stage3_mistral_final_inference"

export HF_HOME="$repository_dir/hf_cache"
export HF_DATASETS_CACHE="$repository_dir/hf_cache"
export TRANSFORMERS_CACHE="$repository_dir/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DEACTIVATE_ASYNC_LOAD=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$repository_dir"

if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: Python interpreter is missing: $python_bin" >&2
    exit 1
fi
if [[ ! -f "$checkpoint_dir/adapter_config.json" ]]; then
    echo "ERROR: Stage 3 adapter configuration is missing: $checkpoint_dir" >&2
    exit 1
fi
if [[ ! -f "$checkpoint_dir/adapter_model.safetensors" \
      && ! -f "$checkpoint_dir/adapter_model.bin" ]]; then
    echo "ERROR: Stage 3 adapter weights are missing: $checkpoint_dir" >&2
    exit 1
fi
if [[ ! -d "$prepared_dir/test" ]]; then
    echo "ERROR: Prepared test data is missing: $prepared_dir/test" >&2
    exit 1
fi

echo "Host: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Test documents: $(find "$prepared_dir/test" -maxdepth 1 -type f -name '*.json' | wc -l)"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

"$python_bin" code/script/stage3_mistral_final_inference.py predict \
    --checkpoint "$checkpoint_dir" \
    --data-dir "$prepared_dir" \
    --output-root "$output_dir"

nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
echo "Summary: $output_dir/inference_summary.json"
echo "End: $(date --iso-8601=seconds)"
