#!/usr/bin/env bash
#SBATCH --job-name=stage3_final_smoke
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/woody/iwi5/iwi5413h/kvp10k_thesis/logs/%x-%j.out
#SBATCH --error=/home/woody/iwi5/iwi5413h/kvp10k_thesis/logs/%x-%j.err

set -euo pipefail

repository_dir="/home/woody/iwi5/iwi5413h/kvp10k_thesis"
python_bin="$repository_dir/env/kvp10k_env/bin/python"
checkpoint_dir="$repository_dir/data/outputs/stage3_mistral/checkpoint"
prepared_dir="$repository_dir/data/prepared"
selection_file="$repository_dir/data/outputs/stage3_mistral_final_longest_prompt.json"
smoke_output="$repository_dir/data/outputs/stage3_mistral_final_inference_smoke"

export HF_HOME="$repository_dir/hf_cache"
export HF_DATASETS_CACHE="$repository_dir/hf_cache"
export TRANSFORMERS_CACHE="$repository_dir/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DEACTIVATE_ASYNC_LOAD=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$repository_dir"

if [[ ! -d "$checkpoint_dir" ]]; then
    echo "ERROR: Final Stage 3 checkpoint is missing: $checkpoint_dir" >&2
    exit 1
fi
if [[ ! -f "$selection_file" ]]; then
    echo "ERROR: Longest-prompt selection is missing: $selection_file" >&2
    exit 1
fi

echo "Host: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

"$python_bin" code/script/stage3_mistral_final_inference.py smoke \
    --checkpoint "$checkpoint_dir" \
    --data-dir "$prepared_dir" \
    --output-root "$smoke_output" \
    --selection-file "$selection_file"

nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
echo "End: $(date --iso-8601=seconds)"
