#!/usr/bin/env bash
#SBATCH --job-name=kvp_stage3_resume
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

repository_dir="/home/woody/iwi5/iwi5413h/kvp10k_thesis"
python_bin="$repository_dir/env/kvp10k_env/bin/python"
prepared_dir="$repository_dir/data/prepared"
output_dir="$repository_dir/data/outputs/stage3_mistral_clean"
final_checkpoint="$output_dir/checkpoint"

export HF_HOME="$repository_dir/hf_cache"
export HF_DATASETS_CACHE="$repository_dir/hf_cache"
export TRANSFORMERS_CACHE="$repository_dir/hf_cache"
export KVP10K_HF_CACHE="$repository_dir/hf_cache"
export KVP10K_OUTPUT_DIR="$repository_dir/data/outputs"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$repository_dir"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "SLURM job: $SLURM_JOB_ID"

if [[ -d "$final_checkpoint" ]]; then
    echo "Final Stage 3 checkpoint already exists: $final_checkpoint"
    echo "No resume is necessary."
    exit 0
fi

latest_checkpoint=""
mapfile -t checkpoint_dirs < <(find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V)
for ((index=${#checkpoint_dirs[@]} - 1; index >= 0; index--)); do
    candidate="${checkpoint_dirs[$index]}"
    if [[ -f "$candidate/trainer_state.json" \
          && -f "$candidate/optimizer.pt" \
          && -f "$candidate/scheduler.pt" \
          && -f "$candidate/adapter_config.json" \
          && ( -f "$candidate/adapter_model.safetensors" || -f "$candidate/adapter_model.bin" ) ]]; then
        latest_checkpoint="$candidate"
        break
    fi
done

if [[ -z "$latest_checkpoint" ]]; then
    echo "ERROR: No complete Stage 3 training checkpoint exists in $output_dir" >&2
    exit 1
fi

echo "Resume checkpoint: $latest_checkpoint"
nvidia-smi -L

"$python_bin" code/script/mistral_baseline.py train \
    --data_dir "$prepared_dir/train" \
    --output_dir "$output_dir" \
    --resume_from_checkpoint "$latest_checkpoint"

if [[ ! -d "$final_checkpoint" ]]; then
    echo "ERROR: Training ended without the final checkpoint: $final_checkpoint" >&2
    exit 1
fi

echo "Final Stage 3 checkpoint: $final_checkpoint"
echo "DATE: $(date)"
echo "DONE"
