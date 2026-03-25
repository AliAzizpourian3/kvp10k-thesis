#!/usr/bin/env bash
#SBATCH --job-name=kvp_gpu_check
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "$WORK/kvp10k_thesis/code/script"

echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "WORK: $WORK"

echo "=== nvidia-smi ==="
command -v nvidia-smi
nvidia-smi -L
nvidia-smi

echo "=== torch cuda check ==="
"$WORK/kvp10k_thesis/env/kvp10k_env/bin/python" - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device count:', torch.cuda.device_count())
    print('device 0:', torch.cuda.get_device_name(0))
PY
