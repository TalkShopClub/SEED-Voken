#!/bin/bash
# Train Residual IBQ: frozen L1 (IBQ-1024) + trainable L2 (1024) residual codebook.
# Data: ubuntu_images_raw + win_mac_images_raw + imagenet-1k, random crop 384x384.
# Logs GT vs reconstruction to WandB every 5000 steps. Checkpoints best 2.

set -e

export WANDB_API_KEY="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"
export OMP_NUM_THREADS=24
export MASTER_ADDR=${1:-localhost}
export MASTER_PORT=${2:-10056}
export NODE_RANK=${3:-0}

cd "$(dirname "$0")/../../.."

echo "Starting Residual IBQ training..."
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

NODE_RANK=$NODE_RANK python main.py fit \
    --config configs/IBQ/gpu/residual_ibq_1024_384.yaml
