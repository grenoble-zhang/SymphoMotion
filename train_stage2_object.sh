#!/usr/bin/env bash
# Stage 2: Object Control Training
# This script sets up environment variables and runs object control training

set -euo pipefail

# Navigate to project root
cd "$(dirname "${BASH_SOURCE[0]}")"

# Stage 2 Configuration
export CSV_PATH=data/train.csv
export PRETRAINED_MODEL_PATH=pretrained_models/Wan2.1-I2V-14B-720P-Diffusers
export CONTROLNET_PATH=pretrained_checkpoints/camera_control/controlnet.pth
export OUTPUT_DIR=outputs/object_control

echo "=========================================="
echo "Stage 2: Object Control Training"
echo "=========================================="
echo "CSV_PATH: $CSV_PATH"
echo "PRETRAINED_MODEL_PATH: $PRETRAINED_MODEL_PATH"
echo "CONTROLNET_PATH: $CONTROLNET_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run training
bash scripts/train_object_control.sh
