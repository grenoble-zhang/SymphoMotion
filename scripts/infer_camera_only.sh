#!/usr/bin/env bash
# Camera-only inference (no object trajectories)

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Default configuration
export VALIDATION_CSV_PATH=${VALIDATION_CSV_PATH:-assets/demo.csv}
export PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-pretrained_models/Wan2.1-I2V-14B-720P-Diffusers}
export CONFIG_PATH=${CONFIG_PATH:-configs/uni3c_controlnet_config.json}
export CONTROLNET_PATH=${CONTROLNET_PATH:-pretrained_checkpoints/camera_control/controlnet.pth}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/inference_camera_only}
export NUM_GPUS=${NUM_GPUS:-8}
export USE_OBJECT_PROMPT=0

echo "Running camera-only inference..."
echo "  VALIDATION_CSV_PATH: $VALIDATION_CSV_PATH"
echo "  PRETRAINED_MODEL_PATH: $PRETRAINED_MODEL_PATH"
echo "  CONTROLNET_PATH: $CONTROLNET_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  NUM_GPUS: $NUM_GPUS"
echo ""

bash scripts/infer.sh
