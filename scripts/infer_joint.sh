#!/usr/bin/env bash
# Joint camera and object control inference

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Default configuration
export VALIDATION_CSV_PATH=${VALIDATION_CSV_PATH:-assets/demo.csv}
export PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-pretrained_models/Wan2.1-I2V-14B-720P-Diffusers}
export CONFIG_PATH=${CONFIG_PATH:-configs/uni3c_controlnet_config.json}
export CONTROLNET_PATH=${CONTROLNET_PATH:-pretrained_checkpoints/camera_control/controlnet.pth}
export OBJ_INJECTOR_PATH=${OBJ_INJECTOR_PATH:-pretrained_checkpoints/object_control/object_injector.pth}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/inference}
export NUM_GPUS=${NUM_GPUS:-8}

echo "Running joint camera and object control inference..."
echo "  VALIDATION_CSV_PATH: $VALIDATION_CSV_PATH"
echo "  PRETRAINED_MODEL_PATH: $PRETRAINED_MODEL_PATH"
echo "  CONTROLNET_PATH: $CONTROLNET_PATH"
echo "  OBJ_INJECTOR_PATH: $OBJ_INJECTOR_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  NUM_GPUS: $NUM_GPUS"
echo ""

bash scripts/infer.sh
