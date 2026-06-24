#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

: "${VALIDATION_CSV_PATH:?Set VALIDATION_CSV_PATH to a CSV manifest with a 'path' column.}"
: "${PRETRAINED_MODEL_PATH:?Set PRETRAINED_MODEL_PATH to Wan2.1-I2V-14B-720P-Diffusers or 480P-Diffusers.}"
: "${CONTROLNET_PATH:?Set CONTROLNET_PATH to a trained camera ControlNet .pth file.}"

CONFIG_PATH=${CONFIG_PATH:-configs/uni3c_controlnet_config.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/inference}
NUM_GPUS=${NUM_GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29501}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

EXTRA_ARGS=()
if [[ "${USE_OBJECT_PROMPT:-1}" == "1" ]]; then
  : "${OBJ_INJECTOR_PATH:?Set OBJ_INJECTOR_PATH when USE_OBJECT_PROMPT=1.}"
  EXTRA_ARGS+=(--use_object_prompt --obj_injector_path "$OBJ_INJECTOR_PATH")
fi
if [[ "${SAVE_CONCAT_VIDEO:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_concat_video)
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python3 -m torch.distributed.run --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  infer.py \
  --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
  --config_path "$CONFIG_PATH" \
  --validation_csv_path "$VALIDATION_CSV_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --controlnet_path "$CONTROLNET_PATH" \
  --num_frames "${NUM_FRAMES:-81}" \
  --max_area "${MAX_AREA:-399360}" \
  --guidance_scale "${GUIDANCE_SCALE:-5.0}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-40}" \
  --fps "${FPS:-16}" \
  --seed "${SEED:-42}" \
  --use_camera_embedding \
  "${EXTRA_ARGS[@]}"
