#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

: "${CSV_PATH:?Set CSV_PATH to a CSV manifest with a 'path' column.}"
: "${PRETRAINED_MODEL_PATH:?Set PRETRAINED_MODEL_PATH to Wan2.1-I2V-14B-720P-Diffusers or 480P-Diffusers.}"

CONFIG_PATH=${CONFIG_PATH:-configs/uni3c_controlnet_config.json}
CONTROLNET_PATH=${CONTROLNET_PATH:-controlnet.pth}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/camera_control}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-deepspeed_configs/accelerate_single_8gpu.yaml}
MASTER_PORT=${MASTER_PORT:-29501}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

EXTRA_ARGS=()
if [[ -n "${REPORT_TO:-}" ]]; then
  EXTRA_ARGS+=(--report_to "$REPORT_TO")
fi
if [[ "${USE_LOG_VALIDATION:-0}" == "1" ]]; then
  : "${VALIDATION_CSV_PATH:?Set VALIDATION_CSV_PATH when USE_LOG_VALIDATION=1.}"
  EXTRA_ARGS+=(--use_log_validation --validation_csv_path "$VALIDATION_CSV_PATH")
fi

accelerate launch --config_file "$ACCELERATE_CONFIG" --main_process_port "$MASTER_PORT" \
  train.py \
  --csv_path "$CSV_PATH" \
  --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
  --config_path "$CONFIG_PATH" \
  --controlnet_path "$CONTROLNET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_frames "${NUM_FRAMES:-81}" \
  --max_area "${MAX_AREA:-399360}" \
  --use_camera_embedding \
  --train_architecture controller_only \
  --train_batch_size "${TRAIN_BATCH_SIZE:-1}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-20}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-20000}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-250}" \
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT:-20}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --gradient_checkpointing \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --lr_scheduler "${LR_SCHEDULER:-constant_with_warmup}" \
  --lr_warmup_steps "${LR_WARMUP_STEPS:-400}" \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --logging_dir logs \
  --allow_tf32 \
  --nccl_timeout "${NCCL_TIMEOUT:-7200}" \
  "${EXTRA_ARGS[@]}"
