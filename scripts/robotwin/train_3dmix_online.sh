#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-turbovla_robotwin_online_3dmix}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
export WARMUP_STEPS="${WARMUP_STEPS:-1000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"

: "${VGGT_MODEL_PATH:?Set VGGT_MODEL_PATH to the local VGGT checkpoint.}"
: "${VGGT_CODE_PATH:?Set VGGT_CODE_PATH to the local VGGT source directory.}"

bash "${SCRIPT_DIR}/train.sh" \
  --framework.three_dmix.enabled true \
  --framework.three_dmix.online true \
  --framework.three_dmix.vggt_dim "${VGGT_DIM:-2048}" \
  --framework.three_dmix.semantic_pool "${SEMANTIC_POOL:-vl}" \
  --framework.three_dmix.output_scale_init "${THREEDMIX_OUTPUT_SCALE_INIT:-0.0}" \
  --framework.three_dmix.vggt_model_path "${VGGT_MODEL_PATH}" \
  --framework.three_dmix.vggt_code_path "${VGGT_CODE_PATH}" \
  --framework.three_dmix.vggt_input_size "${VGGT_INPUT_SIZE:-518}" \
  --trainer.learning_rate.three_dmix "${THREEDMIX_LR:-1.0e-4}" \
  --trainer.learning_rate.action_head "${ACTION_HEAD_LR:-1.0e-5}" \
  --trainer.freeze_modules "text_encoder,vision_encoder,vision_projection,vision_language_interaction" \
  "$@"
