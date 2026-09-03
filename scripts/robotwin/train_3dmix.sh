#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${VGGT_FEATURE_ROOT:?Set VGGT_FEATURE_ROOT to the offline native-resolution VGGT feature cache.}"

export RUN_ID="${RUN_ID:-turbovla_robotwin_click_3dmix}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
export WARMUP_STEPS="${WARMUP_STEPS:-1000}"
vggt_feature_pattern="${VGGT_FEATURE_PATTERN:-}"
if [[ -z "${vggt_feature_pattern}" ]]; then
  vggt_feature_pattern='{dataset}/{trajectory_id}/{base_index}.pt'
fi

bash "${SCRIPT_DIR}/train.sh" \
  --framework.three_dmix.enabled true \
  --framework.three_dmix.vggt_dim "${VGGT_DIM:-2048}" \
  --framework.three_dmix.semantic_pool "${SEMANTIC_POOL:-vl}" \
  --framework.three_dmix.output_scale_init "${THREEDMIX_OUTPUT_SCALE_INIT:-0.0}" \
  --datasets.vla_data.vggt_feature_root "${VGGT_FEATURE_ROOT}" \
  --datasets.vla_data.vggt_feature_pattern "${vggt_feature_pattern}" \
  --trainer.learning_rate.three_dmix "${THREEDMIX_LR:-1.0e-4}" \
  --trainer.learning_rate.action_head "${ACTION_HEAD_LR:-1.0e-5}" \
  --trainer.freeze_modules "text_encoder,vision_encoder,vision_projection,vision_language_interaction" \
  "$@"
