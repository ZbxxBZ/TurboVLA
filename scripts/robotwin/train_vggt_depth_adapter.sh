#!/usr/bin/env bash
set -euo pipefail

: "${ROBOTWIN_RGBD_ROOT:?set ROBOTWIN_RGBD_ROOT to the HDF5 RGB-D dataset}"
: "${VGGT_REPO_PATH:?set VGGT_REPO_PATH to the VGGT source tree}"
: "${VGGT_WEIGHTS_PATH:?set VGGT_WEIGHTS_PATH to VGGT-1B model.pt}"
: "${VGGT_ADAPTER_OUTPUT_DIR:?set VGGT_ADAPTER_OUTPUT_DIR for Stage 1 checkpoints}"

python scripts/robotwin/train_vggt_depth_adapter.py \
  --dataset-root "${ROBOTWIN_RGBD_ROOT}" \
  --vggt-repo "${VGGT_REPO_PATH}" \
  --vggt-weights "${VGGT_WEIGHTS_PATH}" \
  --output-dir "${VGGT_ADAPTER_OUTPUT_DIR}" \
  --epochs "${VGGT_ADAPTER_EPOCHS:-3}" \
  --batch-size "${VGGT_ADAPTER_BATCH_SIZE:-1}" \
  --workers "${VGGT_ADAPTER_WORKERS:-4}" \
  --frame-stride "${VGGT_ADAPTER_FRAME_STRIDE:-1}" \
  --learning-rate "${VGGT_ADAPTER_LR:-3e-4}" \
  --stage1-mode "${VGGT_ADAPTER_STAGE1_MODE:-dpt_dense}" \
  --dpt-feature-dim "${VGGT_ADAPTER_DPT_FEATURE_DIM:-256}" \
  --log-every "${VGGT_ADAPTER_LOG_EVERY:-20}"
