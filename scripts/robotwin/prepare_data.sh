#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/robotwin/prepare_data.sh
#   bash scripts/robotwin/prepare_data.sh /path/to/storage
#   DEST=/path/to/storage bash scripts/robotwin/prepare_data.sh
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/robotwin/prepare_data.sh
#
# Default behavior:
#   - downloads the LeRobot/parquet clean split from Hugging Face
#   - stores it under:   $DEST/robotwin_clean/StarVLA_RoboTwin_Clean
#   - creates symlink:   playground/Datasets/RoboTwin
#
CUR="$(pwd)"
DEST="${DEST:-${1:-$CUR/playground/Datasets}}"
DATASET_REPO="${DATASET_REPO:-StarVLA/RoboTwin-Clean}"
DATASET_ALIAS="${DATASET_ALIAS:-RoboTwin}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$DEST/robotwin_clean}"
LOCAL_DIR="${LOCAL_DIR:-$DOWNLOAD_ROOT/StarVLA_RoboTwin_Clean}"
LINK_PATH="${LINK_PATH:-$CUR/playground/Datasets/$DATASET_ALIAS}"

mkdir -p "$DOWNLOAD_ROOT"
mkdir -p "$CUR/playground/Datasets"

if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi

python -m pip install -U "huggingface-hub[hf_xet]==0.35.3"

hf download "$DATASET_REPO" \
  --repo-type dataset \
  --local-dir "$LOCAL_DIR"

ln -sfn "$LOCAL_DIR" "$LINK_PATH"

echo ""
echo "Done. RoboTwin clean LeRobot dataset is ready."
echo "  repo:       $DATASET_REPO"
echo "  local dir:  $LOCAL_DIR"
echo "  symlink:    $LINK_PATH"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  echo "  hf mirror:  $HF_ENDPOINT"
fi
echo ""
echo "For StarVLA training, use:"
echo "  export ROBOTWIN_DATA_ROOT=$LINK_PATH"
echo "  bash scripts/robotwin/train.sh"
