#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${ROBOTWIN_DATA_ROOT:?Set ROBOTWIN_DATA_ROOT to the converted RoboTwin dataset root.}"
: "${BERT_MODEL_PATH:?Set BERT_MODEL_PATH to a local bert-base-uncased directory.}"
: "${DINOV3_MODEL_PATH:?Set DINOV3_MODEL_PATH to a local DINOv3 model directory.}"
: "${TURBOVLA_INIT_CKPT:?Set TURBOVLA_INIT_CKPT to the official RoboTwin RGB checkpoint.}"

stage1_run_id="${STAGE1_RUN_ID:-turbovla_robotwin_rgbd_arch18_200_depthgate_stage1_20260815}"
final_run_id="${RUN_ID:-turbovla_robotwin_rgbd_arch18_200_depthgate_3ep_20260815}"
run_root_dir="${RUN_ROOT_DIR:-results/Checkpoints}"
stage1_steps="${STAGE1_STEPS:-1275}"
stage2_steps="${STAGE2_STEPS:-6372}"
stage1_output="${run_root_dir}/${stage1_run_id}"
final_output="${run_root_dir}/${final_run_id}"
stage1_checkpoint="${stage1_output}/final_model/pytorch_model.pt"
log_root="${REPO_ROOT}/results/Logs"
log_file="${log_root}/${final_run_id}.log"
exit_file="${log_root}/${final_run_id}.exit_code"
official_checkpoint="${TURBOVLA_INIT_CKPT}"

mkdir -p "${log_root}"
exec > >(tee -a "${log_file}") 2>&1

record_exit() {
    printf '%s\n' "$?" > "${exit_file}"
}
trap record_exit EXIT

export WANDB_MODE="${WANDB_MODE:-disabled}"

if [[ ! -f "${stage1_checkpoint}" ]]; then
    if [[ -e "${stage1_output}" ]]; then
        echo "[ERROR] Incomplete stage-1 output already exists: ${stage1_output}" >&2
        exit 1
    fi
    echo "[INFO] Stage 1/2: depth-only gate warmup (${stage1_steps} steps)."
    env \
      RUN_ROOT_DIR="${run_root_dir}" \
      RUN_ID="${stage1_run_id}" \
      TURBOVLA_INIT_CKPT="${official_checkpoint}" \
      MAX_TRAIN_STEPS="${stage1_steps}" \
      WARMUP_STEPS=100 \
      SAVE_INTERVAL=999999 \
      EMA_DECAY=0 \
      MAIN_PROCESS_PORT=29640 \
      bash "${SCRIPT_DIR}/train.sh" \
        --trainer.train_modules "depth_encoder,depth_fusion" \
        --trainer.eval_interval 999999 \
        --trainer.depth_gate_warmup_steps "${stage1_steps}" \
        --trainer.depth_gate_warmup_start 0.02 \
        --trainer.depth_gate_warmup_end 0.08
else
    echo "[INFO] Reusing completed stage-1 checkpoint: ${stage1_checkpoint}"
fi

if [[ -e "${final_output}" ]]; then
    echo "[ERROR] Final output already exists: ${final_output}" >&2
    exit 1
fi

echo "[INFO] Stage 2/2: full fine-tuning except DINO/BERT (${stage2_steps} steps)."
env \
  RUN_ROOT_DIR="${run_root_dir}" \
  RUN_ID="${final_run_id}" \
  TURBOVLA_INIT_CKPT="${stage1_checkpoint}" \
  MAX_TRAIN_STEPS="${stage2_steps}" \
  WARMUP_STEPS=300 \
  SAVE_INTERVAL=999999 \
  EMA_DECAY=0.999 \
  MAIN_PROCESS_PORT=29640 \
  bash "${SCRIPT_DIR}/train.sh" \
    --trainer.eval_interval 999999 \
    --trainer.depth_gate_warmup_steps 0

echo "[INFO] Depth-gate recovery training completed: ${final_output}"
