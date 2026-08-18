#!/usr/bin/env bash
set -euo pipefail

cd /root/TurboVLA-repro

session_name="${CHAIN_SESSION:-official_eval_then_depth_train}"
chain_log="${CHAIN_LOG:-/root/TurboVLA-repro/logs/official_eval_then_depth_train.log}"

if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "[ERROR] tmux session already exists: ${session_name}" >&2
    exit 1
fi

mkdir -p "$(dirname "${chain_log}")"
# Prevent a stale tmux/server environment from selecting an earlier 8-epoch
# depth checkpoint. The chain script also pins and records the official path.
unset TURBOVLA_INIT_CKPT
tmux new-session -d -s "${session_name}" \
    "cd /root/TurboVLA-repro && \
     export EVAL_SESSION=official_rgb55k_eval5 && \
     export EVAL_PROCESS_PATTERN=official_rgb55k_eval5_seed0 && \
     export EVAL_ROOT=/root/TurboVLA-repro/results/robotwin_eval/official_rgb55k_10tasks_5eps_seed0 && \
     export TASK_FILE=/root/TurboVLA-repro/scripts/robotwin/rgbd_tasks_10.txt && \
     export EXPECTED_TASKS=10 && \
     export EPISODES_PER_TASK=5 && \
     export EXPECTED_SEED_START=100000 && \
     export POLL_SECONDS=30 && \
     export ROBOTWIN_DATA_ROOT=/root/robotwin_rgbd_lerobot && \
     export STARVLA_PYTHON=/root/miniconda3/envs/myconda/bin/python && \
     export BERT_MODEL_PATH=/root/TurboVLA-repro/pretrained/bert-base-uncased && \
     export DINOV3_MODEL_PATH=/root/TurboVLA-repro/pretrained/dinov3-vitl16-robotwin-checkpoint && \
     export CONFIG_YAML=experiments/robotwin/configs/clean50_depth.yaml && \
     export RUN_ROOT_DIR=results/Checkpoints && \
     export RUN_ID=turbovla_rgbd10_official_depth_action_ft_5ep_20260814 && \
     export TRAIN_EPOCHS=5 && \
     export CUDA_VISIBLE_DEVICES=0 && \
     export NUM_PROCESSES=1 && \
     export MAIN_PROCESS_PORT=29630 && \
     export PER_DEVICE_BATCH_SIZE=1 && \
     export GRADIENT_ACCUMULATION_STEPS=16 && \
     export LEARNING_RATE=5.0e-05 && \
     export DEPTH_LEARNING_RATE=1.0e-04 && \
     export INTERACTION_LEARNING_RATE=5.0e-06 && \
     export ACTION_DECODER_LEARNING_RATE=1.0e-05 && \
     export EMA_DECAY=0.999 && \
     export EMA_DEVICE=cuda && \
     export LOGGING_FREQUENCY=25 && \
     export WANDB_MODE=disabled && \
     exec bash scripts/robotwin/chain_eval_to_depth_training.sh >> '${chain_log}' 2>&1"

echo "[INFO] Evaluation-to-training chain is armed."
echo "[INFO] tmux session: ${session_name}"
echo "[INFO] log: ${chain_log}"
