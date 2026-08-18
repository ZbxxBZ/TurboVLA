#!/usr/bin/env bash
set -euo pipefail

cd /root/TurboVLA-repro

export LD_LIBRARY_PATH="/root/nvidia-graphics-580.105.08/rootfs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export CUDA_VISIBLE_DEVICES=0
export ROBOTWIN_PATH=/root/RoboTwin
export STARVLA_PYTHON=/root/venvs/robotwin/bin/python
export ROBOTWIN_PYTHON=/root/venvs/robotwin/bin/python
export ROBOTWIN_TEST_NUM=2
export ROBOTWIN_SEED=1
export ROBOTWIN_JOBS_PER_GPU=1
export ROBOTWIN_BASE_PORT=7200
export ROBOTWIN_SERVER_TIMEOUT=600
export ROBOTWIN_LOG_ROOT=/root/TurboVLA-repro/logs/robotwin_eval_smoke_2ep_seed1_detail
export POLICY_NAME=rgbd_depth8_smoke_2ep_seed1

checkpoint=/root/TurboVLA-repro/results/Checkpoints/turbovla_robotwin_rgbd10_depth_only_8ep_fixed_20260813/final_model/ema_ema_pytorch_model.pt

bash scripts/robotwin/evaluate.sh "${checkpoint}" click_bell \
    > /root/TurboVLA-repro/logs/robotwin_eval_smoke_rgbd_2ep_seed1.log 2>&1
