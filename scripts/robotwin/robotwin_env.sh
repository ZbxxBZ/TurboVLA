#!/usr/bin/env bash

export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/my_nvidia_icd.json}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LD_LIBRARY_PATH="/root/nvidia-driver-580.76.05${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/root/robotwin_src/RoboTwin-main}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/root/miniconda3/envs/robotwin/bin/python}"
export ROBOTWIN_RGBD_ROOT="${ROBOTWIN_RGBD_ROOT:-/root/dataset}"
