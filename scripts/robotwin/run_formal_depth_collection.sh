#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ROBOTWIN_PATH="${ROBOTWIN_PATH:-/root/robotwin_src/RoboTwin-main}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/root/miniconda3/envs/robotwin/bin/python}"
ROBOTWIN_ENV_SCRIPT="${ROBOTWIN_ENV_SCRIPT:-/root/robotwin_env.sh}"
OUTPUT_ROOT="${ROBOTWIN_RGBD_ROOT:-/root/dataset/dinov3_depth_formal_head}"
PHYSICAL_GPUS="${ROBOTWIN_COLLECTION_GPUS:-0}"
WORKERS_PER_GPU="${ROBOTWIN_WORKERS_PER_GPU:-4}"
CONFIG_TEMPLATE="${SCRIPT_DIR}/configs/demo_clean_depth.yml"
COLLECTOR="${SCRIPT_DIR}/collect_robotwin_rgbd.py"
FORMAL_LOG="${OUTPUT_ROOT}/_formal/master.log"

if [[ -f "${ROBOTWIN_ENV_SCRIPT}" ]]; then
    # shellcheck source=/dev/null
    source "${ROBOTWIN_ENV_SCRIPT}"
fi

mkdir -p "$(dirname "${FORMAL_LOG}")"

run_stage() {
    local stage_name="$1"
    local episodes_per_task="$2"
    local task_file="$3"

    echo "[$(date --iso-8601=seconds)] starting ${stage_name}" | tee -a "${FORMAL_LOG}"
    "${ROBOTWIN_PYTHON}" "${COLLECTOR}" \
        --robotwin-path "${ROBOTWIN_PATH}" \
        --python "${ROBOTWIN_PYTHON}" \
        --output-root "${OUTPUT_ROOT}" \
        --task-file "${task_file}" \
        --config-template "${CONFIG_TEMPLATE}" \
        --config-name demo_clean_depth_turbovla \
        --episodes-per-task "${episodes_per_task}" \
        --gpus "${PHYSICAL_GPUS}" \
        --workers-per-gpu "${WORKERS_PER_GPU}" \
        --skip-render-smoke 2>&1 | tee -a "${FORMAL_LOG}"
    echo "[$(date --iso-8601=seconds)] completed ${stage_name}" | tee -a "${FORMAL_LOG}"
}

cd "${REPO_ROOT}"
run_stage "click tasks: 60 episodes each" 60 "${SCRIPT_DIR}/rgbd_formal_click_tasks.txt"
run_stage "other Clean tasks: 5 episodes each" 5 "${SCRIPT_DIR}/rgbd_formal_other_clean_tasks.txt"

echo "[$(date --iso-8601=seconds)] formal collection complete" | tee -a "${FORMAL_LOG}"
