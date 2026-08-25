#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SESSION_NAME="${ROBOTWIN_RGBD_TMUX_SESSION:-robotwin_rgbd_100}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/root/RoboTwin}"
ROBOTWIN_ENV_SCRIPT="${ROBOTWIN_ENV_SCRIPT:-/root/robotwin_env.sh}"
if [[ -f "${ROBOTWIN_ENV_SCRIPT}" ]]; then
    # shellcheck source=/dev/null
    source "${ROBOTWIN_ENV_SCRIPT}"
fi
if [[ -z "${ROBOTWIN_PYTHON:-}" ]]; then
    for python_candidate in \
        /root/venvs/robotwin/bin/python \
        /root/miniconda3/envs/robotwin/bin/python \
        /root/anaconda3/envs/robotwin/bin/python \
        /opt/conda/envs/robotwin/bin/python; do
        if [[ -x "${python_candidate}" ]]; then
            ROBOTWIN_PYTHON="${python_candidate}"
            break
        fi
    done
fi
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python}"
ROBOTWIN_RGBD_ROOT="${ROBOTWIN_RGBD_ROOT:-/root/dataset}"
ROBOTWIN_COLLECTION_GPUS="${ROBOTWIN_COLLECTION_GPUS:-}"
ROBOTWIN_WORKERS_PER_GPU="${ROBOTWIN_WORKERS_PER_GPU:-1}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
TASK_FILE="${ROBOTWIN_RGBD_TASK_FILE:-${SCRIPT_DIR}/rgbd_tasks_10.txt}"
CONFIG_TEMPLATE="${ROBOTWIN_RGBD_CONFIG_TEMPLATE:-${SCRIPT_DIR}/configs/demo_clean_depth.yml}"
MASTER_LOG="${ROBOTWIN_RGBD_ROOT}/_autocollect/master.log"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/robotwin/start_rgbd_collection.sh start
  bash scripts/robotwin/start_rgbd_collection.sh status
  bash scripts/robotwin/start_rgbd_collection.sh logs
  bash scripts/robotwin/start_rgbd_collection.sh attach
  bash scripts/robotwin/start_rgbd_collection.sh stop

Environment overrides:
  ROBOTWIN_PATH                 RoboTwin repository outside /mnt
  ROBOTWIN_ENV_SCRIPT           Headless Vulkan environment script
  ROBOTWIN_PYTHON               Python from the RoboTwin conda environment
  ROBOTWIN_RGBD_ROOT            Raw output directory outside /mnt
  ROBOTWIN_COLLECTION_GPUS      Comma-separated GPU IDs (default: auto-detect)
  ROBOTWIN_WORKERS_PER_GPU      Concurrent processes per physical GPU (default: 1)
  EPISODES_PER_TASK             Successful episodes per task (default: 10)
  ROBOTWIN_RGBD_TMUX_SESSION    tmux session name
  ROBOTWIN_RGBD_EXTRA_ARGS      Extra collect_robotwin_rgbd.py arguments
EOF
}

require_tmux() {
    if ! command -v tmux >/dev/null 2>&1; then
        echo "[ERROR] tmux is required" >&2
        exit 1
    fi
}

session_exists() {
    tmux has-session -t "${SESSION_NAME}" 2>/dev/null
}

show_status() {
    status_command=(
        "${ROBOTWIN_PYTHON}"
        "${SCRIPT_DIR}/collect_robotwin_rgbd.py"
        --robotwin-path "${ROBOTWIN_PATH}"
        --output-root "${ROBOTWIN_RGBD_ROOT}"
        --task-file "${TASK_FILE}"
        --episodes-per-task "${EPISODES_PER_TASK}"
        --workers-per-gpu "${ROBOTWIN_WORKERS_PER_GPU}"
        --status-only
    )
    if [[ -n "${ROBOTWIN_COLLECTION_GPUS}" ]]; then
        status_command+=(--gpus "${ROBOTWIN_COLLECTION_GPUS}")
    fi
    "${status_command[@]}" || true
}

action="${1:-start}"
case "${action}" in
    start)
        require_tmux
        if session_exists; then
            echo "[INFO] collection tmux session already exists: ${SESSION_NAME}"
            show_status
            exit 0
        fi
        if [[ ! -d "${ROBOTWIN_PATH}" ]]; then
            echo "[ERROR] RoboTwin path does not exist: ${ROBOTWIN_PATH}" >&2
            exit 1
        fi
        if [[ ! -x "${ROBOTWIN_PYTHON}" ]]; then
            echo "[ERROR] RoboTwin Python is not executable: ${ROBOTWIN_PYTHON}" >&2
            exit 1
        fi
        case "$(realpath -m "${ROBOTWIN_PATH}")" in
            /mnt|/mnt/*)
                echo "[ERROR] copy RoboTwin source from /mnt to /root first" >&2
                exit 1
                ;;
        esac
        case "$(realpath -m "${ROBOTWIN_RGBD_ROOT}")" in
            /mnt|/mnt/*)
                echo "[ERROR] collection output must be outside /mnt" >&2
                exit 1
                ;;
        esac
        mkdir -p "$(dirname "${MASTER_LOG}")"
        command=(
            "${ROBOTWIN_PYTHON}"
            "${SCRIPT_DIR}/collect_robotwin_rgbd.py"
            --robotwin-path "${ROBOTWIN_PATH}"
            --python "${ROBOTWIN_PYTHON}"
            --output-root "${ROBOTWIN_RGBD_ROOT}"
            --task-file "${TASK_FILE}"
            --config-template "${CONFIG_TEMPLATE}"
            --episodes-per-task "${EPISODES_PER_TASK}"
            --workers-per-gpu "${ROBOTWIN_WORKERS_PER_GPU}"
        )
        if [[ -n "${ROBOTWIN_COLLECTION_GPUS}" ]]; then
            command+=(--gpus "${ROBOTWIN_COLLECTION_GPUS}")
        fi
        if [[ -n "${ROBOTWIN_RGBD_EXTRA_ARGS:-}" ]]; then
            # shellcheck disable=SC2206
            extra_args=(${ROBOTWIN_RGBD_EXTRA_ARGS})
            command+=("${extra_args[@]}")
        fi
        printf -v launch_command '%q ' "${command[@]}"
        printf -v quoted_log '%q' "${MASTER_LOG}"
        printf -v quoted_repo '%q' "${REPO_ROOT}"
        printf -v quoted_env_script '%q' "${ROBOTWIN_ENV_SCRIPT}"
        tmux new-session -d -s "${SESSION_NAME}" \
            "cd ${quoted_repo} && source ${quoted_env_script} && set -o pipefail && ${launch_command} 2>&1 | tee -a ${quoted_log}"
        echo "[INFO] started tmux session: ${SESSION_NAME}"
        echo "[INFO] master log: ${MASTER_LOG}"
        echo "[INFO] status: bash ${SCRIPT_DIR}/start_rgbd_collection.sh status"
        ;;
    status)
        show_status
        if command -v tmux >/dev/null 2>&1 && session_exists; then
            echo "[INFO] tmux session is running: ${SESSION_NAME}"
        else
            echo "[INFO] tmux session is not running: ${SESSION_NAME}"
        fi
        ;;
    logs)
        if [[ ! -f "${MASTER_LOG}" ]]; then
            echo "[ERROR] master log does not exist yet: ${MASTER_LOG}" >&2
            exit 1
        fi
        tail -n 100 -f "${MASTER_LOG}"
        ;;
    attach)
        require_tmux
        if ! session_exists; then
            echo "[ERROR] tmux session is not running: ${SESSION_NAME}" >&2
            exit 1
        fi
        exec tmux attach-session -t "${SESSION_NAME}"
        ;;
    stop)
        require_tmux
        if session_exists; then
            tmux send-keys -t "${SESSION_NAME}" C-c
            echo "[INFO] requested a graceful stop for ${SESSION_NAME}"
        else
            echo "[INFO] tmux session is not running: ${SESSION_NAME}"
        fi
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "[ERROR] unknown action: ${action}" >&2
        usage >&2
        exit 2
        ;;
esac
