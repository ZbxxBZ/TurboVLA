#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ckpt_path="${1:-${CKPT_PATH:-}}"
if [[ -z "${ckpt_path}" ]]; then
    echo "Usage: bash scripts/robotwin/evaluate.sh <checkpoint.{pt,safetensors}> [task ...]" >&2
    echo "       If no task is provided, all 50 clean tasks are evaluated." >&2
    exit 1
fi
shift || true

if [[ ! -f "${ckpt_path}" ]]; then
    echo "[ERROR] Checkpoint not found: ${ckpt_path}" >&2
    exit 1
fi

: "${ROBOTWIN_PATH:?Set ROBOTWIN_PATH to the RoboTwin repository root.}"

export STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python}"
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-20}"
export ROBOTWIN_POLICY_NAME="${ROBOTWIN_POLICY_NAME:-model2robotwin_interface}"
export DEPLOY_POLICY_TEMPLATE_PATH="${DEPLOY_POLICY_TEMPLATE_PATH:-${REPO_ROOT}/experiments/robotwin/evaluation/deploy_policy.yml}"

tasks=("$@")
if (( ${#tasks[@]} == 0 )); then
    tasks=(all)
fi

bash "${SCRIPT_DIR}/start_eval.sh" \
  --mode demo_clean \
  --name "${POLICY_NAME:-turbovla_clean50}" \
  --ckpt "${ckpt_path}" \
  --jobs-per-gpu "${ROBOTWIN_JOBS_PER_GPU:-1}" \
  --base-port "${ROBOTWIN_BASE_PORT:-7100}" \
  "${tasks[@]}"
