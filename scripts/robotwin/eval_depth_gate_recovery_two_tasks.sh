#!/usr/bin/env bash
set -euo pipefail

repo_root="/root/TurboVLA-repro"
train_run_id="turbovla_robotwin_rgbd_arch18_200_depthgate_3ep_20260815"
eval_run_id="depthgate3ep_click_alarmclock_adjust_bottle_2eps_seed0_20260815"
train_exit="${repo_root}/results/Logs/${train_run_id}.exit_code"
checkpoint="${repo_root}/results/Checkpoints/${train_run_id}/final_model/ema_ema_pytorch_model.pt"
eval_root="${repo_root}/results/Evaluations/${eval_run_id}"
log_root="${eval_root}/logs"
main_log="${eval_root}/main.log"

while [[ ! -f "${train_exit}" ]]; do
    sleep 60
done

if [[ "$(tr -d '[:space:]' < "${train_exit}")" != "0" ]]; then
    echo "[ERROR] Training failed; evaluation will not start." >&2
    exit 1
fi
if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] Final EMA checkpoint is missing: ${checkpoint}" >&2
    exit 1
fi
if [[ -e "${eval_root}" ]]; then
    echo "[ERROR] Evaluation output already exists: ${eval_root}" >&2
    exit 1
fi

source /root/robotwin_env.sh
mkdir -p "${log_root}"

set +e
env \
  ROBOTWIN_PATH=/root/RoboTwin \
  STARVLA_PYTHON=/root/miniconda3/envs/myconda/bin/python \
  ROBOTWIN_PYTHON=/root/venvs/robotwin/bin/python \
  ROBOTWIN_TEST_NUM=2 \
  ROBOTWIN_SEED=0 \
  ROBOTWIN_DEPTH_INPUT_MODE=real \
  ROBOTWIN_JOBS_PER_GPU=1 \
  ROBOTWIN_LOG_ROOT="${log_root}" \
  ROBOTWIN_BASE_PORT=7620 \
  ROBOTWIN_SERVER_TIMEOUT=600 \
  ROBOTWIN_POLICY_NAME=model2robotwin_interface \
  POLICY_NAME="${eval_run_id}" \
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  LD_LIBRARY_PATH="/root/miniconda3/envs/myconda/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}" \
  bash "${repo_root}/scripts/robotwin/evaluate.sh" "${checkpoint}" \
    click_alarmclock adjust_bottle \
  2>&1 | tee "${main_log}"
status=${PIPESTATUS[0]}

if (( status == 0 )); then
  /root/miniconda3/envs/myconda/bin/python "${repo_root}/scripts/robotwin/summarize_eval_logs.py" \
    "${log_root}" \
    --episodes-csv "${eval_root}/episodes.csv" \
    --summary-csv "${eval_root}/summary.csv" \
    2>&1 | tee -a "${main_log}"
  status=${PIPESTATUS[0]}
fi

printf '%s\n' "${status}" > "${eval_root}/exit_code"
exit "${status}"
