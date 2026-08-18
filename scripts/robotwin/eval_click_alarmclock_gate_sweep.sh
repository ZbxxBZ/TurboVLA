#!/usr/bin/env bash
set -euo pipefail

repo_root="/root/TurboVLA-repro"
checkpoint="${repo_root}/results/Checkpoints/turbovla_robotwin_rgbd_arch18_200_depthgate_3ep_20260815/final_model/ema_ema_pytorch_model.pt"
sweep_id="depthgate3ep_click_alarmclock_gate_sweep_seed0_20260815"
sweep_root="${repo_root}/results/Evaluations/${sweep_id}"

if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] checkpoint is missing: ${checkpoint}" >&2
    exit 1
fi
if [[ -e "${sweep_root}" ]]; then
    echo "[ERROR] sweep output already exists: ${sweep_root}" >&2
    exit 1
fi

source /root/robotwin_env.sh
mkdir -p "${sweep_root}"

gates=(0.02 0.08 0.16)
labels=(0p02 0p08 0p16)

for index in "${!gates[@]}"; do
    gate="${gates[$index]}"
    label="${labels[$index]}"
    run_id="${sweep_id}_gate_${label}"
    run_root="${sweep_root}/gate_${label}"
    log_root="${run_root}/logs"
    mkdir -p "${log_root}"

    echo "[INFO] evaluating click_alarmclock with fixed gate=${gate}"
    set +e
    env \
      ROBOTWIN_PATH=/root/RoboTwin \
      STARVLA_PYTHON=/root/miniconda3/envs/myconda/bin/python \
      ROBOTWIN_PYTHON=/root/venvs/robotwin/bin/python \
      ROBOTWIN_TEST_NUM=1 \
      ROBOTWIN_SEED=0 \
      ROBOTWIN_DEPTH_INPUT_MODE=real \
      ROBOTWIN_ACTION_TRACE_PATH="${run_root}/action_trace.jsonl" \
      TURBOVLA_DEPTH_GATE_OVERRIDE="${gate}" \
      ROBOTWIN_JOBS_PER_GPU=1 \
      ROBOTWIN_LOG_ROOT="${log_root}" \
      ROBOTWIN_BASE_PORT="$((7700 + index * 10))" \
      ROBOTWIN_SERVER_TIMEOUT=600 \
      ROBOTWIN_POLICY_NAME=model2robotwin_interface \
      POLICY_NAME="${run_id}" \
      VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json \
      __GLX_VENDOR_LIBRARY_NAME=nvidia \
      LD_LIBRARY_PATH="/root/miniconda3/envs/myconda/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}" \
      bash "${repo_root}/scripts/robotwin/evaluate.sh" "${checkpoint}" click_alarmclock \
      2>&1 | tee "${run_root}/main.log"
    status=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "${status}" > "${run_root}/exit_code"
    if (( status != 0 )); then
        exit "${status}"
    fi
done

/root/miniconda3/envs/myconda/bin/python \
  "${repo_root}/scripts/robotwin/analyze_click_alarmclock_trace.py" \
  --trace "0.02=${sweep_root}/gate_0p02/action_trace.jsonl" \
  --trace "0.08=${sweep_root}/gate_0p08/action_trace.jsonl" \
  --trace "0.16=${sweep_root}/gate_0p16/action_trace.jsonl" \
  --output "${sweep_root}/action_metrics.json" \
  2>&1 | tee "${sweep_root}/analysis.log"

printf '0\n' > "${sweep_root}/exit_code"
