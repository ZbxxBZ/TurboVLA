#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

run_id="${RUN_ID:-turbovla_rgbd10_official_depth_action_ft_10ep_mb8ga2_final_20260814}"
session_name="${PIPELINE_SESSION:-rgbd10_train10_mb8ga2_eval5_backup}"
pipeline_root="${REPO_ROOT}/results/pipelines/${run_id}"
pipeline_log="${pipeline_root}/pipeline.log"
output_dir="${REPO_ROOT}/results/Checkpoints/${run_id}"
eval_dir="${REPO_ROOT}/results/robotwin_eval/${run_id}_eval5_seed0"
backup_dir="/mnt/turbovla_backups/${run_id}"

if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "[ERROR] tmux session already exists: ${session_name}" >&2
    exit 1
fi
for path in "${output_dir}" "${eval_dir}" "${backup_dir}" "${backup_dir}.partial"; do
    if [[ -e "${path}" ]]; then
        echo "[ERROR] Refusing to overwrite existing path: ${path}" >&2
        exit 1
    fi
done
if [[ -e "${pipeline_root}" ]]; then
    echo "[ERROR] Pipeline directory already exists: ${pipeline_root}" >&2
    exit 1
fi

mkdir -p "${pipeline_root}"
printf 'queued\n' > "${pipeline_root}/state"
printf '%s\tqueued\n' "$(date --iso-8601=seconds)" > "${pipeline_root}/state_history.log"

tmux new-session -d -s "${session_name}" \
    "cd '${REPO_ROOT}' && exec env RUN_ID='${run_id}' PIPELINE_ROOT='${pipeline_root}' bash '${SCRIPT_DIR}/train_then_eval_backup.sh' >> '${pipeline_log}' 2>&1"

echo "[INFO] Training/evaluation pipeline started."
echo "[INFO] tmux session: ${session_name}"
echo "[INFO] state: ${pipeline_root}/state"
echo "[INFO] log: ${pipeline_log}"
echo "[INFO] follow: tail -f '${pipeline_log}'"
