#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

run_id="${RUN_ID:-turbovla_rgbd10_official_depth_action_ft_10ep_mb8ga2_final_20260814}"
run_root_dir="${RUN_ROOT_DIR:-results/Checkpoints}"
output_dir="${REPO_ROOT}/${run_root_dir}/${run_id}"
pipeline_root="${PIPELINE_ROOT:-${REPO_ROOT}/results/pipelines/${run_id}}"
eval_name="${run_id}_eval5_seed0"
eval_root="${REPO_ROOT}/results/robotwin_eval/${eval_name}"
backup_root="/mnt/turbovla_backups/${run_id}"
backup_staging="${backup_root}.partial"

official_checkpoint="${REPO_ROOT}/ckpt/robotwin/steps_55000_ema_model.safetensors"
official_checkpoint_sha256_expected="d0183df6bafd44507b6c797da5c5ab080ef8446cde4a8127d7280546d9f7c034"
baseline_root="${REPO_ROOT}/results/robotwin_eval/official_rgb55k_10tasks_5eps_seed0"
data_root="/root/robotwin_rgbd_lerobot"
task_file="${SCRIPT_DIR}/rgbd_tasks_10.txt"
config_yaml="experiments/robotwin/configs/clean50_depth.yaml"
bert_path="${REPO_ROOT}/pretrained/bert-base-uncased"
dino_path="${REPO_ROOT}/pretrained/dinov3-vitl16-robotwin-checkpoint"
starvla_python="/root/miniconda3/envs/myconda/bin/python"
robotwin_python="/root/venvs/robotwin/bin/python"
robotwin_path="/root/RoboTwin"

epochs=10
total_frames_expected=23137
num_processes=1
per_device_batch_size=8
gradient_accumulation_steps=2
global_batch=16
max_train_steps=14461
warmup_steps=1447
save_interval=7231
episodes_per_task=5
expected_tasks=10

mkdir -p "${pipeline_root}"
state_file="${pipeline_root}/state"
history_file="${pipeline_root}/state_history.log"
exit_file="${pipeline_root}/exit_code"
manifest_file="${pipeline_root}/launch.env"

set_state() {
    local state="$1"
    printf '%s\n' "${state}" > "${state_file}"
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "${state}" >> "${history_file}"
}

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

on_exit() {
    local status=$?
    if [[ "${status}" -ne 0 ]]; then
        printf '%s\n' "${status}" > "${exit_file}"
        set_state failed
        log "Pipeline failed with exit status ${status}."
    fi
}
trap on_exit EXIT

require_file() {
    [[ -f "$1" ]] || { log "ERROR: required file is missing: $1"; return 1; }
}

require_dir() {
    [[ -d "$1" ]] || { log "ERROR: required directory is missing: $1"; return 1; }
}

set_state validating_training
log "Starting preflight for ${run_id}."

require_file "${official_checkpoint}"
require_file "${task_file}"
require_file "${config_yaml}"
require_file "${baseline_root}/episodes.csv"
require_file "${baseline_root}/summary.csv"
require_file "${baseline_root}/run.env"
require_file "${baseline_root}/tasks.txt"
require_file "${data_root}/validation_report.json"
require_file "${bert_path}/config.json"
require_file "${dino_path}/config.json"
require_file "${starvla_python}"
require_file "${robotwin_python}"
require_dir "${robotwin_path}"
require_file /etc/vulkan/icd.d/my_nvidia_icd.json

if [[ -e "${output_dir}" ]]; then
    log "ERROR: training output already exists: ${output_dir}"
    exit 1
fi
if [[ -e "${eval_root}" ]]; then
    log "ERROR: evaluation output already exists: ${eval_root}"
    exit 1
fi
if [[ -e "${backup_root}" || -e "${backup_staging}" ]]; then
    log "ERROR: backup destination already exists: ${backup_root} or ${backup_staging}"
    exit 1
fi
if pgrep -af '[t]rain_robotwin_clean_act_pi05_recipe.py' >/dev/null; then
    log "ERROR: another RoboTwin training process is active."
    pgrep -af '[t]rain_robotwin_clean_act_pi05_recipe.py' || true
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
    log "ERROR: GPU 0 already has an active compute process."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    exit 1
fi

official_checkpoint_sha256="$(sha256sum "${official_checkpoint}" | awk '{print $1}')"
if [[ "${official_checkpoint_sha256}" != "${official_checkpoint_sha256_expected}" ]]; then
    log "ERROR: official checkpoint SHA-256 mismatch: ${official_checkpoint_sha256}"
    exit 1
fi

"${starvla_python}" - \
    "${data_root}" \
    "${task_file}" \
    "${baseline_root}/episodes.csv" \
    "${baseline_root}/summary.csv" \
    "${baseline_root}/run.env" \
    "${baseline_root}/tasks.txt" \
    "${official_checkpoint}" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

data_root = Path(sys.argv[1])
task_file = Path(sys.argv[2])
baseline_csv = Path(sys.argv[3])
baseline_summary_csv = Path(sys.argv[4])
baseline_run_env = Path(sys.argv[5])
baseline_tasks_file = Path(sys.argv[6])
official_checkpoint = Path(sys.argv[7]).resolve()
tasks = [
    line.strip()
    for line in task_file.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(tasks) != 10 or len(set(tasks)) != 10:
    raise SystemExit(f"expected 10 unique tasks, found {len(tasks)} entries")

report = json.loads((data_root / "validation_report.json").read_text(encoding="utf-8"))
expected_report = {"status": "passed", "tasks": 10, "episodes": 100, "frames": 23137}
for key, expected in expected_report.items():
    if report.get(key) != expected:
        raise SystemExit(f"validation_report {key}={report.get(key)!r}, expected {expected!r}")

depth_keys = {
    "observation.depths.cam_high",
    "observation.depths.cam_left_wrist",
    "observation.depths.cam_right_wrist",
}
total_frames = 0
for task in tasks:
    info_path = data_root / "Clean" / task / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("total_episodes", -1)) != 10:
        raise SystemExit(f"{task}: expected 10 episodes, found {info.get('total_episodes')}")
    missing = depth_keys.difference(info.get("features", {}))
    if missing:
        raise SystemExit(f"{task}: missing depth features {sorted(missing)}")
    total_frames += int(info["total_frames"])
if total_frames != 23137:
    raise SystemExit(f"expected 23137 frames, found {total_frames}")

baseline_rows = list(csv.DictReader(baseline_csv.open(newline="", encoding="utf-8")))
counts = Counter(row["task"] for row in baseline_rows)
if len(baseline_rows) != 50 or counts != Counter({task: 5 for task in tasks}):
    raise SystemExit(f"invalid baseline episode manifest: rows={len(baseline_rows)}, counts={dict(counts)}")
baseline_tasks = [line.strip() for line in baseline_tasks_file.read_text().splitlines() if line.strip()]
if baseline_tasks != tasks:
    raise SystemExit("official baseline task list does not match the training/evaluation task list")
run_env = {}
for line in baseline_run_env.read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        run_env[key] = value
if Path(run_env.get("source_checkpoint", "")).resolve() != official_checkpoint:
    raise SystemExit(f"baseline source checkpoint is not the pinned official checkpoint: {run_env.get('source_checkpoint')!r}")
if int(run_env.get("test_num_per_task", -1)) != 5 or int(run_env.get("robotwin_seed", -1)) != 0:
    raise SystemExit("official baseline must use five episodes per task and ROBOTWIN_SEED=0")
summary_rows = list(csv.DictReader(baseline_summary_csv.open(newline="", encoding="utf-8")))
if sum(int(row["successes"]) for row in summary_rows) != 24 or sum(int(row["trials"]) for row in summary_rows) != 50:
    raise SystemExit("official baseline summary must be the verified 24/50 result")
print("preflight dataset and baseline manifests validated")
PY

available_root_kib="$(df -Pk /root | awk 'NR==2 {print $4}')"
available_mnt_kib="$(df -Pk /mnt | awk 'NR==2 {print $4}')"
minimum_root_kib=$((12 * 1024 * 1024))
minimum_mnt_kib=$((10 * 1024 * 1024))
if (( available_root_kib < minimum_root_kib )); then
    log "ERROR: /root has less than 12 GiB available."
    exit 1
fi
if (( available_mnt_kib < minimum_mnt_kib )); then
    log "ERROR: /mnt has less than 10 GiB available."
    exit 1
fi

cat > "${manifest_file}" <<EOF
run_id=${run_id}
started_at=$(date --iso-8601=seconds)
repo_root=${REPO_ROOT}
official_checkpoint=${official_checkpoint}
official_checkpoint_sha256=${official_checkpoint_sha256}
data_root=${data_root}
task_file=${task_file}
baseline_root=${baseline_root}
config_yaml=${config_yaml}
train_epochs=${epochs}
total_frames=${total_frames_expected}
num_processes=${num_processes}
per_device_batch_size=${per_device_batch_size}
gradient_accumulation_steps=${gradient_accumulation_steps}
global_batch=${global_batch}
max_train_steps=${max_train_steps}
warmup_steps=${warmup_steps}
save_interval=${save_interval}
train_modules=depth_fusion.cross_attention,depth_fusion.depth_gate
depth_encoder_lr=1.0e-04
depth_fusion_lr=1.0e-04
vision_language_interaction_lr=5.0e-06
action_head_decoder_lr=1.0e-05
ema_decay=0.999
eval_checkpoint=final_model/ema_ema_pytorch_model.pt
eval_depth_input_mode=real
eval_tasks=${expected_tasks}
eval_episodes_per_task=${episodes_per_task}
eval_seed=0
backup_root=${backup_root}
EOF
cp "${task_file}" "${pipeline_root}/tasks.txt"
cp "${config_yaml}" "${pipeline_root}/training_config.source.yaml"
cp "${SCRIPT_DIR}/train_then_eval_backup.sh" "${pipeline_root}/"
cp "${SCRIPT_DIR}/start_train_then_eval_backup.sh" "${pipeline_root}/"
source_snapshot="${pipeline_root}/source_snapshot"
source_hashes="${pipeline_root}/source_sha256.txt"
source_files=(
    scripts/robotwin/train_then_eval_backup.sh
    scripts/robotwin/start_train_then_eval_backup.sh
    scripts/robotwin/train.sh
    scripts/robotwin/evaluate.sh
    scripts/robotwin/start_eval.sh
    scripts/robotwin/eval_task.sh
    scripts/robotwin/run_policy_server.sh
    scripts/robotwin/run_legacy_eval.py
    scripts/robotwin/summarize_eval_logs.py
    scripts/robotwin/rgbd_tasks_10.txt
    experiments/robotwin/configs/clean50_depth.yaml
    experiments/robotwin/configs/modality_depth.json
    experiments/robotwin/evaluation/deploy_policy.yml
    experiments/robotwin/evaluation/model2robotwin_interface.py
    turbovla/models/configuration.py
    turbovla/models/depth_encoder.py
    turbovla/models/depth_fusion.py
    turbovla/models/turbovla.py
    third_party/starvla_runtime/starVLA/model/framework/VLM4A/TurboVLA.py
    third_party/starvla_runtime/starVLA/training/train_robotwin_clean_act_pi05_recipe.py
    third_party/starvla_runtime/starVLA/training/train_starvla.py
    third_party/starvla_runtime/starVLA/training/trainer_utils/trainer_tools.py
)
: > "${source_hashes}"
for source_file in "${source_files[@]}"; do
    require_file "${source_file}"
    mkdir -p "${source_snapshot}/$(dirname "${source_file}")"
    cp -a "${source_file}" "${source_snapshot}/${source_file}"
    sha256sum "${source_file}" >> "${source_hashes}"
done

export ROBOTWIN_DATA_ROOT="${data_root}"
export BERT_MODEL_PATH="${bert_path}"
export DINOV3_MODEL_PATH="${dino_path}"
export TURBOVLA_INIT_CKPT="${official_checkpoint}"
export STARVLA_PYTHON="${starvla_python}"
export CONFIG_YAML="${config_yaml}"
export RUN_ID="${run_id}"
export RUN_ROOT_DIR="${run_root_dir}"
export CUDA_VISIBLE_DEVICES=0
export NUM_PROCESSES="${num_processes}"
export MAIN_PROCESS_PORT=29630
export PER_DEVICE_BATCH_SIZE="${per_device_batch_size}"
export GRADIENT_ACCUMULATION_STEPS="${gradient_accumulation_steps}"
export MAX_TRAIN_STEPS="${max_train_steps}"
export WARMUP_STEPS="${warmup_steps}"
export SAVE_INTERVAL="${save_interval}"
export LOGGING_FREQUENCY=25
export LEARNING_RATE=5.0e-05
export DEPTH_LEARNING_RATE=1.0e-04
export INTERACTION_LEARNING_RATE=5.0e-06
export ACTION_DECODER_LEARNING_RATE=1.0e-05
export EMA_DECAY=0.999
export EMA_DEVICE=cuda
export WANDB_MODE=disabled
export LD_LIBRARY_PATH="/root/miniconda3/envs/myconda/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

set_state training
log "Starting 10-epoch training: steps=${max_train_steps}, batch=${per_device_batch_size}, accumulation=${gradient_accumulation_steps}."
bash "${SCRIPT_DIR}/train.sh" \
    --trainer.train_modules \
    "depth_fusion.cross_attention,depth_fusion.depth_gate"

set_state validating_training
final_model="${output_dir}/final_model/pytorch_model.pt"
final_ema="${output_dir}/final_model/ema_ema_pytorch_model.pt"
midpoint_model="${output_dir}/checkpoints/steps_${save_interval}_pytorch_model.pt"
midpoint_ema="${output_dir}/checkpoints/steps_${save_interval}_ema_pytorch_model.pt"
for checkpoint in "${final_model}" "${final_ema}" "${midpoint_model}" "${midpoint_ema}"; do
    require_file "${checkpoint}"
    size_bytes="$(stat -c '%s' "${checkpoint}")"
    if (( size_bytes < 500000000 )); then
        log "ERROR: checkpoint is unexpectedly small (${size_bytes} bytes): ${checkpoint}"
        exit 1
    fi
done
require_file "${output_dir}/config.full.yaml"
require_file "${output_dir}/config.yaml"
require_file "${output_dir}/dataset_statistics.json"

"${starvla_python}" - "${output_dir}/config.full.yaml" "${official_checkpoint}" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
expected_checkpoint = str(Path(sys.argv[2]).resolve())
checks = {
    "max_train_steps": (int(cfg.trainer.max_train_steps), 14461),
    "num_warmup_steps": (int(cfg.trainer.num_warmup_steps), 1447),
    "save_interval": (int(cfg.trainer.save_interval), 7231),
    "per_device_batch_size": (int(cfg.datasets.vla_data.per_device_batch_size), 8),
    "gradient_accumulation_steps": (int(cfg.trainer.gradient_accumulation_steps), 2),
    "pretrained_ckpt": (str(Path(cfg.framework.initialization.pretrained_ckpt).resolve()), expected_checkpoint),
}
for name, (actual, expected) in checks.items():
    if actual != expected:
        raise SystemExit(f"{name}={actual!r}, expected {expected!r}")
actual_modules = str(cfg.trainer.train_modules).replace(" ", "")
expected_modules = "depth_fusion.cross_attention,depth_fusion.depth_gate"
if actual_modules != expected_modules:
    raise SystemExit(f"train_modules={actual_modules!r}, expected {expected_modules!r}")
if not bool(cfg.framework.depth.enabled) or not bool(cfg.framework.depth_fusion.enabled):
    raise SystemExit("depth encoder and fusion must both be enabled")
print("final training configuration validated")
PY

{
    sha256sum "${official_checkpoint}"
    sha256sum "${midpoint_model}" "${midpoint_ema}" "${final_model}" "${final_ema}"
} > "${pipeline_root}/checkpoint_sha256.txt"
log "Training completed and final EMA checkpoint validated."

mkdir -p "${eval_root}/logs"
cp "${task_file}" "${eval_root}/tasks.txt"
cat > "${eval_root}/run.env" <<EOF
run_id=${run_id}
checkpoint=${final_ema}
checkpoint_sha256=$(sha256sum "${final_ema}" | awk '{print $1}')
depth_input_mode=real
test_num_per_task=${episodes_per_task}
robotwin_seed=0
jobs_per_gpu=1
gpu=0
policy_name=${eval_name}
started_at=$(date --iso-8601=seconds)
EOF

export ROBOTWIN_PATH="${robotwin_path}"
export STARVLA_PYTHON="${starvla_python}"
export ROBOTWIN_PYTHON="${robotwin_python}"
export ROBOTWIN_TEST_NUM="${episodes_per_task}"
export ROBOTWIN_SEED=0
export ROBOTWIN_DEPTH_INPUT_MODE=real
export ROBOTWIN_JOBS_PER_GPU=1
export ROBOTWIN_LOG_ROOT="${eval_root}/logs"
export ROBOTWIN_BASE_PORT=7420
export ROBOTWIN_SERVER_TIMEOUT=600
export ROBOTWIN_POLICY_NAME=model2robotwin_interface
export POLICY_NAME="${eval_name}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

set_state evaluating
log "Starting RoboTwin evaluation: 10 tasks x 5 episodes, real depth, seed 0."
bash "${SCRIPT_DIR}/evaluate.sh" "${final_ema}" "${task_file}"

set_state summarizing
"${starvla_python}" "${SCRIPT_DIR}/summarize_eval_logs.py" \
    "${eval_root}/logs" \
    --episodes-csv "${eval_root}/episodes.csv" \
    --summary-csv "${eval_root}/summary.csv"

"${starvla_python}" - \
    "${baseline_root}/episodes.csv" \
    "${baseline_root}/summary.csv" \
    "${eval_root}/episodes.csv" \
    "${eval_root}/summary.csv" \
    "${task_file}" \
    "${eval_root}" <<'PY'
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

baseline_episodes_path = Path(sys.argv[1])
baseline_summary_path = Path(sys.argv[2])
new_episodes_path = Path(sys.argv[3])
new_summary_path = Path(sys.argv[4])
task_file = Path(sys.argv[5])
output_root = Path(sys.argv[6])
tasks = [line.strip() for line in task_file.read_text().splitlines() if line.strip() and not line.startswith("#")]

def read_rows(path):
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))

baseline_episodes = read_rows(baseline_episodes_path)
new_episodes = read_rows(new_episodes_path)
expected_counts = Counter({task: 5 for task in tasks})
new_counts = Counter(row["task"] for row in new_episodes)
if len(new_episodes) != 50 or new_counts != expected_counts:
    raise SystemExit(f"expected 50 episodes and five per task, got rows={len(new_episodes)}, counts={dict(new_counts)}")

baseline_seeds = defaultdict(list)
new_seeds = defaultdict(list)
for row in baseline_episodes:
    baseline_seeds[row["task"]].append(int(row["seed"]))
for row in new_episodes:
    new_seeds[row["task"]].append(int(row["seed"]))
seed_match = {task: baseline_seeds[task] == new_seeds[task] for task in tasks}
comparison = {
    task: {"baseline": baseline_seeds[task], "new": new_seeds[task], "match": seed_match[task]}
    for task in tasks
}
(output_root / "seed_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
if not all(seed_match.values()):
    mismatches = {task: comparison[task] for task in tasks if not seed_match[task]}
    raise SystemExit(f"evaluation seeds do not match the official baseline: {json.dumps(mismatches)}")

baseline_summary = {row["task"]: row for row in read_rows(baseline_summary_path)}
new_summary = {row["task"]: row for row in read_rows(new_summary_path)}
comparison_rows = []
baseline_total = 0
new_total = 0
for task in tasks:
    baseline_successes = int(baseline_summary[task]["successes"])
    new_successes = int(new_summary[task]["successes"])
    baseline_total += baseline_successes
    new_total += new_successes
    comparison_rows.append({
        "task": task,
        "baseline_successes": baseline_successes,
        "rgbd_successes": new_successes,
        "trials": 5,
        "baseline_rate": baseline_successes / 5,
        "rgbd_rate": new_successes / 5,
        "delta_rate": (new_successes - baseline_successes) / 5,
    })
comparison_rows.append({
    "task": "OVERALL",
    "baseline_successes": baseline_total,
    "rgbd_successes": new_total,
    "trials": 50,
    "baseline_rate": baseline_total / 50,
    "rgbd_rate": new_total / 50,
    "delta_rate": (new_total - baseline_total) / 50,
})
comparison_path = output_root / "baseline_comparison.csv"
with comparison_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=comparison_rows[0].keys())
    writer.writeheader()
    writer.writerows(comparison_rows)
(output_root / "result.json").write_text(
    json.dumps({
        "tasks": 10,
        "episodes": 50,
        "seeds_match_baseline": True,
        "baseline_successes": baseline_total,
        "baseline_rate": baseline_total / 50,
        "rgbd_successes": new_total,
        "rgbd_rate": new_total / 50,
        "delta_successes": new_total - baseline_total,
        "delta_rate": (new_total - baseline_total) / 50,
    }, indent=2) + "\n",
    encoding="utf-8",
)
print(f"validated matched seeds and results: baseline={baseline_total}/50 rgbd={new_total}/50")
PY

"${starvla_python}" - "${eval_root}/logs" <<'PY'
import re
import sys
from pathlib import Path

logs = sorted(Path(sys.argv[1]).glob("*_eval.log"))
if len(logs) != 10:
    raise SystemExit(f"expected 10 evaluation logs, found {len(logs)}")
bad = []
for path in logs:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", path.read_text(encoding="utf-8", errors="replace"))
    if "depth_input_mode: real" not in text or "depth_observations=True" not in text:
        bad.append(path.name)
if bad:
    raise SystemExit(f"real depth was not confirmed in logs: {bad}")
print("all 10 evaluation logs confirm real depth observations")
PY

artifact_root="${eval_root}/robotwin_artifacts"
mkdir -p "${artifact_root}"
while IFS= read -r task; do
    [[ -n "${task}" ]] || continue
    source_root="${robotwin_path}/eval_result/${task}/model2robotwin_interface/demo_clean/${eval_name}"
    require_dir "${source_root}"
    mkdir -p "${artifact_root}/${task}"
    cp -a "${source_root}/." "${artifact_root}/${task}/"
done < "${task_file}"
find "${artifact_root}" -type f -print0 | sort -z | xargs -0 sha256sum > "${eval_root}/robotwin_artifacts_sha256.txt"
printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "${eval_root}/run.env"
log "Evaluation summaries and RoboTwin artifacts validated."

set_state backing_up
log "Copying checkpoints, logs, configuration, and evaluation outputs to ${backup_root}."
backup_manifest="${pipeline_root}/backup_sha256.txt"
: > "${backup_manifest}"
while IFS= read -r -d '' source_path; do
    relative_path="${source_path#${output_dir}/}"
    printf '%s  training/%s\n' "$(sha256sum "${source_path}" | awk '{print $1}')" "${relative_path}" \
        >> "${backup_manifest}"
done < <(find "${output_dir}" -type f -print0 | sort -z)
while IFS= read -r -d '' source_path; do
    relative_path="${source_path#${eval_root}/}"
    printf '%s  evaluation/%s\n' "$(sha256sum "${source_path}" | awk '{print $1}')" "${relative_path}" \
        >> "${backup_manifest}"
done < <(find "${eval_root}" -type f -print0 | sort -z)
printf '%s  official_baseline_episodes.csv\n' \
    "$(sha256sum "${baseline_root}/episodes.csv" | awk '{print $1}')" >> "${backup_manifest}"
printf '%s  official_baseline_summary.csv\n' \
    "$(sha256sum "${baseline_root}/summary.csv" | awk '{print $1}')" >> "${backup_manifest}"

mkdir -p "${backup_staging}"
cp -a "${output_dir}" "${backup_staging}/training"
cp -a "${eval_root}" "${backup_staging}/evaluation"
cp -a "${pipeline_root}" "${backup_staging}/pipeline"
cp -a "${baseline_root}/episodes.csv" "${backup_staging}/official_baseline_episodes.csv"
cp -a "${baseline_root}/summary.csv" "${backup_staging}/official_baseline_summary.csv"

for relative_path in \
    "training/checkpoints/steps_${save_interval}_pytorch_model.pt" \
    "training/checkpoints/steps_${save_interval}_ema_pytorch_model.pt" \
    "training/final_model/pytorch_model.pt" \
    "training/final_model/ema_ema_pytorch_model.pt"; do
    source_hash="$(sha256sum "${output_dir}/${relative_path#training/}" | awk '{print $1}')"
    backup_hash="$(sha256sum "${backup_staging}/${relative_path}" | awk '{print $1}')"
    if [[ "${source_hash}" != "${backup_hash}" ]]; then
        log "ERROR: backup SHA-256 mismatch for ${relative_path}"
        exit 1
    fi
done
(cd "${backup_staging}" && sha256sum --check "pipeline/backup_sha256.txt") \
    > "${backup_staging}/pipeline/backup_verification.log"
mv "${backup_staging}" "${backup_root}"

printf '0\n' > "${exit_file}"
set_state complete
log "Pipeline complete. Backup verified at ${backup_root}."
cp -a "${state_file}" "${history_file}" "${exit_file}" "${pipeline_root}/pipeline.log" "${backup_root}/pipeline/"
trap - EXIT
