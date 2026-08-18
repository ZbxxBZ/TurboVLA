#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

eval_session="${EVAL_SESSION:-official_rgb55k_eval5}"
eval_process_pattern="${EVAL_PROCESS_PATTERN:-official_rgb55k_eval5_seed0}"
eval_root="${EVAL_ROOT:-${REPO_ROOT}/results/robotwin_eval/official_rgb55k_10tasks_5eps_seed0}"
task_file="${TASK_FILE:-${SCRIPT_DIR}/rgbd_tasks_10.txt}"
expected_tasks="${EXPECTED_TASKS:-10}"
episodes_per_task="${EPISODES_PER_TASK:-5}"
expected_seed_start="${EXPECTED_SEED_START:-100000}"
poll_seconds="${POLL_SECONDS:-30}"

data_root="${ROBOTWIN_DATA_ROOT:-/root/robotwin_rgbd_lerobot}"
epochs="${TRAIN_EPOCHS:-5}"
num_processes="${NUM_PROCESSES:-1}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-1}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-16}"
run_id="${RUN_ID:-turbovla_rgbd10_official_depth_action_ft_5ep_20260814}"
run_root_dir="${RUN_ROOT_DIR:-results/Checkpoints}"
output_dir="${run_root_dir}/${run_id}"
log_dir="${TRAIN_LOG_DIR:-${REPO_ROOT}/logs}"
train_log="${log_dir}/${run_id}.log"
exit_file="${log_dir}/${run_id}.exit"
state_file="${log_dir}/${run_id}.chain_state"

# This experiment must start from the unmodified official RGB checkpoint. Do
# not accept TURBOVLA_INIT_CKPT from the caller, because an old shell or tmux
# environment may still point at the earlier depth-only 8-epoch run.
official_checkpoint="${REPO_ROOT}/ckpt/robotwin/steps_55000_ema_model.safetensors"
bert_path="${BERT_MODEL_PATH:-${REPO_ROOT}/pretrained/bert-base-uncased}"
dino_path="${DINOV3_MODEL_PATH:-${REPO_ROOT}/pretrained/dinov3-vitl16-robotwin-checkpoint}"
python_bin="${STARVLA_PYTHON:-/root/miniconda3/envs/myconda/bin/python}"

mkdir -p "${log_dir}"
rm -f "${exit_file}"

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${train_log}"
}

fail() {
    log "ERROR: $*"
    printf 'blocked\n' > "${state_file}"
    printf '1\n' > "${exit_file}"
    exit 1
}

count_completed_episodes() {
    "${python_bin}" - "${eval_root}/logs" <<'PY'
import re
import sys
from pathlib import Path

ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
result_re = re.compile(
    r"Success rate:\s*\d+/(\d+)\s*=>\s*[0-9.]+%,\s*current seed:\s*\d+"
)
total = 0
for path in Path(sys.argv[1]).glob("*_eval.log"):
    text = ansi_escape.sub("", path.read_text(encoding="utf-8", errors="replace"))
    trials = [int(value) for value in result_re.findall(text)]
    total += max(trials, default=0)
print(total)
PY
}

[[ -f "${task_file}" ]] || fail "task file is missing: ${task_file}"
[[ -d "${eval_root}/logs" ]] || fail "evaluation log directory is missing: ${eval_root}/logs"
[[ -d "${data_root}/Clean" ]] || fail "RGB-D dataset is missing: ${data_root}/Clean"
[[ -f "${official_checkpoint}" ]] || fail "official checkpoint is missing: ${official_checkpoint}"
[[ "${official_checkpoint}" == "${REPO_ROOT}/ckpt/robotwin/steps_55000_ema_model.safetensors" ]] || \
    fail "initial checkpoint is not the pinned official checkpoint: ${official_checkpoint}"
[[ -f "${bert_path}/config.json" ]] || fail "BERT model is missing: ${bert_path}/config.json"
[[ -f "${dino_path}/config.json" ]] || fail "DINO model is missing: ${dino_path}/config.json"
[[ -x "${python_bin}" ]] || fail "training Python is not executable: ${python_bin}"

task_count="$(awk 'NF && $1 !~ /^#/' "${task_file}" | wc -l)"
[[ "${task_count}" -eq "${expected_tasks}" ]] || \
    fail "expected ${expected_tasks} tasks in ${task_file}, found ${task_count}"

"${python_bin}" - "${eval_root}/run.env" "${eval_root}/tasks.txt" "${task_file}" \
    "${official_checkpoint}" "${episodes_per_task}" <<'PY' >> "${train_log}" 2>&1 || \
    fail "official RGB evaluation manifest validation failed"
import sys
from pathlib import Path

run_env_path = Path(sys.argv[1])
evaluated_tasks_path = Path(sys.argv[2])
expected_tasks_path = Path(sys.argv[3])
official_checkpoint = str(Path(sys.argv[4]).resolve())
episodes_per_task = int(sys.argv[5])
values = {}
for line in run_env_path.read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        values[key] = value
if str(Path(values.get("source_checkpoint", "")).resolve()) != official_checkpoint:
    raise SystemExit(
        f"evaluation source checkpoint is {values.get('source_checkpoint')!r}, expected {official_checkpoint!r}"
    )
if int(values.get("test_num_per_task", -1)) != episodes_per_task:
    raise SystemExit(
        f"evaluation test_num_per_task={values.get('test_num_per_task')!r}, expected {episodes_per_task}"
    )
evaluated_tasks = [line.strip() for line in evaluated_tasks_path.read_text().splitlines() if line.strip()]
expected_tasks = [line.strip() for line in expected_tasks_path.read_text().splitlines() if line.strip()]
if evaluated_tasks != expected_tasks:
    raise SystemExit("evaluation task list does not match the requested task list")
print(f"validated official RGB evaluation manifest: checkpoint={official_checkpoint} tasks={len(expected_tasks)}")
PY

"${python_bin}" - "${data_root}" "${task_file}" <<'PY' >> "${train_log}" 2>&1 || \
    fail "RGB-D training dataset metadata validation failed"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
task_path = Path(sys.argv[2])
tasks = [line.strip() for line in task_path.read_text().splitlines() if line.strip() and not line.startswith("#")]
report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
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
    info_path = root / "Clean" / task / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("total_episodes", -1)) != 10:
        raise SystemExit(f"{task}: expected 10 episodes, found {info.get('total_episodes')}")
    missing = depth_keys.difference(info.get("features", {}))
    if missing:
        raise SystemExit(f"{task}: missing depth features {sorted(missing)}")
    total_frames += int(info["total_frames"])
if total_frames != 23137:
    raise SystemExit(f"expected 23137 frames, found {total_frames}")
print("validated RGB-D dataset: tasks=10 episodes=100 frames=23137 depth_views=3")
PY

printf 'waiting_for_evaluation\n' > "${state_file}"
log "Waiting for evaluation process ${eval_process_pattern} to finish (tmux=${eval_session})."
expected_episodes=$((expected_tasks * episodes_per_task))
while true; do
    completed="$(count_completed_episodes)"
    if [[ "${completed}" -gt "${expected_episodes}" ]]; then
        fail "evaluation produced ${completed} results; expected exactly ${expected_episodes}"
    fi
    if [[ "${completed}" -eq "${expected_episodes}" ]]; then
        if ! pgrep -f -- "${eval_process_pattern}" >/dev/null; then
            break
        fi
        log "All ${completed}/${expected_episodes} results exist; waiting for evaluation cleanup."
        sleep "${poll_seconds}"
        continue
    fi
    if ! pgrep -f -- "${eval_process_pattern}" >/dev/null; then
        fail "evaluation stopped after ${completed}/${expected_episodes} episodes"
    fi
    log "Evaluation progress: ${completed}/${expected_episodes} episodes."
    sleep "${poll_seconds}"
done

log "Evaluation session exited; regenerating and validating its summary."
"${python_bin}" "${SCRIPT_DIR}/summarize_eval_logs.py" \
    "${eval_root}/logs" \
    --episodes-csv "${eval_root}/episodes.csv" \
    --summary-csv "${eval_root}/summary.csv" >> "${train_log}" 2>&1 || \
    fail "evaluation logs could not be summarized"

"${python_bin}" - "${eval_root}/episodes.csv" "${task_file}" \
    "${episodes_per_task}" "${expected_seed_start}" "${eval_root}/seeds.json" <<'PY' >> "${train_log}" 2>&1 || \
    fail "evaluation result validation failed"
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

episodes_path = Path(sys.argv[1])
task_path = Path(sys.argv[2])
episodes_per_task = int(sys.argv[3])
seed_start = int(sys.argv[4])
seeds_path = Path(sys.argv[5])
tasks = [line.strip() for line in task_path.read_text().splitlines() if line.strip() and not line.startswith("#")]
rows = list(csv.DictReader(episodes_path.open(newline="", encoding="utf-8")))
expected_rows = len(tasks) * episodes_per_task
if len(rows) != expected_rows:
    raise SystemExit(f"expected {expected_rows} episodes, found {len(rows)}")
counts = Counter(row["task"] for row in rows)
if counts != Counter({task: episodes_per_task for task in tasks}):
    raise SystemExit(f"unexpected per-task episode counts: {dict(counts)}")
seeds = defaultdict(set)
for row in rows:
    seeds[row["task"]].add(int(row["seed"]))
duplicate_seeds = {
    task: sorted(seeds[task])
    for task in tasks
    if len(seeds[task]) != episodes_per_task
}
if duplicate_seeds:
    raise SystemExit(f"expected {episodes_per_task} unique seeds per task: {duplicate_seeds}")
too_small = {
    task: sorted(value for value in seeds[task] if value < seed_start)
    for task in tasks
    if any(value < seed_start for value in seeds[task])
}
if too_small:
    raise SystemExit(f"evaluation seeds below {seed_start}: {too_small}")
seed_manifest = {task: sorted(seeds[task]) for task in tasks}
seeds_path.write_text(json.dumps(seed_manifest, indent=2) + "\n", encoding="utf-8")
print(f"validated episodes={len(rows)} tasks={len(tasks)}")
print(f"seeds_json={seeds_path}")
print(json.dumps(seed_manifest, sort_keys=True))
PY

if [[ -e "${output_dir}" ]]; then
    fail "training output already exists: ${output_dir}"
fi
if pgrep -af 'train_robotwin_clean_act_pi05_recipe.py' >/dev/null; then
    fail "another RoboTwin training process is already running"
fi
gpu_waits=0
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do
    if [[ "${gpu_waits}" -ge 20 ]]; then
        fail "GPU still has a compute process 10 minutes after evaluation"
    fi
    log "Waiting for evaluation GPU processes to exit."
    gpu_waits=$((gpu_waits + 1))
    sleep 30
done

# This loader's balanced mixture has one sample per source frame per epoch.
total_frames="$(${python_bin} - "${data_root}" "${task_file}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tasks = [line.strip() for line in Path(sys.argv[2]).read_text().splitlines() if line.strip() and not line.startswith("#")]
total = 0
for task in tasks:
    info_path = root / "Clean" / task / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total += int(info["total_frames"])
print(total)
PY
)"
global_batch=$((num_processes * per_device_batch_size * gradient_accumulation_steps))
max_train_steps=$(((total_frames * epochs + global_batch - 1) / global_batch))
warmup_steps=$(((max_train_steps + 9) / 10))
save_interval=$(((max_train_steps + epochs - 1) / epochs))
official_checkpoint_sha256="$(sha256sum "${official_checkpoint}" | awk '{print $1}')"

cat > "${output_dir}.launch.env" <<EOF
run_id=${run_id}
official_checkpoint=${official_checkpoint}
official_checkpoint_sha256=${official_checkpoint_sha256}
data_root=${data_root}
tasks=${task_count}
episodes_per_task=${episodes_per_task}
train_epochs=${epochs}
total_frames=${total_frames}
num_processes=${num_processes}
per_device_batch_size=${per_device_batch_size}
gradient_accumulation_steps=${gradient_accumulation_steps}
global_batch=${global_batch}
max_train_steps=${max_train_steps}
warmup_steps=${warmup_steps}
save_interval=${save_interval}
train_modules=depth_encoder,depth_fusion,vision_language_interaction,action_head.decoder
EOF

log "Evaluation validated. Starting official-checkpoint RGB-D fine-tuning."
log "epochs=${epochs} frames_per_epoch=${total_frames} global_batch=${global_batch} optimizer_steps=${max_train_steps}"
printf 'training\n' > "${state_file}"

export ROBOTWIN_DATA_ROOT="${data_root}"
export BERT_MODEL_PATH="${bert_path}"
export DINOV3_MODEL_PATH="${dino_path}"
export TURBOVLA_INIT_CKPT="${official_checkpoint}"
export STARVLA_PYTHON="${python_bin}"
export RUN_ID="${run_id}"
export RUN_ROOT_DIR="${run_root_dir}"
export NUM_PROCESSES="${num_processes}"
export PER_DEVICE_BATCH_SIZE="${per_device_batch_size}"
export GRADIENT_ACCUMULATION_STEPS="${gradient_accumulation_steps}"
export MAX_TRAIN_STEPS="${max_train_steps}"
export WARMUP_STEPS="${warmup_steps}"
export SAVE_INTERVAL="${save_interval}"
export LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-25}"
export LEARNING_RATE="${LEARNING_RATE:-5.0e-05}"
export DEPTH_LEARNING_RATE="${DEPTH_LEARNING_RATE:-1.0e-04}"
export INTERACTION_LEARNING_RATE="${INTERACTION_LEARNING_RATE:-5.0e-06}"
export ACTION_DECODER_LEARNING_RATE="${ACTION_DECODER_LEARNING_RATE:-1.0e-05}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export LD_LIBRARY_PATH="/root/miniconda3/envs/myconda/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

set +e
bash "${SCRIPT_DIR}/train.sh" \
    --trainer.train_modules \
    "depth_encoder,depth_fusion,vision_language_interaction,action_head.decoder" \
    >> "${train_log}" 2>&1
status=$?
set -e
printf '%s\n' "${status}" > "${exit_file}"
if [[ "${status}" -eq 0 ]]; then
    printf 'complete\n' > "${state_file}"
    log "Training completed successfully: ${output_dir}/final_model"
else
    printf 'failed\n' > "${state_file}"
    log "Training exited with status ${status}."
fi
exit "${status}"
