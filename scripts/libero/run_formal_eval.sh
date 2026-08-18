#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="${TURBOVLA_ROOT:-/root/TurboVLA-repro}"
RUN_ID="${1:-formal_bf16_seed7_$(date +%Y%m%d_%H%M%S)}"
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
SUCCESS_VIDEOS_PER_TASK="${SUCCESS_VIDEOS_PER_TASK:--1}"

CONDA_ROOT="/root/miniconda3"
CONDA_ENV="myconda"
DINOV3_DIR="$ROOT_DIR/pretrained/dinov3-vitb16-pretrain-lvd1689m-converted"
BERT_DIR="$ROOT_DIR/pretrained/bert-base-uncased"
STATS_PATH="$ROOT_DIR/experiments/libero/configs/libero_all4_stats.json"
LIBERO_ROOT="/root/LIBERO"

RUN_DIR="$ROOT_DIR/outputs/evaluation/$RUN_ID"
PARTIAL_DIR="$RUN_DIR/partials"
AGGREGATE_DIR="$RUN_DIR/aggregates"
LOG_DIR="$RUN_DIR/logs"
FAILED_DIR="$RUN_DIR/failed"
VIDEO_DIR="$RUN_DIR/videos"
JOBS_FILE="$RUN_DIR/jobs.tsv"
PROGRESS_FILE="$RUN_DIR/progress.tsv"
STATUS_FILE="$RUN_DIR/status.txt"

if [[ ! "$PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "PARALLEL_JOBS must be a positive integer, got: $PARALLEL_JOBS" >&2
    exit 2
fi
if [[ "$SAVE_VIDEO" != "true" && "$SAVE_VIDEO" != "false" ]]; then
    echo "SAVE_VIDEO must be true or false, got: $SAVE_VIDEO" >&2
    exit 2
fi
if [[ ! "$SUCCESS_VIDEOS_PER_TASK" =~ ^(-1|[0-9]+)$ ]]; then
    echo "SUCCESS_VIDEOS_PER_TASK must be -1 or a non-negative integer, got: $SUCCESS_VIDEOS_PER_TASK" >&2
    exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
    exit 2
fi

mkdir -p "$PARTIAL_DIR" "$AGGREGATE_DIR" "$LOG_DIR" "$FAILED_DIR" "$VIDEO_DIR"
START_EPOCH="$(date +%s)"
printf 'RUNNING\t%s\tepoch=%s\n' "$(date --iso-8601=seconds)" "$START_EPOCH" > "$STATUS_FILE"

on_exit() {
    local exit_code=$?
    if (( exit_code != 0 )); then
        printf 'FAILED\t%s\texit_code=%s\n' "$(date --iso-8601=seconds)" "$exit_code" >> "$STATUS_FILE"
    fi
}
trap on_exit EXIT

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

required_paths=(
    "$DINOV3_DIR/config.json"
    "$DINOV3_DIR/model.safetensors"
    "$DINOV3_DIR/preprocessor_config.json"
    "$BERT_DIR/config.json"
    "$BERT_DIR/model.safetensors"
    "$BERT_DIR/tokenizer.json"
    "$STATS_PATH"
    "$LIBERO_ROOT"
    "$ROOT_DIR/ckpt/libero/spatial.pth"
    "$ROOT_DIR/ckpt/libero/object.pth"
    "$ROOT_DIR/ckpt/libero/goal.pth"
    "$ROOT_DIR/ckpt/libero/long.pth"
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required path is missing: $required_path" >&2
        exit 2
    fi
done

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'parallel_jobs=%s\n' "$PARALLEL_JOBS"
    printf 'precision=bf16\nseed=7\nchunk_size=12\nnum_open_loop_steps=12\n'
    printf 'num_trials_per_task=50\ntasks_per_suite=10\ntotal_episodes=2000\n'
    printf 'save_video=%s\n' "$SAVE_VIDEO"
    printf 'success_videos_per_task=%s\n' "$SUCCESS_VIDEOS_PER_TASK"
    python --version
    python - <<'PY'
import torch
import transformers
import mujoco

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"mujoco={mujoco.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
PY
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "$RUN_DIR/manifest.txt"

python -m pip freeze > "$RUN_DIR/environment_pip_freeze.txt"
sha256sum \
    "$DINOV3_DIR/model.safetensors" \
    "$BERT_DIR/model.safetensors" \
    "$ROOT_DIR/ckpt/libero/spatial.pth" \
    "$ROOT_DIR/ckpt/libero/object.pth" \
    "$ROOT_DIR/ckpt/libero/goal.pth" \
    "$ROOT_DIR/ckpt/libero/long.pth" \
    > "$RUN_DIR/input_sha256.txt"

: > "$JOBS_FILE"
for task_id in {0..9}; do printf 'long\tlibero_10\t%s\n' "$task_id" >> "$JOBS_FILE"; done
for task_id in {0..9}; do printf 'goal\tlibero_goal\t%s\n' "$task_id" >> "$JOBS_FILE"; done
for task_id in {0..9}; do printf 'object\tlibero_object\t%s\n' "$task_id" >> "$JOBS_FILE"; done
for task_id in {0..9}; do printf 'spatial\tlibero_spatial\t%s\n' "$task_id" >> "$JOBS_FILE"; done
: > "$PROGRESS_FILE"

validate_result() {
    local result_path=$1
    local expected_suite=$2
    local expected_task_id=$3
    python - "$result_path" "$expected_suite" "$expected_task_id" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
suite = sys.argv[2]
task_id = int(sys.argv[3])
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["task_suite_name"] == suite
assert payload["requested_task_ids"] == [task_id]
assert payload["num_trials_per_task"] == 50
assert payload["seed"] == 7
assert payload["precision"] == "bf16"
assert payload["num_open_loop_steps"] == 12
assert payload["save_video"] == (os.environ["SAVE_VIDEO"] == "true")
assert int(payload["success_videos_per_task"]) == int(os.environ["SUCCESS_VIDEOS_PER_TASK"])
assert len(payload["tasks"]) == 1
task = payload["tasks"][0]
assert int(task["task_id"]) == task_id
assert int(task["episodes"]) == 50
assert int(payload["total_episodes"]) == 50
assert int(payload["total_successes"]) == int(task["successes"])
PY
}

run_one() {
    set -uo pipefail
    local checkpoint_tag=$1
    local suite=$2
    local task_id=$3
    local task_tag
    task_tag="$(printf '%02d' "$task_id")"
    local suite_dir="$PARTIAL_DIR/$suite"
    local final_result="$suite_dir/task${task_tag}.json"
    local temp_result="$suite_dir/.task${task_tag}.json.tmp.$$"
    local eval_log="$LOG_DIR/${suite}_task${task_tag}.log"
    local console_log="$LOG_DIR/${suite}_task${task_tag}.console.log"

    mkdir -p "$suite_dir"
    if [[ -f "$final_result" ]] && validate_result "$final_result" "$suite" "$task_id" 2>/dev/null; then
        printf 'SKIP\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$suite" "$task_id" "$final_result" >> "$PROGRESS_FILE"
        return 0
    fi

    printf 'START\t%s\t%s\t%s\tpid=%s\n' "$(date --iso-8601=seconds)" "$suite" "$task_id" "$$" >> "$PROGRESS_FILE"
    set +e
    python "$ROOT_DIR/experiments/libero/evaluate.py" \
        --ckpt_path "$ROOT_DIR/ckpt/libero/$checkpoint_tag.pth" \
        --libero_root "$LIBERO_ROOT" \
        --dinov3_path "$DINOV3_DIR" \
        --bert_path "$BERT_DIR" \
        --stats_path "$STATS_PATH" \
        --stats_key libero_all4_no_noops \
        --task_suite_name "$suite" \
        --task_ids "$task_id" \
        --num_trials_per_task 50 \
        --num_steps_wait 10 \
        --num_open_loop_steps 12 \
        --chunk_size 12 \
        --env_img_res 256 \
        --seed 7 \
        --control_mode relative \
        --mujoco_gl osmesa \
        --pyopengl_platform osmesa \
        --precision bf16 \
        --allow_hf_download false \
        --save_video "$SAVE_VIDEO" \
        --success_videos_per_task "$SUCCESS_VIDEOS_PER_TASK" \
        --video_out_path "$VIDEO_DIR" \
        --result_json_path "$temp_result" \
        --log_path "$eval_log" \
        > "$console_log" 2>&1
    local eval_exit=$?
    set -e
    if (( eval_exit != 0 )); then
        if [[ -e "$temp_result" ]]; then
            mv "$temp_result" "$FAILED_DIR/${suite}_task${task_tag}.json.tmp.$$"
        fi
        printf 'FAIL\t%s\t%s\t%s\texit=%s\n' "$(date --iso-8601=seconds)" "$suite" "$task_id" "$eval_exit" >> "$PROGRESS_FILE"
        return "$eval_exit"
    fi
    if ! validate_result "$temp_result" "$suite" "$task_id" >> "$console_log" 2>&1; then
        mv "$temp_result" "$FAILED_DIR/${suite}_task${task_tag}.invalid.json"
        printf 'FAIL\t%s\t%s\t%s\tinvalid_result\n' "$(date --iso-8601=seconds)" "$suite" "$task_id" >> "$PROGRESS_FILE"
        return 3
    fi
    mv "$temp_result" "$final_result"
    printf 'DONE\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$suite" "$task_id" "$final_result" >> "$PROGRESS_FILE"
}

export ROOT_DIR DINOV3_DIR BERT_DIR STATS_PATH LIBERO_ROOT
export RUN_DIR PARTIAL_DIR LOG_DIR FAILED_DIR VIDEO_DIR PROGRESS_FILE SAVE_VIDEO SUCCESS_VIDEOS_PER_TASK
export -f validate_result run_one

echo "Starting $RUN_ID with $PARALLEL_JOBS parallel task processes"
if ! xargs -P "$PARALLEL_JOBS" -n 3 bash -c 'run_one "$@"' _ < "$JOBS_FILE"; then
    echo "At least one task process failed; see $PROGRESS_FILE and $LOG_DIR" >&2
    exit 1
fi

done_count="$(find "$PARTIAL_DIR" -type f -name 'task*.json' | wc -l)"
if [[ "$done_count" -ne 40 ]]; then
    echo "Expected 40 completed task JSON files, found $done_count" >&2
    exit 1
fi

for suite in libero_spatial libero_object libero_goal libero_10; do
    mapfile -t suite_results < <(find "$PARTIAL_DIR/$suite" -maxdepth 1 -type f -name 'task*.json' | sort)
    if [[ "${#suite_results[@]}" -ne 10 ]]; then
        echo "Expected 10 task results for $suite, found ${#suite_results[@]}" >&2
        exit 1
    fi
    python "$ROOT_DIR/scripts/libero/aggregate_results.py" \
        "${suite_results[@]}" \
        --output "$AGGREGATE_DIR/$suite.json"
done

python - "$AGGREGATE_DIR" "$RUN_DIR/overall_summary.json" "$SAVE_VIDEO" "$SUCCESS_VIDEOS_PER_TASK" <<'PY'
import json
import sys
from pathlib import Path

aggregate_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
save_video = sys.argv[3] == "true"
success_videos_per_task = int(sys.argv[4])
suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
suites = []
for name in suite_order:
    payload = json.loads((aggregate_dir / f"{name}.json").read_text(encoding="utf-8"))
    suites.append(
        {
            "task_suite_name": name,
            "total_episodes": int(payload["total_episodes"]),
            "total_successes": int(payload["total_successes"]),
            "success_rate": float(payload["overall_success_rate"]),
        }
    )
total_episodes = sum(item["total_episodes"] for item in suites)
total_successes = sum(item["total_successes"] for item in suites)
result = {
    "protocol": {
        "precision": "bf16",
        "seed": 7,
        "chunk_size": 12,
        "num_open_loop_steps": 12,
        "num_trials_per_task": 50,
        "save_video": save_video,
        "success_videos_per_task": success_videos_per_task,
    },
    "suites": suites,
    "total_episodes": total_episodes,
    "total_successes": total_successes,
    "overall_success_rate": total_successes / total_episodes,
    "macro_average_success_rate": sum(item["success_rate"] for item in suites) / len(suites),
}
assert total_episodes == 2000
output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
PY

python - "$PARTIAL_DIR" "$VIDEO_DIR" "$RUN_DIR/video_inventory.json" "$SAVE_VIDEO" "$SUCCESS_VIDEOS_PER_TASK" <<'PY'
import json
import sys
from pathlib import Path

partial_dir = Path(sys.argv[1])
video_dir = Path(sys.argv[2])
output_path = Path(sys.argv[3])
save_video = sys.argv[4] == "true"
success_limit = int(sys.argv[5])
checkpoint_by_suite = {
    "libero_spatial": "spatial",
    "libero_object": "object",
    "libero_goal": "goal",
    "libero_10": "long",
}

tasks = []
expected_total = 0
actual_total = 0
total_size_bytes = 0
for result_path in sorted(partial_dir.glob("*/task*.json")):
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    suite = payload["task_suite_name"]
    task = payload["tasks"][0]
    task_id = int(task["task_id"])
    episodes = int(task["episodes"])
    successes = int(task["successes"])
    failures = episodes - successes
    if save_video:
        expected_success = successes if success_limit < 0 else min(successes, success_limit)
        expected_failure = failures
    else:
        expected_success = 0
        expected_failure = 0

    task_video_dir = video_dir / checkpoint_by_suite[suite] / suite
    success_files = sorted(task_video_dir.glob(f"task{task_id:02d}_ep*_success.mp4"))
    failure_files = sorted(task_video_dir.glob(f"task{task_id:02d}_ep*_failure.mp4"))
    if len(success_files) != expected_success or len(failure_files) != expected_failure:
        raise ValueError(
            f"video count mismatch for {suite} task {task_id}: "
            f"success {len(success_files)}/{expected_success}, "
            f"failure {len(failure_files)}/{expected_failure}"
        )
    files = success_files + failure_files
    size_bytes = sum(path.stat().st_size for path in files)
    expected_total += expected_success + expected_failure
    actual_total += len(files)
    total_size_bytes += size_bytes
    tasks.append(
        {
            "task_suite_name": suite,
            "task_id": task_id,
            "episodes": episodes,
            "successes": successes,
            "failures": failures,
            "success_videos": len(success_files),
            "failure_videos": len(failure_files),
            "video_size_bytes": size_bytes,
        }
    )

all_videos = list(video_dir.rglob("*.mp4"))
if len(all_videos) != actual_total or actual_total != expected_total:
    raise ValueError(
        f"global video count mismatch: files={len(all_videos)}, "
        f"accounted={actual_total}, expected={expected_total}"
    )
inventory = {
    "save_video": save_video,
    "success_videos_per_task": success_limit,
    "all_failures_required": save_video,
    "video_count": actual_total,
    "video_size_bytes": total_size_bytes,
    "tasks": tasks,
}
output_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(
    f"Video inventory verified: {actual_total} files, "
    f"{total_size_bytes} bytes"
)
PY

END_EPOCH="$(date +%s)"
printf 'EVALUATION_COMPLETED\t%s\tepoch=%s\telapsed_seconds=%s\n' \
    "$(date --iso-8601=seconds)" "$END_EPOCH" "$((END_EPOCH - START_EPOCH))" \
    >> "$STATUS_FILE"

ARCHIVE="$ROOT_DIR/outputs/reproduction/${RUN_ID}.tar.gz"
ARCHIVE_SHA="$ARCHIVE.sha256"
tar -C "$(dirname "$RUN_DIR")" -czf "$ARCHIVE" "$(basename "$RUN_DIR")"
sha256sum "$ARCHIVE" > "$ARCHIVE_SHA"
archive_size="$(stat --printf='%s' "$ARCHIVE")"
printf 'archive=%s\narchive_size_bytes=%s\n' "$ARCHIVE" "$archive_size"
cat "$ARCHIVE_SHA"

if [[ -d /mnt ]]; then
    available_bytes="$(df --output=avail -B1 /mnt | tail -n 1 | tr -d ' ')"
    if (( archive_size < available_bytes )); then
        destination="/mnt/$(basename "$ARCHIVE")"
        if [[ -e "$destination" ]]; then
            echo "Archive destination already exists; not overwriting: $destination" >&2
        else
            cp --reflink=never --preserve=timestamps "$ARCHIVE" "$destination"
            cp --reflink=never --preserve=timestamps "$ARCHIVE_SHA" "$destination.sha256"
            source_hash="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
            destination_hash="$(sha256sum "$destination" | awk '{print $1}')"
            if [[ "$source_hash" != "$destination_hash" ]]; then
                echo "Copied archive hash mismatch" >&2
                exit 4
            fi
            stat --printf='archive_source device=%d inode=%i size=%s path=%n\n' "$ARCHIVE"
            stat --printf='archive_copy device=%d inode=%i size=%s path=%n\n' "$destination"
            echo "Copied result archive to $destination"
        fi
    else
        echo "Result archive does not fit in /mnt: size=$archive_size available=$available_bytes" >&2
    fi
fi

printf 'PACKAGING_COMPLETED\t%s\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
echo "FORMAL_EVALUATION_COMPLETE run_id=$RUN_ID"
