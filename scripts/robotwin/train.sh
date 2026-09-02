#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/starvla_runtime:${PYTHONPATH:-}"

: "${ROBOTWIN_DATA_ROOT:?Set ROBOTWIN_DATA_ROOT to the converted RoboTwin dataset root.}"
: "${BERT_MODEL_PATH:?Set BERT_MODEL_PATH to a local bert-base-uncased directory.}"
: "${TURBOVLA_INIT_CKPT:?Set TURBOVLA_INIT_CKPT to groundingdino_swint_ogc.pth.}"
: "${DINOV3_MODEL_PATH:?Set DINOV3_MODEL_PATH to a local DINOv3 model directory.}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,virbr0,veth}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"

config_yaml="${CONFIG_YAML:-experiments/robotwin/configs/clean50.yaml}"
run_root_dir="${RUN_ROOT_DIR:-results/Checkpoints}"
run_id="${RUN_ID:-turbovla_robotwin_clean50_dinov3large_act_pi05_bs192_100k}"
launcher_python="${STARVLA_PYTHON:-python}"
num_processes="${NUM_PROCESSES:-4}"
main_process_port="${MAIN_PROCESS_PORT:-29630}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-48}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
max_train_steps="${MAX_TRAIN_STEPS:-100000}"
warmup_steps="${WARMUP_STEPS:-1000}"
save_interval="${SAVE_INTERVAL:-5000}"
logging_frequency="${LOGGING_FREQUENCY:-50}"
learning_rate="${LEARNING_RATE:-5.0e-05}"
ema_decay="${EMA_DECAY:-0.999}"
ema_device="${EMA_DEVICE:-cuda}"
output_dir="${run_root_dir}/${run_id}"

for required_file in \
    "${DINOV3_MODEL_PATH}/config.json" \
    "${BERT_MODEL_PATH}/config.json" \
    "${TURBOVLA_INIT_CKPT}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "[ERROR] Required model file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ -e "${output_dir}" ]]; then
    echo "[ERROR] Output directory already exists: ${output_dir}" >&2
    echo "        Set RUN_ID to a new name to start another experiment." >&2
    exit 1
fi

echo "[INFO] TurboVLA | RoboTwin clean50"
echo "[INFO] data_root=${ROBOTWIN_DATA_ROOT}"
echo "[INFO] output_dir=${output_dir}"
echo "[INFO] processes=${num_processes} per_device_bs=${per_device_batch_size} grad_accum=${gradient_accumulation_steps}"
echo "[INFO] global_batch=$((num_processes * per_device_batch_size * gradient_accumulation_steps))"
echo "[INFO] lr=${learning_rate} warmup=${warmup_steps} ema_decay=${ema_decay}"

mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp third_party/starvla_runtime/starVLA/training/train_robotwin_clean_act_pi05_recipe.py "${output_dir}/"

"${launcher_python}" -m accelerate.commands.launch \
  --config_file experiments/robotwin/configs/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port "${main_process_port}" \
  third_party/starvla_runtime/starVLA/training/train_robotwin_clean_act_pi05_recipe.py \
  --config_yaml "${config_yaml}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --trainer.learning_rate.base "${learning_rate}" \
  --trainer.learning_rate.text_encoder "${learning_rate}" \
  --trainer.learning_rate.vision_encoder "${learning_rate}" \
  --trainer.learning_rate.vision_language_interaction "${learning_rate}" \
  --trainer.learning_rate.vision_projection "${learning_rate}" \
  --trainer.learning_rate.action_head "${learning_rate}" \
  --trainer.gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --trainer.ema_decay "${ema_decay}" \
  --trainer.ema_device "${ema_device}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.num_warmup_steps "${warmup_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${WANDB_PROJECT:-turbovla_robotwin}" \
  --wandb_entity "${WANDB_ENTITY:-your_wandb_entity}" \
  "$@"
