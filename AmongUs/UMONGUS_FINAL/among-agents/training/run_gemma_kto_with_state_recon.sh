#!/usr/bin/env bash
# Gemma KTO training with auxiliary SFT(CE) role-prediction loss.
#
# Example:
#   CUDA_DEVICES=0,1 USE_WANDB=false MODELS="E2B E4B 31B" \
#   bash among-agents/training/run_gemma_kto_with_state_recon.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

resolve_gemma_model() {
    local value="$1"
    local normalized
    normalized="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    normalized="${normalized//_/-}"
    case "$normalized" in
        e2b|gemmae2b|gemma-e2b|gemma-4-e2b|gemma-4-e2b-it|google/gemma-4-e2b-it)
            echo "google/gemma-4-E2B-it" ;;
        *)
            echo "$value" ;;
    esac
}

slugify_model_ref() {
    local value="$1"
    value="${value%/}"
    value="${value##*/}"
    value="${value//\//-}"
    value="${value//:/-}"
    value="${value// /-}"
    value="$(echo "$value" | tr -cd '[:alnum:]._-' | sed -E 's/-+/-/g; s/^-|-$//g')"
    echo "${value:-model}"
}

normalize_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$REPO_ROOT/$path"
    fi
}

count_cuda_devices() {
    local devices="$1"
    devices="${devices//[[:space:]]/}"
    if [[ -z "$devices" ]]; then
        echo 0
        return
    fi

    local -a device_ids
    IFS=',' read -r -a device_ids <<< "$devices"
    echo "${#device_ids[@]}"
}

check_cuda_free_memory() {
    local devices="$1"
    local min_free_mb="$2"

    if [[ "$min_free_mb" == "0" ]]; then
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[gemma-kto] WARN: nvidia-smi not found; skipping GPU memory preflight."
        return 0
    fi

    local -A free_by_gpu=()
    local line gpu free
    while IFS=',' read -r gpu free; do
        gpu="${gpu//[[:space:]]/}"
        free="${free//[[:space:]]/}"
        [[ -n "$gpu" && -n "$free" ]] && free_by_gpu["$gpu"]="$free"
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

    local -a device_ids
    local too_low=0
    IFS=',' read -r -a device_ids <<< "$devices"
    for gpu in "${device_ids[@]}"; do
        gpu="${gpu//[[:space:]]/}"
        if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
            echo "[gemma-kto] WARN: skipping memory check for non-numeric CUDA device '$gpu'."
            continue
        fi
        free="${free_by_gpu[$gpu]:-}"
        if [[ -z "$free" ]]; then
            echo "[gemma-kto] WARN: could not read free memory for GPU $gpu."
            continue
        fi
        echo "[gemma-kto] GPU $gpu free memory: ${free} MiB"
        if (( free < min_free_mb )); then
            echo "[gemma-kto] ERROR: GPU $gpu has ${free} MiB free, below MIN_FREE_CUDA_MB=${min_free_mb}." >&2
            too_low=1
        fi
    done

    if (( too_low )); then
        echo "[gemma-kto] Selected GPU(s) are too occupied. Use freer CUDA_DEVICES, stop old jobs, or set MIN_FREE_CUDA_MB=0 to bypass this check." >&2
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits >&2 || true
        exit 1
    fi
}

if [[ -n "${MODELS:-}" ]]; then
    read -r -a MODEL_SCHEDULE <<< "$MODELS"
else
    MODEL_SCHEDULE=("E2B" "E4B" "31B")
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
CUDA_DEVICES="${CUDA_DEVICES//[[:space:]]/}"
if [[ -z "$CUDA_DEVICES" ]]; then
    echo "[gemma-kto] ERROR: CUDA_DEVICES must not be empty." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

if [[ -z "${NUM_GPUS:-}" ]]; then
    NUM_GPUS="$(count_cuda_devices "$CUDA_DEVICES")"
else
    NUM_GPUS="$NUM_GPUS"
fi
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || (( NUM_GPUS < 1 )); then
    echo "[gemma-kto] ERROR: NUM_GPUS must be a positive integer, got '$NUM_GPUS'." >&2
    exit 1
fi
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-training/configs/accelerate_zero3_bf16_no_offload.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-models/gemma-kto-sft-ce}"
KTO_DATASET_PATH="${KTO_DATASET_PATH:-rewards_with_belief/data/gemma/kto_dataset}"
AUX_SFT_DATASET_PATH="${AUX_SFT_DATASET_PATH:-rewards_with_belief/data/gemma/role_prediction_sft_dataset}"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LR="${LR:-5e-7}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3584}"
BETA="${BETA:-0.1}"
TARGET_TRUE_RATIO="${TARGET_TRUE_RATIO:-0.6}"
GROUP_CYCLE_LENGTH="${GROUP_CYCLE_LENGTH:-16}"
AUX_LOSS_WEIGHT="${AUX_LOSS_WEIGHT:-auto}"
DESIRABLE_WEIGHT="${DESIRABLE_WEIGHT:-auto}"
UNDESIRABLE_WEIGHT="${UNDESIRABLE_WEIGHT:-1.0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
REPORT_TO="${REPORT_TO:-${USE_WANDB:-false}}"
MIN_FREE_CUDA_MB="${MIN_FREE_CUDA_MB:-40000}"

case "${REPORT_TO,,}" in
    true|1|yes|on) REPORT_TO="wandb" ;;
    false|0|no|off) REPORT_TO="none" ;;
esac

ACCELERATE_CONFIG="$(normalize_path "$ACCELERATE_CONFIG")"
OUTPUT_ROOT="$(normalize_path "$OUTPUT_ROOT")"
KTO_DATASET_PATH="$(normalize_path "$KTO_DATASET_PATH")"
AUX_SFT_DATASET_PATH="$(normalize_path "$AUX_SFT_DATASET_PATH")"
mkdir -p "$OUTPUT_ROOT"

echo "[gemma-kto] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[gemma-kto] NUM_GPUS=$NUM_GPUS"
check_cuda_free_memory "$CUDA_DEVICES" "$MIN_FREE_CUDA_MB"

for raw_model in "${MODEL_SCHEDULE[@]}"; do
    model_ref="$(resolve_gemma_model "$raw_model")"
    model_slug="$(slugify_model_ref "$model_ref")"
    output_dir="$OUTPUT_ROOT/$model_slug"

    echo "[gemma-kto] model=$model_ref"
    echo "[gemma-kto] output=$output_dir"

    "$PYTHON_BIN" -m accelerate.commands.launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_processes "$NUM_GPUS" \
        "$SCRIPT_DIR/kto_with_state_recon.py" \
        --model_path "$model_ref" \
        --dataset_path "$KTO_DATASET_PATH" \
        --aux_dataset_path "$AUX_SFT_DATASET_PATH" \
        --output_dir "$output_dir" \
        --max_length "$MAX_LENGTH" \
        --max_prompt_length "$MAX_PROMPT_LENGTH" \
        --beta "$BETA" \
        --desirable_weight "$DESIRABLE_WEIGHT" \
        --undesirable_weight "$UNDESIRABLE_WEIGHT" \
        --learning_rate "$LR" \
        --lr_scheduler_type cosine \
        --per_device_train_batch_size "$PER_DEVICE_BATCH" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --num_train_epochs "$NUM_EPOCHS" \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --save_steps 200 \
        --save_total_limit 1 \
        --gradient_checkpointing \
        --save_safetensors \
        --save_only_model \
        --target_true_ratio "$TARGET_TRUE_RATIO" \
        --group_cycle_length "$GROUP_CYCLE_LENGTH" \
        --aux_loss_weight "$AUX_LOSS_WEIGHT" \
        --torch_dtype "$TORCH_DTYPE" \
        --report_to "$REPORT_TO" \
        --run_name "$model_slug-MARBO"
done
