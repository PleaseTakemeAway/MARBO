#!/bin/bash
# KTO Training with accelerate launch + DeepSpeed ZeRO-3
# Usage: bash run_auxiliary_kto_accelerate.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"


NUM_GPUS=4

export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1


MODEL_PATH=""
REF_MODEL_PATH=""
DATASET_PATH=""
OUTPUT_DIR=""
ACCELERATE_CONFIG=""   
AUX_DATASET_PATH=""

PER_DEVICE_BATCH=2
GRAD_ACCUM=16
NUM_EPOCHS=10
LR=1e-6
MAX_LENGTH=16384
BETA=0.1
DESIRABLE_WEIGHT=0.7
UNDESIRABLE_WEIGHT=1.5
TARGET_TRUE_RATIO=0.5
PHASE_CYCLE_LENGTH=16
TORCH_DTYPE="bfloat16"
DATASET_SPLIT="train"
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.1
LOGGING_STEPS=10
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=1
REPORT_TO="${REPORT_TO:-none}"
OPTIM="adamw_torch"
RESUME_FROM_CHECKPOINT=""
DEBUG_BATCH_RATIO_STEPS=20
AUX_LOSS_WEIGHT=0.2

# ── bool argparse flags (0/1) ─────────────────────────────────────────────────
USE_GRADIENT_CHECKPOINTING=1
USE_SAVE_SAFETENSORS=1
USE_SAVE_ONLY_MODEL=1
USE_TRUST_REMOTE_CODE=0
USE_DEBUG_BATCH_RATIO=1

# ──────────────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

TRAIN_ARGS=(
    --model_path "$MODEL_PATH"
    --dataset_path "$DATASET_PATH"
    --dataset_split "$DATASET_SPLIT"
    --output_dir "$OUTPUT_DIR"
    --max_length "$MAX_LENGTH"
    --beta "$BETA"
    --desirable_weight "$DESIRABLE_WEIGHT"
    --undesirable_weight "$UNDESIRABLE_WEIGHT"
    --learning_rate "$LR"
    --lr_scheduler_type "$LR_SCHEDULER_TYPE"
    --per_device_train_batch_size "$PER_DEVICE_BATCH"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --num_train_epochs "$NUM_EPOCHS"
    --warmup_ratio "$WARMUP_RATIO"
    --logging_steps "$LOGGING_STEPS"
    --save_steps "$SAVE_STEPS"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
    --report_to "$REPORT_TO"
    --optim "$OPTIM"
    --target_true_ratio "$TARGET_TRUE_RATIO"
    --phase_cycle_length "$PHASE_CYCLE_LENGTH"
    --debug_batch_ratio_steps "$DEBUG_BATCH_RATIO_STEPS"
    --aux_loss_weight "$AUX_LOSS_WEIGHT"
    --torch_dtype "$TORCH_DTYPE"
)

# optional string args
if [[ -n "$REF_MODEL_PATH" ]]; then
    TRAIN_ARGS+=(--ref_model_path "$REF_MODEL_PATH")
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
if [[ -n "$AUX_DATASET_PATH" ]]; then
    TRAIN_ARGS+=(--aux_dataset_path "$AUX_DATASET_PATH")
fi

# bool flags
if [[ "$USE_GRADIENT_CHECKPOINTING" == "1" ]]; then
    TRAIN_ARGS+=(--gradient_checkpointing)
fi
if [[ "$USE_SAVE_SAFETENSORS" == "1" ]]; then
    TRAIN_ARGS+=(--save_safetensors)
fi
if [[ "$USE_SAVE_ONLY_MODEL" == "1" ]]; then
    TRAIN_ARGS+=(--save_only_model)
fi
if [[ "$USE_TRUST_REMOTE_CODE" == "1" ]]; then
    TRAIN_ARGS+=(--trust_remote_code)
fi
if [[ "$USE_DEBUG_BATCH_RATIO" == "1" ]]; then
    TRAIN_ARGS+=(--debug_batch_ratio)
fi

accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --num_processes "$NUM_GPUS" \
    "$SCRIPT_DIR/kto_batch.py" \
    "${TRAIN_ARGS[@]}"
