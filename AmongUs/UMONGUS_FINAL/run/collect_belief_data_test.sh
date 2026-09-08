#!/usr/bin/env bash
# Data-collection launcher for the submission setup.
#
# Belief-action models use BeliefStateLLMAgent:
#   action ~ pi(. | trajectory, belief)
# Opponent/measurement models use BeliefPredictOnlyLLMAgent:
#   belief ~ pi(. | trajectory), action ~ pi(. | trajectory)
#
# Local/open-source models are served through vLLM and routed per model with
# LLM_ROUTES_JSON. Hosted models such as openai/gpt-4o-mini keep using
# OpenRouter.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NAME="${NAME:-gemma_belief_shift}"
NUM_GAMES="${NUM_GAMES:-1}"
RATE_LIMIT="${RATE_LIMIT:-4}"

MODEL_POOL="${MODEL_POOL:-google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-31B-it Qwen/Qwen3.6-27B openai/gpt-4o-mini}"
BELIEF_ACTION_MODELS="${BELIEF_ACTION_MODELS:-google/gemma-4-E2B-it google/gemma-4-E4B-it}"
REQUIRED_MODELS="${REQUIRED_MODELS:-$BELIEF_ACTION_MODELS}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-1 1 0 0 0}"

# Current default only samples the two Gemma belief-action models. If you make
# another open-source model active by giving it positive weight or putting it in
# REQUIRED_MODELS, add/adjust its VLLM_MODEL_SPECS entry below.

OPENROUTER_API_URL="${OPENROUTER_API_URL:-https://openrouter.ai/api/v1/chat/completions}"
OPENROUTER_MODEL_SPECS="${OPENROUTER_MODEL_SPECS:-openai/gpt-4o-mini | openrouter}"

LOG_DIR="${LOG_DIR:-$ROOT/vllm-logs}"
MAX_LEN="${MAX_LEN:-16384}"
GMEM="${GMEM:-0.85}"
DTYPE="${DTYPE:-bfloat16}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
VLLM_PORT_BASE="${VLLM_PORT_BASE:-55000}"
VLLM_SERVER_PORT_STRIDE="${VLLM_SERVER_PORT_STRIDE:-1000}"
VLLM_RANK_PORT_STRIDE="${VLLM_RANK_PORT_STRIDE:-100}"
REUSE_VLLM="${REUSE_VLLM:-1}"
REPLACE_MISMATCHED_VLLM="${REPLACE_MISMATCHED_VLLM:-0}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"

# Fields: label | model_id | gpus | http_port | tensor_parallel | data_parallel
# Only active models are launched, so specs for inactive models may share GPUs.

# VLLM_MODEL_SPECS="${VLLM_MODEL_SPECS:-\
# gemmaE2B | google/gemma-4-E2B-it | 0 | 8000 | 1 | 1
# gemmaE4B | google/gemma-4-E4B-it | 1 | 8001 | 1 | 1
# gemma31 | google/gemma-4-31B-it | 2,3,4,5 | 8002 | 1 | 4
# qwen36_27 | Qwen/Qwen3.6-27B | 4,5,6,7 | 8003 | 1 | 4
# }"


VLLM_MODEL_SPECS="${VLLM_MODEL_SPECS:-\
gemmaE2B | google/gemma-4-E2B-it | 4,5 | 8000 | 1 | 2
gemmaE4B | google/gemma-4-E4B-it | 6,7 | 8001 | 1 | 2
}"



PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ ! -x "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" ]]; then
    echo "[collect_belief_data] ERROR: PYTHON_BIN='$PYTHON_BIN' not found." >&2
    exit 1
fi

JSON_PYTHON_BIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [[ -z "$JSON_PYTHON_BIN" ]]; then
    echo "[collect_belief_data] ERROR: python3/python is required for JSON escaping." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# Keep Triton/Torch/vLLM temporary files under the repo, mirroring the local
# run scripts in ../umongus/run. This avoids noexec /tmp failures on cluster
# nodes.
CACHE_ROOT="${CACHE_ROOT:-$ROOT/.cache}"
export TMPDIR="${TMPDIR:-$CACHE_ROOT/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export TORCH_COMPILE_CACHE_DIR="${TORCH_COMPILE_CACHE_DIR:-$CACHE_ROOT/torch_compile}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
         "$TORCH_COMPILE_CACHE_DIR" "$VLLM_CACHE_ROOT" "$HF_HOME" "$XDG_CACHE_HOME"

declare -a MODEL_POOL_ARR=()
declare -a MODEL_WEIGHTS_ARR=()
declare -a BELIEF_ACTION_MODELS_ARR=()
declare -a REQUIRED_MODELS_ARR=()

# shellcheck disable=SC2206
MODEL_POOL_ARR=(${MODEL_POOL})
if [[ -n "${MODEL_WEIGHTS:-}" ]]; then
    # shellcheck disable=SC2206
    MODEL_WEIGHTS_ARR=(${MODEL_WEIGHTS})
fi
# shellcheck disable=SC2206
BELIEF_ACTION_MODELS_ARR=(${BELIEF_ACTION_MODELS})
# shellcheck disable=SC2206
REQUIRED_MODELS_ARR=(${REQUIRED_MODELS})

if [[ ${#MODEL_WEIGHTS_ARR[@]} -gt 0 && ${#MODEL_WEIGHTS_ARR[@]} -ne ${#MODEL_POOL_ARR[@]} ]]; then
    echo "[collect_belief_data] ERROR: MODEL_WEIGHTS length (${#MODEL_WEIGHTS_ARR[@]}) must match MODEL_POOL length (${#MODEL_POOL_ARR[@]})." >&2
    exit 1
fi

contains() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

is_positive_weight() {
    "$JSON_PYTHON_BIN" - "$1" <<'PY'
import sys
try:
    sys.exit(0 if float(sys.argv[1]) > 0 else 1)
except Exception:
    sys.exit(1)
PY
}

declare -a ACTIVE_MODELS=()
for i in "${!MODEL_POOL_ARR[@]}"; do
    model="${MODEL_POOL_ARR[$i]}"
    active=0
    if contains "$model" "${REQUIRED_MODELS_ARR[@]}"; then
        active=1
    elif [[ ${#MODEL_WEIGHTS_ARR[@]} -eq 0 ]]; then
        active=1
    elif is_positive_weight "${MODEL_WEIGHTS_ARR[$i]}"; then
        active=1
    fi

    if [[ "$active" == "1" ]] && ! contains "$model" "${ACTIVE_MODELS[@]}"; then
        ACTIVE_MODELS+=("$model")
    fi
done

declare -A VLLM_LABEL_BY_MODEL=()
declare -A VLLM_GPUS_BY_MODEL=()
declare -A VLLM_PORT_BY_MODEL=()
declare -A VLLM_TP_BY_MODEL=()
declare -A VLLM_DP_BY_MODEL=()

while IFS= read -r raw; do
    raw="$(trim "$raw")"
    [[ -z "$raw" || "$raw" == \#* ]] && continue

    IFS='|' read -ra fields <<< "$raw"
    if [[ ${#fields[@]} -ne 6 ]]; then
        echo "[collect_belief_data] ERROR: bad VLLM_MODEL_SPECS entry: '$raw'" >&2
        echo "  expected: label | model_id | gpus | port | tp | dp" >&2
        exit 1
    fi

    label="$(trim "${fields[0]}")"
    model="$(trim "${fields[1]}")"
    gpus="$(trim "${fields[2]}")"
    port="$(trim "${fields[3]}")"
    tp="$(trim "${fields[4]}")"
    dp="$(trim "${fields[5]}")"

    VLLM_LABEL_BY_MODEL["$model"]="$label"
    VLLM_GPUS_BY_MODEL["$model"]="$gpus"
    VLLM_PORT_BY_MODEL["$model"]="$port"
    VLLM_TP_BY_MODEL["$model"]="$tp"
    VLLM_DP_BY_MODEL["$model"]="$dp"
done <<< "$VLLM_MODEL_SPECS"

declare -A HOSTED_PROVIDER_BY_MODEL=()
while IFS= read -r raw; do
    raw="$(trim "$raw")"
    [[ -z "$raw" || "$raw" == \#* ]] && continue

    IFS='|' read -ra fields <<< "$raw"
    if [[ ${#fields[@]} -ne 2 ]]; then
        echo "[collect_belief_data] ERROR: bad OPENROUTER_MODEL_SPECS entry: '$raw'" >&2
        echo "  expected: model_id | openrouter" >&2
        exit 1
    fi

    model="$(trim "${fields[0]}")"
    provider="$(trim "${fields[1]}")"
    HOSTED_PROVIDER_BY_MODEL["$model"]="$provider"
done <<< "$OPENROUTER_MODEL_SPECS"

SERVER_PIDS=()
SERVER_NAMES=()
ROUTES_JSON_ENTRIES=()

cleanup() {
    if [[ "$KEEP_SERVERS" == "1" ]]; then
        echo "[collect_belief_data] KEEP_SERVERS=1; leaving ${#SERVER_PIDS[@]} vLLM server(s) running."
        for i in "${!SERVER_PIDS[@]}"; do
            echo "  ${SERVER_NAMES[$i]} pid=${SERVER_PIDS[$i]}"
        done
        return
    fi

    if [[ ${#SERVER_PIDS[@]} -eq 0 ]]; then
        return
    fi

    echo "[collect_belief_data] stopping vLLM server(s)..."
    local pid
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
            command -v pkill >/dev/null 2>&1 && pkill -TERM -P "$pid" 2>/dev/null || true
        fi
    done
    sleep 5
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
            command -v pkill >/dev/null 2>&1 && pkill -KILL -P "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

json_escape() {
    "$JSON_PYTHON_BIN" -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

add_route() {
    local model="$1"
    local url="$2"
    local key="$3"
    local m_esc u_esc k_esc
    m_esc="$(printf '%s' "$model" | json_escape)"
    u_esc="$(printf '%s' "$url" | json_escape)"
    k_esc="$(printf '%s' "$key" | json_escape)"
    ROUTES_JSON_ENTRIES+=("${m_esc}: {\"url\": ${u_esc}, \"key\": ${k_esc}}")
}

is_port_listening() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    elif command -v ss >/dev/null 2>&1; then
        ss -ltn "( sport = :$port )" 2>/dev/null | tail -n +2 | grep -q .
    else
        curl -fsS "http://localhost:${port}/v1/models" >/dev/null 2>&1
    fi
}

server_model_ids() {
    local port="$1"
    curl -fsS "http://localhost:${port}/v1/models" 2>/dev/null \
        | "$JSON_PYTHON_BIN" -c 'import json,sys; data=json.load(sys.stdin); print("\n".join(str(x.get("id","")) for x in data.get("data", [])))'
}

server_has_model() {
    local port="$1"
    local model="$2"
    server_model_ids "$port" 2>/dev/null | grep -Fxq "$model"
}

stop_server_on_port() {
    local port="$1"
    local label="$2"
    local wanted_model="$3"
    local pids=()

    echo "[collect_belief_data] port $port has a different server for '$label'; replacing it with $wanted_model"
    if command -v lsof >/dev/null 2>&1; then
        while IFS= read -r pid; do
            [[ -n "$pid" ]] && pids+=("$pid")
        done < <(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
    fi

    if [[ ${#pids[@]} -eq 0 ]]; then
        echo "[collect_belief_data] ERROR: port $port is occupied, but listener pid could not be found." >&2
        exit 1
    fi

    local pid
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
        command -v pkill >/dev/null 2>&1 && pkill -TERM -P "$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
            command -v pkill >/dev/null 2>&1 && pkill -KILL -P "$pid" 2>/dev/null || true
        fi
    done
}

assert_port_free_or_reusable() {
    local port="$1"
    local label="$2"
    local model="$3"

    if ! is_port_listening "$port"; then
        return 0
    fi

    if [[ "$REUSE_VLLM" == "1" ]] && server_has_model "$port" "$model"; then
        echo "[collect_belief_data] reuse existing '$label' model=$model on port=$port"
        return 0
    fi

    if [[ "$REUSE_VLLM" == "1" && "$REPLACE_MISMATCHED_VLLM" == "1" ]]; then
        stop_server_on_port "$port" "$label" "$model"
        return 0
    fi

    echo "[collect_belief_data] ERROR: port $port ($label) is already in use." >&2
    if [[ "$REUSE_VLLM" == "1" ]]; then
        echo "[collect_belief_data] Existing /v1/models did not match '$model':" >&2
        server_model_ids "$port" >&2 || true
    fi
    command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$port" -sTCP:LISTEN >&2 || true
    exit 1
}

wait_for_server() {
    local port="$1"
    local label="$2"
    local pid="$3"
    local url="http://localhost:${port}/v1/models"
    local deadline=$((SECONDS + STARTUP_TIMEOUT))

    echo "[collect_belief_data] waiting for '$label' (port=$port, timeout=${STARTUP_TIMEOUT}s)..."
    while (( SECONDS < deadline )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[collect_belief_data] ERROR: '$label' exited during startup. See $LOG_DIR/${label}.log" >&2
            exit 1
        fi
        if curl -fsS "$url" >/dev/null 2>&1; then
            echo "[collect_belief_data] '$label' ready"
            return 0
        fi
        sleep 3
    done

    echo "[collect_belief_data] ERROR: '$label' was not ready within ${STARTUP_TIMEOUT}s. See $LOG_DIR/${label}.log" >&2
    exit 1
}

find_free_port() {
    local start="$1"
    local max_try=200
    local p
    for ((p=start; p<start+max_try; p++)); do
        if command -v ss >/dev/null 2>&1; then
            ss -ltn "( sport = :$p )" 2>/dev/null | tail -n +2 | grep -q . && continue
        elif command -v lsof >/dev/null 2>&1; then
            lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 && continue
        fi
        echo "$p"
        return 0
    done
    echo "[collect_belief_data] ERROR: no free port found from $start" >&2
    return 1
}

start_server() {
    local label="$1"
    local model="$2"
    local gpus="$3"
    local port="$4"
    local tp="$5"
    local dp="$6"
    local logfile="$LOG_DIR/${label}.log"

    if [[ "$REUSE_VLLM" == "1" ]] && server_has_model "$port" "$model"; then
        return 0
    fi

    local server_index="${#SERVER_PIDS[@]}"
    local vllm_port_base=$((VLLM_PORT_BASE + server_index * VLLM_SERVER_PORT_STRIDE))
    local vllm_port_max=$((vllm_port_base + dp * VLLM_RANK_PORT_STRIDE))
    if (( vllm_port_max >= 65535 )); then
        echo "[collect_belief_data] ERROR: '$label' internal vLLM port range exceeds 65535." >&2
        exit 1
    fi

    local rpc_port
    rpc_port="$(find_free_port $((port + 10000)))" || exit 1

    echo "[collect_belief_data] launch '$label' model=$model gpus=$gpus port=$port TP=$tp DP=$dp"
    echo "[collect_belief_data]   log -> $logfile"

    (
        export CUDA_VISIBLE_DEVICES="$gpus"
        export VLLM_HOST_IP="127.0.0.1"
        export VLLM_PORT="$vllm_port_base"
        export VLLM_PORT_STRIDE="$VLLM_RANK_PORT_STRIDE"
        unset MASTER_PORT MASTER_ADDR
        exec vllm serve "$model" \
            --port "$port" \
            --tensor-parallel-size "$tp" \
            --data-parallel-size "$dp" \
            --data-parallel-rpc-port "$rpc_port" \
            --api-server-count 1 \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization "$GMEM" \
            --dtype "$DTYPE"
    ) >"$logfile" 2>&1 &

    local pid=$!
    SERVER_PIDS+=("$pid")
    SERVER_NAMES+=("$label")
    wait_for_server "$port" "$label" "$pid"
}

declare -A SEEN_PORTS=()
declare -A SEEN_GPUS=()
declare -a LOCAL_ACTIVE_MODELS=()
declare -a HOSTED_ACTIVE_MODELS=()

for model in "${ACTIVE_MODELS[@]}"; do
    if [[ -n "${VLLM_LABEL_BY_MODEL[$model]:-}" ]]; then
        LOCAL_ACTIVE_MODELS+=("$model")
    elif [[ -n "${HOSTED_PROVIDER_BY_MODEL[$model]:-}" ]]; then
        HOSTED_ACTIVE_MODELS+=("$model")
    else
        echo "[collect_belief_data] ERROR: active open-source model has no vLLM spec: $model" >&2
        echo "  Add it to VLLM_MODEL_SPECS or set its MODEL_WEIGHTS entry to 0 and remove it from REQUIRED_MODELS." >&2
        exit 1
    fi
done

echo "[collect_belief_data] active models: ${ACTIVE_MODELS[*]}"
echo "[collect_belief_data] local vLLM models: ${LOCAL_ACTIVE_MODELS[*]:-<none>}"
echo "[collect_belief_data] hosted models: ${HOSTED_ACTIVE_MODELS[*]:-<none>}"
echo "[collect_belief_data] cache root: $CACHE_ROOT"

if [[ ${#LOCAL_ACTIVE_MODELS[@]} -gt 0 ]] && ! command -v vllm >/dev/null 2>&1; then
    echo "[collect_belief_data] ERROR: vllm command not found, but local models are active." >&2
    echo "  Activate the environment that has vLLM installed, or set PYTHON_BIN/PATH accordingly." >&2
    exit 1
fi

for model in "${HOSTED_ACTIVE_MODELS[@]}"; do
    provider="${HOSTED_PROVIDER_BY_MODEL[$model]}"
    case "$provider" in
        openrouter)
            if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
                echo "[collect_belief_data] ERROR: OPENROUTER_API_KEY is required for hosted model '$model'." >&2
                exit 1
            fi
            add_route "$model" "$OPENROUTER_API_URL" "$OPENROUTER_API_KEY"
            ;;
        *)
            echo "[collect_belief_data] ERROR: unsupported hosted provider '$provider' for '$model'." >&2
            exit 1
            ;;
    esac
done

for model in "${LOCAL_ACTIVE_MODELS[@]}"; do
    label="${VLLM_LABEL_BY_MODEL[$model]}"
    gpus="${VLLM_GPUS_BY_MODEL[$model]}"
    port="${VLLM_PORT_BY_MODEL[$model]}"
    tp="${VLLM_TP_BY_MODEL[$model]}"
    dp="${VLLM_DP_BY_MODEL[$model]}"

    if [[ -n "${SEEN_PORTS[$port]:-}" ]]; then
        echo "[collect_belief_data] ERROR: port $port is assigned to both '${SEEN_PORTS[$port]}' and '$label'." >&2
        exit 1
    fi
    SEEN_PORTS[$port]="$label"

    IFS=',' read -ra gids <<< "$gpus"
    for gid in "${gids[@]}"; do
        [[ -z "$gid" ]] && continue
        if [[ -n "${SEEN_GPUS[$gid]:-}" ]]; then
            echo "[collect_belief_data] ERROR: GPU $gid is assigned to both '${SEEN_GPUS[$gid]}' and '$label'." >&2
            exit 1
        fi
        SEEN_GPUS[$gid]="$label"
    done

    assert_port_free_or_reusable "$port" "$label" "$model"
    add_route "$model" "http://localhost:${port}/v1/chat/completions" "dummy"
done

for model in "${LOCAL_ACTIVE_MODELS[@]}"; do
    start_server \
        "${VLLM_LABEL_BY_MODEL[$model]}" \
        "$model" \
        "${VLLM_GPUS_BY_MODEL[$model]}" \
        "${VLLM_PORT_BY_MODEL[$model]}" \
        "${VLLM_TP_BY_MODEL[$model]}" \
        "${VLLM_DP_BY_MODEL[$model]}"
done

ROUTES_JSON="{$(IFS=,; echo "${ROUTES_JSON_ENTRIES[*]}")}"
FALLBACK_URL=""
FALLBACK_KEY="dummy"

if [[ ${#LOCAL_ACTIVE_MODELS[@]} -gt 0 ]]; then
    first_model="${LOCAL_ACTIVE_MODELS[0]}"
    FALLBACK_URL="http://localhost:${VLLM_PORT_BY_MODEL[$first_model]}/v1/chat/completions"
elif [[ ${#HOSTED_ACTIVE_MODELS[@]} -gt 0 ]]; then
    FALLBACK_URL="$OPENROUTER_API_URL"
    FALLBACK_KEY="$OPENROUTER_API_KEY"
fi

cat > .env <<EOF
LLM_API_URL=$FALLBACK_URL
LLM_API_KEY=$FALLBACK_KEY
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-dummy}
LLM_ROUTES_JSON=$ROUTES_JSON
LLM_TEMPERATURE=${LLM_TEMPERATURE:-0.7}
LLM_TOP_P=${LLM_TOP_P:-1.0}
LLM_MAX_RETRIES=${LLM_MAX_RETRIES:-10}
EOF

export LLM_API_URL="$FALLBACK_URL"
export LLM_API_KEY="$FALLBACK_KEY"
export LLM_ROUTES_JSON="$ROUTES_JSON"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.7}"
export LLM_TOP_P="${LLM_TOP_P:-1.0}"
export LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-10}"

echo "[collect_belief_data] routes: ${#ROUTES_JSON_ENTRIES[@]} model(s)"
echo "[collect_belief_data] fallback URL: ${FALLBACK_URL:-<empty>}"

MODEL_WEIGHT_ARGS=()
if [[ -n "${MODEL_WEIGHTS:-}" ]]; then
    MODEL_WEIGHT_ARGS=(--model_weights "${MODEL_WEIGHTS_ARR[@]}")
fi

"$PYTHON_BIN" collect_belief_data.py \
    --name "$NAME" \
    --num_games "$NUM_GAMES" \
    --rate_limit "$RATE_LIMIT" \
    --model_pool "${MODEL_POOL_ARR[@]}" \
    --belief_action_models "${BELIEF_ACTION_MODELS_ARR[@]}" \
    --required_models "${REQUIRED_MODELS_ARR[@]}" \
    "${MODEL_WEIGHT_ARGS[@]}"
