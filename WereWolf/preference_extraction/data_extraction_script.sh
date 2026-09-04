#!/bin/bash

# If invoked via `sh script.sh`, re-exec with bash for bash-specific syntax.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

# Keep both names populated so OpenAI-compatible clients pick up the key.
export OPENROUTER_API_KEY=""
export OPENAI_API_KEY=""
export OPENAI_API_BASE=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

game_dir=""

game_type=9p_seer_witch_guard
sft_model_regx=
out_dir="KTO_selected_data_${game_type}"
# DEVICE

# auto: use --random_play only when the chosen root has direct game_* folders.
# reward.py now also supports dataset roots that contain mixed sub-layouts.
random_play_mode="auto"
run_conflict_filter="false"
run_formatting="true"
run_relabel="false"
# OpenRouter model id used via the OpenAI-compatible client.
llm_verifier=""
speech_samples_path="${speech_samples_path:-$out_dir/adversarial_good_speech_full.json}"

random_play_args=()
if [[ "$random_play_mode" == "true" ]] || { [[ "$random_play_mode" == "auto" ]] && compgen -G "${game_dir}/game_*" > /dev/null; }; then
    random_play_args+=(--random_play)
    echo "Detected random-play layout under ${game_dir}; extracting only direct game_* directories."
else
    echo "Using matchup/self-play layout under ${game_dir}."
fi

mkdir -p $out_dir

echo "Running reward extraction for speech/vote/action/recon..."
python3 "${SCRIPT_DIR}/reward.py" --game_dir "$game_dir" --game_type "$game_type" \
    --sft_model_regx="$sft_model_regx" --out_to "$out_dir" \
    --speech_good_file "adversarial_good_speech.json" \
    --speech_bad_file "adversarial_bad_speech.json" \
    --vote_good_file "villager_good_vote.json" \
    --vote_bad_file "villager_bad_vote.json" \
    --action_good_file "good_actions.json" \
    --action_bad_file "bad_action.json" \
    --recon_good_file "adversarial_good_state_recon.json" \
    --recon_bad_file "adversarial_bad_state_recon.json" \
    --state_pair_samples_file "state_pair_samples.json" \
    "${random_play_args[@]}"

mkdir -p "$out_dir/conflict_speech" "$out_dir/conflict_state_recon"

if [[ "$run_conflict_filter" == "true" ]]; then
    if [[ "$llm_verifier" == *claude* ]] && ! python3 -c "import anthropic" >/dev/null 2>&1; then
        echo "[WARN] anthropic package is not installed; fallback verifier -> openai_gpt4o"
        llm_verifier="openai_gpt4o"
    fi

    if [[ -e "$speech_samples_path" ]]; then
        echo "Filtering conflict from good speech samples..."
        : > "$out_dir/conflict_speech/reverify_speeches.jsonl"
        python3 "${SCRIPT_DIR}/filter_conflict_from_good_speech.py" --game_type "$game_type" \
            --speech_samples "$speech_samples_path" \
            --out_to "$out_dir/conflict_speech/reverify_speeches.jsonl" \
            --llm_verifier "$llm_verifier"
    else
        echo "[WARN] Skip speech conflict filtering: $speech_samples_path not found."
    fi

    if [[ -f "$out_dir/adversarial_good_state_recon.json" ]]; then
        echo "Building desirable state reconstruction full samples..."
        python3 - "$out_dir/adversarial_good_state_recon.json" "$out_dir/adversarial_good_state_recon_full.json" <<'PY'
import json
import os
import sys


good_map_path = sys.argv[1]
out_path = sys.argv[2]


def extract_user_prompt(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if content is not None:
                    return str(content)
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    if isinstance(value, dict):
        if "content" in value:
            return str(value.get("content", ""))
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def extract_assistant_response(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("role") == "assistant":
                content = item.get("content")
                if content is not None:
                    return str(content)
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    if isinstance(value, dict):
        if "content" in value:
            return str(value.get("content", ""))
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


with open(good_map_path, "r", encoding="utf-8") as f:
    good_map = json.load(f)

out_rows = []
for game_path, by_player in good_map.items():
    abs_game_path = os.path.abspath(str(game_path))
    for player_id, phase_list in by_player.items():
        player_file = os.path.join(abs_game_path, f"Player_{player_id}.jsonl")
        if not os.path.exists(player_file):
            continue
        wanted = set(str(p) for p in phase_list)
        if not wanted:
            continue

        phase_to_entry = {}
        with open(player_file, "r", encoding="utf-8") as pf:
            for line in pf:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                phase = str(entry.get("phase", ""))
                if phase in wanted and phase not in phase_to_entry:
                    phase_to_entry[phase] = entry

        for phase in sorted(wanted):
            entry = phase_to_entry.get(phase)
            if not entry:
                continue
            prompt_text = extract_user_prompt(entry.get("prompt"))
            response_text = extract_assistant_response(entry.get("response"))
            if not prompt_text or not response_text:
                continue

            out_rows.append({
                "game_path": abs_game_path,
                "player_id": int(player_id) if str(player_id).isdigit() else player_id,
                "phase": phase,
                "prompt": prompt_text,
                "completion": response_text,
            })

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_rows, f, indent=4, ensure_ascii=False)

print(f"[state_recon_full] saved={len(out_rows)} -> {out_path}")
PY
    else
        echo "[WARN] Skip state_recon conflict preparation: $out_dir/adversarial_good_state_recon.json not found."
    fi
else
    # Prevent stale conflict jsonl files from previous runs from being merged.
    find "$out_dir/conflict_speech" -maxdepth 1 -type f -name "*.jsonl" -delete 2>/dev/null || true
    find "$out_dir/conflict_state_recon" -maxdepth 1 -type f -name "*.jsonl" -delete 2>/dev/null || true
fi


if [[ "$run_formatting" == "true" ]]; then
    echo "Formatting data"
    python3 "${SCRIPT_DIR}/format_training_data_2.py" --selected_path "$out_dir" \
        --conflict_speech_path "$out_dir/conflict_speech" \
        --conflict_state_recon_path "$out_dir/conflict_state_recon" \
        --save_prefix "$out_dir/kto_dataset" \
        --game_type "$game_type" \
        --mode_selection "strict"
fi

# Split dataset by phase (night_skill / speech / vote / state_reconstruction)
# Prefer train_samples produced by format_training_data_2.py.
split_input=""
for cand in \
    "$out_dir/kto_dataset_train_samples.json" \
    "$out_dir/kto_dataset_samples.json" \
    "$out_dir/kto_dataset_train.json"
do
    if [[ -f "$cand" ]]; then
        split_input="$cand"
        break
    fi
done

if [[ -n "$split_input" ]]; then
    split_prefix="$(basename "${split_input%.json}")"
    python3 "${SCRIPT_DIR}/divide.py" --input "$split_input" \
        --output_dir "$out_dir/train_samples" \
        --prefix "$split_prefix"
else
    echo "[WARN] Skip divide.py: no dataset file found under $out_dir"
fi

echo "[DONE] Saved reward splits under: $out_dir"
ls -1 "$out_dir" | grep -E "adversarial_(good|bad)_(speech|state_recon)\.json|villager_(good|bad)_vote\.json|good_actions\.json|bad_action\.json|kto_dataset_samples\.json|kto_dataset_pairs\.json" || true
