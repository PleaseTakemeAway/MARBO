import argparse
import json
import os
import re
from typing import Any

from MARBO.wolf.preference_extraction.utils_english import create_message, get_role_assignment, get_system_prompt, to_text


def collect_game_logs(root_dir: str) -> list[str]:
    """Collect game_log.json files under root_dir (direct game_* or recursive)."""
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        return []

    children = os.listdir(root_dir)
    if any(re.fullmatch(r"game_\d+", c) for c in children):
        matched = []
        for c in children:
            p = os.path.join(root_dir, c, "game_log.json")
            if os.path.isfile(p):
                matched.append(p)
        return sorted(matched)

    matched = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "game_log.json" in filenames:
            matched.append(os.path.join(dirpath, "game_log.json"))
    return sorted(matched)


def _normalize_ground_truth(raw: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        m = re.search(r"\d+", str(k))
        if not m:
            continue
        out[m.group()] = str(v)
    return out


def _infer_n_player(log: list[dict[str, Any]], ground_truth: dict[str, str]) -> int:
    if ground_truth:
        return max(int(k) for k in ground_truth.keys())
    for e in log:
        content = e.get("content")
        if isinstance(content, dict) and "n_player" in content:
            try:
                return int(content["n_player"])
            except Exception:
                pass
    return 9


def _as_player_key_map(role_map: dict[str, str], n_player: int) -> dict[str, str]:
    completion = {f"Player {i}": "Unknown" for i in range(1, n_player + 1)}
    for pid, role in role_map.items():
        completion[f"Player {pid}"] = role
    return completion


def _extract_identity_labels_prompt(event: dict[str, Any]) -> Any:
    content = event.get("content", {})
    if isinstance(content, dict):
        return content.get("identity_labels", "{}")
    return "{}"


def _strip_code_fence(text: str) -> str:
    t = str(text).strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_json_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = _strip_code_fence(value)
    try:
        return json.loads(text)
    except Exception:
        return None


def _canonicalize_identity_labels(value: Any) -> str:
    parsed = _parse_json_maybe(value)
    if isinstance(parsed, dict):
        normalized = {str(k): parsed[k] for k in sorted(parsed.keys(), key=lambda x: str(x))}
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    if isinstance(parsed, list):
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    return _strip_code_fence(str(value))


def _is_missing_prompt(prompt: Any) -> bool:
    if prompt is None:
        return True
    if isinstance(prompt, str):
        t = prompt.strip()
        return t in {"", "{}", "null", "None"}
    if isinstance(prompt, list):
        if len(prompt) == 0:
            return True
        for item in prompt:
            if isinstance(item, dict):
                if str(item.get("content", "")).strip():
                    return False
            elif str(item).strip():
                return False
        return True
    if isinstance(prompt, dict):
        if "content" in prompt:
            return not str(prompt.get("content", "")).strip()
        return len(prompt) == 0
    return False


def _to_prompt_messages(prompt_raw: Any, system_prompt: str) -> list[dict[str, str]]:
    if isinstance(prompt_raw, list):
        msgs: list[dict[str, str]] = []
        for item in prompt_raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user"))
            content = item.get("content", "")
            content_text = content if isinstance(content, str) else to_text(content)
            if not content_text.strip():
                continue
            msgs.append(create_message(role, content_text))
        if not msgs:
            return []
        has_system = any(m.get("role") == "system" for m in msgs)
        if has_system:
            return msgs
        return [create_message("system", system_prompt)] + msgs

    if isinstance(prompt_raw, dict):
        role = str(prompt_raw.get("role", "user"))
        content_text = to_text(prompt_raw.get("content", ""))
        if not content_text.strip():
            return []
        msg = create_message(role, content_text)
        if role == "system":
            return [msg]
        return [create_message("system", system_prompt), msg]

    prompt_text = to_text(prompt_raw).strip()
    if not prompt_text:
        return []
    return [
        create_message("system", system_prompt),
        create_message("user", prompt_text),
    ]


def _completion_to_json_dict_maybe(completion_raw: Any) -> dict[str, Any] | None:
    """
    Parse completion into a JSON dict.
    Return None when:
      - not valid JSON text
      - valid JSON but not object/dict (e.g., list/number/string)
    """
    if isinstance(completion_raw, dict):
        return completion_raw

    if isinstance(completion_raw, str):
        parsed = _parse_json_maybe(completion_raw)
        return parsed if isinstance(parsed, dict) else None

    text = to_text(completion_raw).strip()
    parsed = _parse_json_maybe(text)
    return parsed if isinstance(parsed, dict) else None


def _is_valid_completion_json_dict(completion_raw: Any) -> bool:
    return isinstance(_completion_to_json_dict_maybe(completion_raw), dict)


def _to_completion_messages(completion_raw: Any) -> list[dict[str, str]]:
    if not _is_valid_completion_json_dict(completion_raw):
        return []

    # Keep original completion logic/content as-is; only align data type/shape.
    text = completion_raw if isinstance(completion_raw, str) else to_text(completion_raw)
    text = text.strip()
    if not text:
        return []
    return [create_message("assistant", text)]


def _build_kto_sample(
    prompt_raw: Any,
    completion_raw: Any,
    role_label: dict[str, str],
    phase: str,
    game_path: str,
    system_prompt: str,
) -> dict[str, Any] | None:
    prompt_msgs = _to_prompt_messages(prompt_raw, system_prompt)
    if _is_missing_prompt(prompt_msgs):
        return None
    completion_msgs = _to_completion_messages(completion_raw)
    if not completion_msgs:
        return None
    if not str(completion_msgs[0].get("content", "")).strip():
        return None

    return {
        "prompt": prompt_msgs,
        "completion": completion_msgs,
        "label": True,
        "role_label": role_label,
        "phase": phase,
        "game_path": game_path,
        "state_recon": None,
    }


def _resolve_output_path(output_path: str, include_post_state_phase: bool) -> str:
    abs_path = os.path.abspath(output_path)
    is_dir_like = (
        abs_path.endswith(os.sep)
        or output_path.endswith(os.sep)
        or os.path.isdir(abs_path)
    )

    if is_dir_like:
        default_name = "state_reconstruction_samples.json"
        if include_post_state_phase:
            default_name = "state_reconstruction_samples_with_post.json"
        return os.path.join(abs_path, default_name)

    if not include_post_state_phase:
        return abs_path

    dir_name, file_name = os.path.split(abs_path)
    stem, ext = os.path.splitext(file_name)
    suffix = "_with_post"
    if stem.endswith(suffix):
        new_file_name = file_name
    else:
        new_file_name = f"{stem}{suffix}{ext}"
    return os.path.join(dir_name, new_file_name)


def _load_player_state_recon_entries(
    game_path: str,
    player_id: str,
    cache: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    key = (os.path.abspath(game_path), str(player_id))
    if key in cache:
        return cache[key]

    player_file = os.path.join(key[0], f"Player_{key[1]}.jsonl")
    rows: list[dict[str, Any]] = []
    if os.path.isfile(player_file):
        with open(player_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                phase = str(obj.get("phase", ""))
                if "state_reconstruction" not in phase:
                    continue
                rows.append(obj)

    cache[key] = rows
    return rows


def _resolve_prompt_from_state_sample(
    event: dict[str, Any],
    game_path: str,
    player_cache: dict[tuple[str, str], list[dict[str, Any]]],
    include_post_state_phase: bool = False,
) -> tuple[Any, str | None]:
    """
    Find original state_reconstruction sample prompt from Player_{id}.jsonl.
    Match priority:
      1) identity_labels(response) exact canonical match
      2) phase/day heuristic fallback
      3) fallback to event identity_labels text
    Returns: (prompt, matched_phase)
    """
    source = event.get("source")
    source_id_match = re.search(r"\d+", str(source))
    if source_id_match is None:
        return _extract_identity_labels_prompt(event), None
    source_id = source_id_match.group()

    identity_labels = _extract_identity_labels_prompt(event)
    target_canon = _canonicalize_identity_labels(identity_labels)
    entries = _load_player_state_recon_entries(game_path, source_id, player_cache)

    # 1) response exact match
    for obj in entries:
        resp_canon = _canonicalize_identity_labels(obj.get("response", ""))
        if resp_canon == target_canon:
            prompt = obj.get("prompt", identity_labels)
            return prompt, str(obj.get("phase", "")) or None

    # 2) fallback by event day/time to avoid empty prompt
    day = event.get("day")
    time_text = str(event.get("time", "")).lower()
    try:
        day_i = int(day)
    except Exception:
        day_i = None
    if day_i is not None:
        for obj in entries:
            phase = str(obj.get("phase", ""))
            if not phase.startswith(f"{day_i}_"):
                continue
            if "night" in time_text and "_night_state_reconstruction" in phase:
                return obj.get("prompt", identity_labels), phase
            if "night" not in time_text and _is_day_target_state_phase(
                phase, include_post_state_phase=include_post_state_phase
            ):
                return obj.get("prompt", identity_labels), phase

    return identity_labels, None


def _event_phase(event: dict[str, Any]) -> str:
    day = event.get("day")
    try:
        day_i = int(day)
    except Exception:
        day_i = 0
    time_text = str(event.get("time", "")).lower()
    if "night" in time_text:
        return f"{day_i}_night_state_reconstruction"
    return f"{day_i}_day_state_reconstruction"


def _is_day_pre_state_phase(phase: str | None) -> bool:
    if not phase:
        return False
    return "_day_pre_state_reconstruction" in str(phase)


def _is_day_post_state_phase(phase: str | None) -> bool:
    if not phase:
        return False
    return "_day_post_state_reconstruction" in str(phase)


def _is_day_target_state_phase(phase: str | None, include_post_state_phase: bool = False) -> bool:
    if _is_day_pre_state_phase(phase):
        return True
    if include_post_state_phase and _is_day_post_state_phase(phase):
        return True
    return False


def day_1_state_reconstruction(
    event: dict[str, Any],
    ground_truth: dict[str, str],
    n_player: int,
    game_path: str,
    prompt_text: Any,
    system_prompt: str,
    phase_name: str | None = None,
) -> dict[str, Any] | None:
    """
    Build sample for 1_day_state_reconstruction:
      - knows own role
      - if own role is Werewolf, also knows wolf teammates
      - all others are Unknown
    """
    if event.get("event") != "state_reconstruction":
        return None
    if _event_phase(event) != "1_day_state_reconstruction":
        return None

    source = event.get("source")
    source_id = re.search(r"\d+", str(source))
    if source_id is None:
        return None
    viewer_id = source_id.group()

    completion = {f"Player {i}": "Unknown" for i in range(1, n_player + 1)}
    viewer_role = ground_truth.get(viewer_id, "Unknown")
    completion[f"Player {viewer_id}"] = viewer_role

    # teammate info: Werewolf only (villager-side players generally don't know teammates).
    if viewer_role == "Werewolf":
        for pid, role in ground_truth.items():
            if role == "Werewolf":
                completion[f"Player {pid}"] = "Werewolf"

    # Day-1 reconstruction should not expose full ground-truth role labels.
    # Keep role_label aligned with the day-1 teachable target (known + unknown).
    day1_role_label = {
        str(i): completion.get(f"Player {i}", "Unknown")
        for i in range(1, n_player + 1)
    }

    output_phase = phase_name or "1_day_state_reconstruction"
    return _build_kto_sample(
        prompt_raw=prompt_text,
        completion_raw=completion,
        role_label=day1_role_label,
        phase=output_phase,
        game_path=game_path,
        system_prompt=system_prompt,
    )


def day_state_reconstruction(
    event: dict[str, Any],
    ground_truth: dict[str, str],
    n_player: int,
    game_path: str,
    prompt_text: Any,
    system_prompt: str,
    phase_name: str | None = None,
) -> dict[str, Any] | None:
    """
    Build sample for all non-1_day state_reconstruction phases:
      - completion is full ground-truth role map.
    """
    if event.get("event") != "state_reconstruction":
        return None
    phase = _event_phase(event)
    if phase == "1_day_state_reconstruction":
        return None

    source = event.get("source")
    source_id = re.search(r"\d+", str(source))
    if source_id is None:
        return None

    output_phase = phase_name or phase
    return _build_kto_sample(
        prompt_raw=prompt_text,
        completion_raw=_as_player_key_map(ground_truth, n_player),
        role_label=ground_truth,
        phase=output_phase,
        game_path=game_path,
        system_prompt=system_prompt,
    )


def build_state_reconstruction_samples(
    log: list[dict[str, Any]],
    game_path: str,
    system_prompt: str,
    include_post_state_phase: bool = False,
) -> list[dict[str, Any]]:
    gt_raw = get_role_assignment(log)
    ground_truth = _normalize_ground_truth(gt_raw)
    if not ground_truth:
        return []

    n_player = _infer_n_player(log, ground_truth)
    samples: list[dict[str, Any]] = []
    player_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for event in log:
        if event.get("event") != "state_reconstruction":
            continue
        prompt_text, matched_phase = _resolve_prompt_from_state_sample(
            event,
            game_path,
            player_cache,
            include_post_state_phase=include_post_state_phase,
        )
        if _is_missing_prompt(prompt_text):
            continue

        day1_sample = day_1_state_reconstruction(
            event=event,
            ground_truth=ground_truth,
            n_player=n_player,
            game_path=game_path,
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            phase_name=matched_phase,
        )
        if day1_sample is not None:
            samples.append(day1_sample)
            continue

        other_sample = day_state_reconstruction(
            event=event,
            ground_truth=ground_truth,
            n_player=n_player,
            game_path=game_path,
            prompt_text=prompt_text,
            system_prompt=system_prompt,
            phase_name=matched_phase,
        )
        if other_sample is not None:
            samples.append(other_sample)

    return samples


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build state reconstruction reward samples.\n"
            "- default: collect *_day_pre_state_reconstruction phases\n"
            "- optional: include *_day_post_state_reconstruction via --include_post_state_phase\n"
            "- day 1: own role (+ wolf teammates only)\n"
            "- other days: full ground truth"
        )
    )

    # game_dir="experiments/260418/ours_v1_gpt/w-gpt_vs_v-sft"
    # game_type=9p_seer_witch_guard
    parser.add_argument("--game_dir", default="experiments/260418/ours_v1_gpt/w-gpt_vs_v-sft",type=str)
    parser.add_argument("--output_path",default='experiments_v1' ,type=str)
    parser.add_argument("--game_type", type=str, default="9p_seer_witch_guard")
    parser.add_argument(
        "--include_post_state_phase",
        action="store_true",
        help=(
            "Also include *_day_post_state_reconstruction phases (default: pre-only). "
            "When enabled, output filename gets '_with_post' suffix automatically."
        ),
    )
    return parser.parse_args()

def label_state_reconstruction():
    args = parse_args()
    system_prompt = get_system_prompt(args.game_type)
    output_path = _resolve_output_path(args.output_path, args.include_post_state_phase)
    game_logs = collect_game_logs(args.game_dir)
    if not game_logs:
        raise FileNotFoundError(f"No game_log.json found under: {args.game_dir}")

    all_samples: list[dict[str, Any]] = []
    for game_log_path in game_logs:
        with open(game_log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        game_path = os.path.dirname(game_log_path)
        all_samples.extend(
            build_state_reconstruction_samples(
                log,
                game_path,
                system_prompt,
                include_post_state_phase=args.include_post_state_phase,
            )
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"[done] samples={len(all_samples)} -> {output_path}")

def main():
    args = parse_args()
    system_prompt = get_system_prompt(args.game_type)
    output_path = _resolve_output_path(args.output_path, args.include_post_state_phase)
    game_logs = collect_game_logs(args.game_dir)
    if not game_logs:
        raise FileNotFoundError(f"No game_log.json found under: {args.game_dir}")

    all_samples: list[dict[str, Any]] = []
    for game_log_path in game_logs:
        with open(game_log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        game_path = os.path.dirname(game_log_path)
        all_samples.extend(
            build_state_reconstruction_samples(
                log,
                game_path,
                system_prompt,
                include_post_state_phase=args.include_post_state_phase,
            )
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"[done] samples={len(all_samples)} -> {output_path}")


if __name__ == "__main__":
    main()
