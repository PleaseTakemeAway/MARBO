"""
Build a KTO dataset from Among Us compact agent logs for belief-enabled agents.

The output schema matches TRL's KTOTrainer input:
    {
        "prompt": [{"role": "system", "content": ...},
                   {"role": "user", "content": ...}],
        "completion": [{"role": "assistant", "content": ...}],
        "label": bool,
        "phase": str,
    }
"""

import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset

from .base import RewardFunction
from . import sampling_config, utils
from .dataset_sampling import (
    add_model_ratios_to_weights,
    allocate_counts,
    append_model_ratio_key,
    flatten_kto_weights,
    infer_feasible_total,
    normalize_bucket_key,
    resolve_model_ratios,
    stratified_sample_records,
)
from .progress import iter_progress, progress_bar


_PHASE_RE = re.compile(r"Phase:\s*([A-Za-z]+\s*phase)", re.IGNORECASE)
_ACTION_BLOCK_RE = re.compile(r"(\[Action\]\s*)(.*)", re.IGNORECASE | re.DOTALL)
_SPEAK_ACTION_RE = re.compile(r"(?:^\s*\d+\.\s*)?SPEAK\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_TASK_ACTION_TYPES = ("CALL MEETING", "COMPLETE TASK", "KILL", "MOVE", "REPORT", "VENT", "VIEW MONITOR")
_TASK_ACTION_TYPE_FALLBACK = "other_task_action"


def _format_user_prompt(prompt_field: Any) -> str:
    if isinstance(prompt_field, str):
        return prompt_field

    if isinstance(prompt_field, dict):
        summarization = prompt_field.get("Summarization", "")
        all_info = prompt_field.get("All Info", "")
        memory = prompt_field.get("Memory", "")
        phase = prompt_field.get("Phase", "")
        return (
            f"Summarization: {summarization}\n\n"
            f"{all_info}\n\n"
            f"Memory: {memory}"
            f"                    \n\nPhase: {phase}. Return your output."
        )

    return str(prompt_field)


def _extract_phase(prompt_field: Any) -> str:
    if isinstance(prompt_field, dict):
        return str(prompt_field.get("Phase", "")).strip()
    if isinstance(prompt_field, str):
        match = _PHASE_RE.search(prompt_field)
        return match.group(1).strip() if match else ""
    return ""


def _replace_action_block(full_response: str, action_text: str) -> str:
    match = _ACTION_BLOCK_RE.search(full_response or "")
    if not match:
        return f"{str(full_response or '').rstrip()}\n[Action] {action_text}".strip()
    return f"{full_response[:match.start(2)]}{action_text}"


def _raw_action_from_full_response(full_response: str) -> str:
    match = _ACTION_BLOCK_RE.search(full_response or "")
    if not match:
        return ""
    return utils.first_line(match.group(2).strip())


def _canonical_speech_action(raw_action: str) -> str:
    match = _SPEAK_ACTION_RE.search(raw_action or "")
    if not match:
        return ""
    return f"SPEAK: {match.group(1).strip()}"


def _completion_for_applied_action(entry: Dict[str, Any]) -> str:
    """
    Keep the original logged reasoning, but make the saved [Action] line match
    the action that the environment actually accepted.

    Repair attempts use the same original prompt for training. Fallback actions
    are excluded by the reward dispatcher before rows are created.
    """
    interaction = entry.get("interaction", {})
    full_response = str(interaction.get("full_response", "") or "")
    applied_action = utils.extract_action(entry)
    if not applied_action:
        return full_response

    category = utils.action_category(applied_action)
    raw_action = _raw_action_from_full_response(full_response)

    if category == "SPEAK":
        speech_action = _canonical_speech_action(raw_action) or _canonical_speech_action(applied_action)
        if speech_action:
            return _replace_action_block(full_response, speech_action)
        return _replace_action_block(full_response, applied_action)

    return _replace_action_block(full_response, applied_action)


def _completion_for_repair_attempt(entry: Dict[str, Any], attempt: Dict[str, Any]) -> str:
    response = str(attempt.get("response", "") or "")
    invalid_action = utils.first_line(str(attempt.get("invalid_action", "") or ""))
    if not invalid_action:
        return response
    if _ACTION_BLOCK_RE.search(response):
        return response
    return _replace_action_block(response, invalid_action)


def _task_action_type(action: str) -> str:
    action_upper = str(action or "").strip().upper()
    if not action_upper:
        return "<missing_action_line>"
    if action_upper.startswith("COMPLETE "):
        return "COMPLETE TASK"
    if action_upper.startswith("REPORT"):
        return "REPORT"
    for action_type in _TASK_ACTION_TYPES:
        if action_upper.startswith(action_type):
            return action_type
    return "<missing_action_line>"


def _task_action_detail_type(action: str) -> str:
    category = utils.action_category(str(action or "").strip())
    if category == "COMPLETE TASK":
        return "complete_task"
    if category == "MOVE":
        return "movement"
    if category in {"COMPLETE FAKE TASK", "KILL", "VENT"}:
        return "impostor_action"
    if category in {"CALL MEETING", "REPORT DEAD BODY"}:
        return "meeting_trigger"
    if category == "VIEW MONITOR":
        return "monitor"
    return _TASK_ACTION_TYPE_FALLBACK


def _is_human_or_random_entry(entry: Dict[str, Any]) -> bool:
    model = utils.get_model(entry).lower()
    system_prompt = str(entry.get("interaction", {}).get("system_prompt", "")).lower()
    if "homosapiens" in model or "human agent" in system_prompt:
        return True
    return model in {"random", "randomagent", "random-agent"}


def _can_use_repair_negative_entry(entry: Dict[str, Any]) -> bool:
    return (
        utils.is_belief_enabled_entry(entry)
        and not _is_human_or_random_entry(entry)
        and utils.is_action_entry(entry)
    )


def _available_action_match(action_text: str, available_actions: List[str]) -> bool:
    candidate = str(action_text or "").strip()
    if not candidate:
        return False
    for available_action in available_actions:
        if available_action.startswith("SPEAK:"):
            continue
        if candidate == available_action or available_action in candidate:
            return True
    return False


def _repair_negative_rows_for_entry(
    entry: Dict[str, Any],
    *,
    system_prompt: str,
    user_prompt: str,
    phase: str,
) -> List[Dict[str, Any]]:
    if phase != "Task phase" or not _can_use_repair_negative_entry(entry):
        return []

    prompt = entry.get("interaction", {}).get("prompt", {})
    if not isinstance(prompt, dict):
        return []

    attempts = prompt.get("Action Repair Attempts")
    if not isinstance(attempts, list) or not attempts:
        return []

    available_actions = utils.available_actions(utils.get_all_info(entry))
    game_key = (entry.get("_log_path", ""), entry.get("game_index", ""))
    source_model = str(entry.get("player", {}).get("model", "")).strip()
    actor_role = str(entry.get("player", {}).get("identity", "")).strip()
    rows: List[Dict[str, Any]] = []

    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        invalid_action = utils.first_line(str(attempt.get("invalid_action", "") or ""))
        if not invalid_action:
            continue
        if _available_action_match(invalid_action, available_actions):
            continue

        completion = _completion_for_repair_attempt(entry, attempt)
        if not completion:
            continue
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "completion": [{"role": "assistant", "content": completion}],
                "label": False,
                "phase": phase,
                "source_model": source_model,
                "actor_role": actor_role,
                "kto_bucket": "task_action",
                "task_action_type": _task_action_detail_type(invalid_action),
                "sample_source": "repair_invalid_action",
                "action_text": invalid_action,
                "_kto_bucket": "task_action",
                "_action_type": _task_action_type(invalid_action),
                "_task_action_type": _task_action_detail_type(invalid_action),
                "_entry": entry,
                "_game_key": game_key,
            }
        )

    return rows


def _normalize_model_filters(include_models: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if not include_models:
        return ()
    model_filters = []
    for raw_model in include_models:
        model_filters.extend(model.strip().lower() for model in str(raw_model).split(",") if model.strip())
    return tuple(model_filters)


def _matches_model_filter(entry: Dict[str, Any], model_filters: Tuple[str, ...]) -> bool:
    if not model_filters:
        return True
    model = str(entry.get("player", {}).get("model", "")).strip().lower()
    return any(model_filter in model for model_filter in model_filters)


def resolve_log_paths(log_paths: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    for raw_path in log_paths:
        path = os.path.abspath(os.path.expanduser(str(raw_path)))
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if os.path.isdir(path):
            direct_file = os.path.join(path, "agent-logs-compact.json")
            if os.path.isfile(direct_file):
                resolved.append(direct_file)
                continue

            child_files = []
            for child_name in os.listdir(path):
                child_file = os.path.join(path, child_name, "agent-logs-compact.json")
                if os.path.isfile(child_file):
                    child_files.append(child_file)
            if child_files:
                resolved.extend(child_files)
                continue

            for root, _, files in os.walk(path):
                if "agent-logs-compact.json" in files:
                    resolved.append(os.path.join(root, "agent-logs-compact.json"))
            continue
        resolved.append(path)
    return sorted(dict.fromkeys(resolved))


def load_log_entries(log_paths: Iterable[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    resolved_paths = resolve_log_paths(log_paths)
    for path in iter_progress(resolved_paths, desc="[kto] loading log files", unit="file"):
        abs_path = os.path.abspath(path)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                entry["_log_path"] = abs_path
                entries.append(entry)
    return entries


def group_by_game(entries: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    games: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        games[(entry.get("_log_path", ""), entry["game_index"])].append(entry)
    return games


def build_kto_dataset(
    log_paths: Iterable[str],
    reward_fn: Optional[RewardFunction] = None,
    post_sampling_reward_fn: Optional[RewardFunction] = None,
    drop_empty_response: bool = True,
    include_models: Optional[Iterable[str]] = None,
    apply_sampling: bool = True,
    task_actions_only: bool = False,
    include_repair_negatives: bool = False,
) -> Dataset:
    if reward_fn is None:
        from .game_reward import GameOutcomeReward

        reward_fn = GameOutcomeReward()

    model_filters = _normalize_model_filters(include_models)
    entries = load_log_entries(log_paths)
    games = group_by_game(entries)
    rows: List[Dict[str, Any]] = []
    replace_task_actions = (
        not task_actions_only
        and apply_sampling
        and getattr(sampling_config, "KTO_REPLACE_TASK_ACTION_WITH_TASK_ONLY", False)
    )
    use_repair_negatives = include_repair_negatives or (
        replace_task_actions
        and getattr(sampling_config, "KTO_REPLACE_TASK_ACTION_INCLUDE_REPAIR_NEGATIVES", True)
    )

    with progress_bar(total=len(entries), desc="[kto] labeling entries", unit="entry") as progress:
        for game_entries in games.values():
            for entry in game_entries:
                progress.update(1)
                if not _matches_model_filter(entry, model_filters):
                    continue

                interaction = entry.get("interaction", {})
                system_prompt = interaction.get("system_prompt", "")
                raw_prompt = interaction.get("prompt", "")
                user_prompt = _format_user_prompt(raw_prompt)
                assistant_response = _completion_for_applied_action(entry)
                phase = _extract_phase(raw_prompt)

                if use_repair_negatives:
                    rows.extend(
                        _repair_negative_rows_for_entry(
                            entry,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            phase=phase,
                        )
                    )

                if drop_empty_response and not assistant_response:
                    continue

                label = reward_fn.label(entry, game_context=game_entries)
                if label is None:
                    continue

                action = utils.extract_action(entry)
                category = utils.action_category(action)
                kto_bucket = _kto_bucket_name(phase, category)
                if task_actions_only and kto_bucket != "task_action":
                    continue
                game_key = (entry.get("_log_path", ""), entry["game_index"])
                source_model = str(entry.get("player", {}).get("model", "")).strip()
                actor_role = str(entry.get("player", {}).get("identity", "")).strip()
                rows.append(
                    {
                        "prompt": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "completion": [{"role": "assistant", "content": assistant_response}],
                        "label": bool(label),
                        "phase": phase,
                        "source_model": source_model,
                        "actor_role": actor_role,
                        "kto_bucket": kto_bucket,
                        "task_action_type": _task_action_detail_type(action),
                        "sample_source": "applied_action",
                        "action_text": action,
                        "_kto_bucket": kto_bucket,
                        "_action_type": _task_action_type(action),
                        "_task_action_type": _task_action_detail_type(action),
                        "_entry": entry,
                        "_game_key": game_key,
                    }
                )

    labeled_rows = rows
    if task_actions_only:
        rows = [row for row in rows if row.get("_kto_bucket") == "task_action"]
        if apply_sampling and getattr(sampling_config, "KTO_TASK_ONLY_SAMPLING_ENABLED", True):
            rows = _sample_task_action_only_rows(rows, include_models)
        return _dataset_from_kto_rows(rows)

    if apply_sampling and sampling_config.KTO_SAMPLING_ENABLED:
        weights = flatten_kto_weights(
            sampling_config.KTO_LABEL_RATIOS,
            sampling_config.KTO_DETAIL_RATIOS,
            getattr(sampling_config, "KTO_ROLE_RATIOS", None),
        )
        use_role_ratios = bool(getattr(sampling_config, "KTO_ROLE_RATIOS", None))
        model_ratios = resolve_model_ratios(
            getattr(sampling_config, "KTO_MODEL_RATIOS", None),
            include_models,
        )
        weights = add_model_ratios_to_weights(weights, model_ratios)
        if post_sampling_reward_fn is not None:
            rows, _ = _sample_with_post_sampling_speech_verifier(
                rows,
                games=games,
                verifier_reward_fn=post_sampling_reward_fn,
                weights=weights,
                model_ratios=model_ratios,
                use_role_ratios=use_role_ratios,
                total_size=sampling_config.KTO_TOTAL_SIZE,
                seed=sampling_config.SAMPLING_SEED,
                allow_oversample=sampling_config.KTO_ALLOW_OVERSAMPLE,
                fill_shortage=sampling_config.KTO_FILL_SHORTAGE,
            )
        else:
            rows, _ = stratified_sample_records(
                rows,
                key_fn=lambda row: _sample_key(row, model_ratios, use_role_ratios),
                weights=weights,
                total_size=sampling_config.KTO_TOTAL_SIZE,
                seed=sampling_config.SAMPLING_SEED,
                allow_oversample=sampling_config.KTO_ALLOW_OVERSAMPLE,
                fill_shortage=sampling_config.KTO_FILL_SHORTAGE,
            )
        if replace_task_actions:
            rows = _replace_sampled_task_action_rows(
                source_rows=labeled_rows,
                sampled_rows=rows,
                include_models=include_models,
            )
        else:
            rows = _rebalance_task_action_rows(labeled_rows, rows, seed=sampling_config.SAMPLING_SEED)
    elif post_sampling_reward_fn is not None:
        rows = _verify_meeting_speech_rows(rows, games=games, verifier_reward_fn=post_sampling_reward_fn)

    return _dataset_from_kto_rows(rows)


def _replace_sampled_task_action_rows(
    source_rows: List[Dict[str, Any]],
    sampled_rows: List[Dict[str, Any]],
    include_models: Optional[Iterable[str]],
) -> List[Dict[str, Any]]:
    task_source_rows = [row for row in source_rows if row.get("_kto_bucket") == "task_action"]
    if not task_source_rows:
        return sampled_rows

    task_rows = _sample_task_action_only_rows(task_source_rows, include_models)
    non_task_rows = [row for row in sampled_rows if row.get("_kto_bucket") != "task_action"]
    print(
        "[kto] replaced sampled task_action rows with task-action-only build: "
        f"old_task={len(sampled_rows) - len(non_task_rows)} new_task={len(task_rows)} "
        f"kept_non_task={len(non_task_rows)}",
        flush=True,
    )
    return non_task_rows + task_rows


def _sample_task_action_only_rows(
    rows: List[Dict[str, Any]],
    include_models: Optional[Iterable[str]],
) -> List[Dict[str, Any]]:
    label_ratios = getattr(sampling_config, "KTO_TASK_ONLY_LABEL_RATIOS", {True: 0.5, False: 0.5})
    type_ratios = getattr(
        sampling_config,
        "KTO_TASK_ONLY_TYPE_RATIOS",
        {
            True: {"complete_task": 0.25, "movement": 0.45, "impostor_action": 0.20, "meeting_trigger": 0.08, "monitor": 0.02},
            False: {"complete_task": 0.25, "movement": 0.45, "impostor_action": 0.20, "meeting_trigger": 0.08, "monitor": 0.02},
        },
    )
    role_ratios = getattr(sampling_config, "KTO_TASK_ONLY_ROLE_RATIOS", getattr(sampling_config, "KTO_ROLE_RATIOS", None))
    model_ratios = resolve_model_ratios(
        getattr(sampling_config, "KTO_TASK_ONLY_MODEL_RATIOS", getattr(sampling_config, "KTO_MODEL_RATIOS", None)),
        include_models,
    )
    weights = flatten_kto_weights(label_ratios, type_ratios, role_ratios)
    weights = add_model_ratios_to_weights(weights, model_ratios)
    use_role_ratios = bool(role_ratios)
    total_size = getattr(sampling_config, "KTO_TASK_ONLY_TOTAL_SIZE", None)
    if total_size is None:
        total_size = getattr(sampling_config, "KTO_TOTAL_SIZE", None)

    sampled, report = stratified_sample_records(
        rows,
        key_fn=lambda row: _task_only_sample_key(row, model_ratios, use_role_ratios),
        weights=weights,
        total_size=total_size,
        seed=sampling_config.SAMPLING_SEED,
        allow_oversample=getattr(sampling_config, "KTO_TASK_ONLY_ALLOW_OVERSAMPLE", sampling_config.KTO_ALLOW_OVERSAMPLE),
        fill_shortage=getattr(sampling_config, "KTO_TASK_ONLY_FILL_SHORTAGE", sampling_config.KTO_FILL_SHORTAGE),
    )
    sampled = _rebalance_task_action_type_labels(rows, sampled, seed=sampling_config.SAMPLING_SEED)
    print(f"[kto] task-only sampling report: {report}", flush=True)
    return sampled


def _balanced_label_targets(
    total: int,
    available_true: int,
    available_false: int,
    true_ratio: float,
    false_ratio: float,
) -> Tuple[int, int]:
    if total <= 0:
        return 0, 0
    if available_true <= 0:
        return 0, min(total, available_false)
    if available_false <= 0:
        return min(total, available_true), 0

    ratio_sum = true_ratio + false_ratio
    desired_true = int(round(total * true_ratio / ratio_sum)) if ratio_sum > 0 else total
    desired_false = total - desired_true
    target_true = min(desired_true, available_true)
    target_false = min(desired_false, available_false)

    shortage = total - target_true - target_false
    if shortage > 0:
        add_true = min(shortage, available_true - target_true)
        target_true += add_true
        shortage -= add_true
    if shortage > 0:
        add_false = min(shortage, available_false - target_false)
        target_false += add_false
    return target_true, target_false


def _rebalance_task_action_type_labels(
    source_rows: List[Dict[str, Any]],
    sampled_rows: List[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    if not getattr(sampling_config, "KTO_TASK_ONLY_TYPE_LABEL_BALANCE_ENABLED", False):
        return sampled_rows

    label_ratios = getattr(sampling_config, "KTO_TASK_ONLY_TYPE_LABEL_RATIOS", {True: 0.6, False: 0.4})
    true_ratio = float(label_ratios.get(True, 0.0))
    false_ratio = float(label_ratios.get(False, 0.0))
    if true_ratio <= 0 and false_ratio <= 0:
        return sampled_rows

    source_by_type_label: Dict[str, Dict[bool, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in source_rows:
        action_type = str(row.get("_task_action_type", row.get("task_action_type", _TASK_ACTION_TYPE_FALLBACK)))
        source_by_type_label[action_type][bool(row["label"])].append(row)

    sampled_type_counts = Counter(
        str(row.get("_task_action_type", row.get("task_action_type", _TASK_ACTION_TYPE_FALLBACK)))
        for row in sampled_rows
    )
    rng = random.Random(seed + 1409)
    selected: List[Dict[str, Any]] = []
    report: Dict[str, Dict[str, int]] = {}

    for action_type in sorted(sampled_type_counts):
        total = sampled_type_counts[action_type]
        true_rows = list(source_by_type_label[action_type].get(True, []))
        false_rows = list(source_by_type_label[action_type].get(False, []))
        target_true, target_false = _balanced_label_targets(
            total,
            len(true_rows),
            len(false_rows),
            true_ratio,
            false_ratio,
        )
        rng.shuffle(true_rows)
        rng.shuffle(false_rows)
        selected.extend(true_rows[:target_true])
        selected.extend(false_rows[:target_false])
        report[action_type] = {"true": target_true, "false": target_false, "total": target_true + target_false}

    if len(selected) != len(sampled_rows):
        return sampled_rows
    rng.shuffle(selected)
    print(f"[kto] task-only type label balance report: {report}", flush=True)
    return selected


def _task_only_sample_key(
    row: Dict[str, Any],
    model_ratios: Dict[str, float],
    use_role_ratios: bool = False,
) -> Tuple[Any, ...]:
    key: Tuple[Any, ...] = (row.get("_task_action_type", _TASK_ACTION_TYPE_FALLBACK), bool(row["label"]))
    if use_role_ratios:
        key = (*key, row.get("actor_role", ""))
    return append_model_ratio_key(key, row.get("source_model", ""), model_ratios)


def _rebalance_task_action_rows(
    source_rows: List[Dict[str, Any]],
    sampled_rows: List[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    if not getattr(sampling_config, "KTO_TASK_ACTION_BALANCE_ENABLED", False):
        return sampled_rows

    sampled_task_rows = [row for row in sampled_rows if row.get("_kto_bucket") == "task_action"]
    if not sampled_task_rows:
        return sampled_rows

    source_by_action_label: Dict[str, Dict[bool, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in source_rows:
        if row.get("_kto_bucket") != "task_action":
            continue
        source_by_action_label[str(row.get("_action_type") or "<missing_action_line>")][bool(row["label"])].append(row)

    sampled_action_counts = Counter(str(row.get("_action_type") or "<missing_action_line>") for row in sampled_task_rows)
    rng = random.Random(seed + 991)
    selected_task_rows: List[Dict[str, Any]] = []
    report_rows = []

    for action_type in sorted(sampled_action_counts):
        target_total = sampled_action_counts[action_type]
        label_pools = source_by_action_label.get(action_type, {})
        true_pool = list(label_pools.get(True, []))
        false_pool = list(label_pools.get(False, []))
        rng.shuffle(true_pool)
        rng.shuffle(false_pool)

        pool_total = len(true_pool) + len(false_pool)
        if pool_total < target_total:
            print(
                "[kto] task_action rebalance skipped "
                f"action={action_type} target={target_total} pool={pool_total}",
                flush=True,
            )
            return sampled_rows

        label_ratios = getattr(sampling_config, "KTO_TASK_ACTION_LABEL_RATIOS", {True: 0.6, False: 0.4})
        balanceable = bool(true_pool and false_pool)

        if balanceable:
            true_ratio = float(label_ratios.get(True, 0.6))
        else:
            true_ratio = len(true_pool) / pool_total if pool_total else 0.0
        requested_true = max(0, min(target_total, int(round(target_total * true_ratio))))
        requested = {True: requested_true, False: target_total - requested_true}

        chosen, chosen_counts = _sample_task_action_labels(
            pools={True: true_pool, False: false_pool},
            requested=requested,
            target_total=target_total,
        )
        selected_task_rows.extend(chosen)
        report_rows.append(
            (
                action_type,
                balanceable,
                len(true_pool),
                len(false_pool),
                chosen_counts[True],
                chosen_counts[False],
            )
        )

    if len(selected_task_rows) != len(sampled_task_rows):
        print(
            "[kto] task_action rebalance did not preserve total; keeping original "
            f"original={len(sampled_task_rows)} new={len(selected_task_rows)}",
            flush=True,
        )
        return sampled_rows

    selected_iter = iter(selected_task_rows)
    rebalanced = [next(selected_iter) if row.get("_kto_bucket") == "task_action" else row for row in sampled_rows]
    summary = {
        action_type: {
            "balanced": balanceable,
            "pool_true": pool_true,
            "pool_false": pool_false,
            "selected_true": selected_true,
            "selected_false": selected_false,
        }
        for action_type, balanceable, pool_true, pool_false, selected_true, selected_false in report_rows
    }
    print(f"[kto] task_action label rebalance summary: {summary}", flush=True)
    return rebalanced


def _sample_task_action_labels(
    pools: Dict[bool, List[Dict[str, Any]]],
    requested: Dict[bool, int],
    target_total: int,
) -> Tuple[List[Dict[str, Any]], Counter]:
    selected: List[Dict[str, Any]] = []
    used = Counter()
    for label in (True, False):
        take = min(int(requested.get(label, 0)), len(pools.get(label, [])))
        if take > 0:
            selected.extend(pools[label][:take])
            used[label] += take

    shortage = target_total - len(selected)
    if shortage > 0:
        labels_by_remaining = sorted(
            (True, False),
            key=lambda label: len(pools.get(label, [])) - used[label],
            reverse=True,
        )
        for label in labels_by_remaining:
            if shortage <= 0:
                break
            pool = pools.get(label, [])
            take = min(shortage, len(pool) - used[label])
            if take <= 0:
                continue
            selected.extend(pool[used[label] : used[label] + take])
            used[label] += take
            shortage -= take

    return selected, used


def _row_key(row: Dict[str, Any], use_role_ratios: bool = False) -> Tuple[Any, ...]:
    key: Tuple[Any, ...] = (row["_kto_bucket"], bool(row["label"]))
    if use_role_ratios:
        key = (*key, row.get("actor_role", ""))
    return key


def _sample_key(row: Dict[str, Any], model_ratios: Dict[str, float], use_role_ratios: bool = False) -> Tuple[Any, ...]:
    return append_model_ratio_key(_row_key(row, use_role_ratios), row.get("source_model", ""), model_ratios)


def _verify_meeting_speech_row(
    row: Dict[str, Any],
    games: Dict[Tuple[str, str], List[Dict[str, Any]]],
    verifier_reward_fn: RewardFunction,
) -> Optional[Dict[str, Any]]:
    if row.get("_kto_bucket") != "meeting_speech":
        return row
    label = verifier_reward_fn.label(row["_entry"], game_context=games.get(row["_game_key"], []))
    if label is None:
        return None
    verified = dict(row)
    verified["label"] = bool(label)
    return verified


def _verify_meeting_speech_rows(
    rows: List[Dict[str, Any]],
    games: Dict[Tuple[str, str], List[Dict[str, Any]]],
    verifier_reward_fn: RewardFunction,
) -> List[Dict[str, Any]]:
    verified_rows: List[Dict[str, Any]] = []
    with progress_bar(total=len(rows), desc="[kto] post-verifying meeting speech", unit="row") as progress:
        for row in rows:
            verified = _verify_meeting_speech_row(row, games=games, verifier_reward_fn=verifier_reward_fn)
            if verified is not None:
                verified_rows.append(verified)
            progress.update(1)
    return verified_rows


def _sample_with_post_sampling_speech_verifier(
    rows: List[Dict[str, Any]],
    games: Dict[Tuple[str, str], List[Dict[str, Any]]],
    verifier_reward_fn: RewardFunction,
    weights: Dict[Tuple[Any, ...], float],
    model_ratios: Dict[str, float],
    use_role_ratios: bool,
    total_size: Optional[int],
    seed: int,
    allow_oversample: bool,
    fill_shortage: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    normalized_weights = {
        normalize_bucket_key(key): float(weight)
        for key, weight in weights.items()
        if float(weight) > 0
    }
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = normalize_bucket_key(_sample_key(row, model_ratios, use_role_ratios))
        if key in normalized_weights:
            buckets[key].append(row)

    rng = random.Random(seed)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    available = Counter({key: len(value) for key, value in buckets.items()})
    if total_size is None:
        total_size = len(rows) if allow_oversample else infer_feasible_total(available, normalized_weights)
    requested = allocate_counts(total_size, normalized_weights)

    selected_by_bucket: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    selected: List[Dict[str, Any]] = []
    selected_speech_source_ids = set()
    verified_speech_candidates: List[Tuple[int, Dict[str, Any]]] = []
    speech_fallback_selected = 0

    speech_requested = {
        key: requested_count
        for key, requested_count in requested.items()
        if key and key[0] == "meeting_speech" and requested_count > 0
    }
    speech_candidates = [
        row
        for key, bucket_rows in buckets.items()
        if key and key[0] == "meeting_speech"
        for row in bucket_rows
    ]
    rng.shuffle(speech_candidates)

    for key, requested_count in requested.items():
        if key in speech_requested:
            continue
        pool = buckets.get(key, [])
        if allow_oversample and pool and requested_count > len(pool):
            chosen = list(pool) + [rng.choice(pool) for _ in range(requested_count - len(pool))]
        else:
            chosen = pool[: min(requested_count, len(pool))]
        selected_by_bucket[key].extend(chosen)
        selected.extend(chosen)

    if speech_requested:
        round_idx = 0
        candidate_ptr = 0
        while candidate_ptr < len(speech_candidates):
            missing = {
                key: requested_count - len(selected_by_bucket[key])
                for key, requested_count in speech_requested.items()
                if len(selected_by_bucket[key]) < requested_count
            }
            if not missing:
                break

            round_idx += 1
            draw_count = min(sum(missing.values()), len(speech_candidates) - candidate_ptr)
            batch = speech_candidates[candidate_ptr : candidate_ptr + draw_count]
            candidate_ptr += draw_count
            print(
                "[kto] LLM speech sampling round "
                f"{round_idx}: verify={len(batch)} missing={dict(missing)} "
                f"remaining_candidates={len(speech_candidates) - candidate_ptr}",
                flush=True,
            )

            with progress_bar(total=len(batch), desc=f"[kto] LLM speech round {round_idx}", unit="call") as progress:
                for row in batch:
                    verified = _verify_meeting_speech_row(row, games=games, verifier_reward_fn=verifier_reward_fn)
                    progress.update(1)
                    if verified is None:
                        continue
                    source_id = id(row)
                    verified_speech_candidates.append((source_id, verified))
                    verified_key = normalize_bucket_key(_sample_key(verified, model_ratios, use_role_ratios))
                    if verified_key not in speech_requested:
                        continue
                    if len(selected_by_bucket[verified_key]) >= speech_requested[verified_key]:
                        continue
                    selected_by_bucket[verified_key].append(verified)
                    selected.append(verified)
                    selected_speech_source_ids.add(source_id)

            fallback_after_rounds = int(getattr(sampling_config, "KTO_SPEECH_FALLBACK_AFTER_ROUNDS", 1))
            if (
                fill_shortage
                and not allow_oversample
                and getattr(sampling_config, "KTO_SPEECH_FILL_SHORTAGE_FROM_VERIFIED", False)
                and round_idx >= fallback_after_rounds
            ):
                speech_fallback_selected += _fill_speech_shortage_from_verified(
                    selected=selected,
                    selected_by_bucket=selected_by_bucket,
                    selected_source_ids=selected_speech_source_ids,
                    verified_candidates=verified_speech_candidates,
                    speech_requested=speech_requested,
                    model_ratios=model_ratios,
                    use_role_ratios=use_role_ratios,
                    keep_label=bool(getattr(sampling_config, "KTO_SPEECH_FILL_SHORTAGE_KEEP_LABEL", True)),
                )
                if len(selected) >= sum(requested.values()):
                    break

    if fill_shortage and len(selected) < sum(requested.values()) and not allow_oversample:
        fill_keys = sorted(requested, key=lambda key: requested[key], reverse=True)
        for key in fill_keys:
            if key and key[0] == "meeting_speech":
                continue
            pool = buckets.get(key, [])
            used = len(selected_by_bucket[key])
            while used < len(pool) and len(selected) < sum(requested.values()):
                item = pool[used]
                selected_by_bucket[key].append(item)
                selected.append(item)
                used += 1

    selected_counts = Counter({key: len(value) for key, value in selected_by_bucket.items()})
    report = {
        "requested_total": sum(requested.values()),
        "selected_total": len(selected),
        "available_by_bucket": dict(available),
        "requested_by_bucket": dict(requested),
        "selected_by_bucket": dict(selected_counts),
        "missing_by_bucket": {
            key: max(0, requested.get(key, 0) - selected_counts.get(key, 0))
            for key in requested
        },
        "speech_fallback_selected": speech_fallback_selected,
    }
    return selected, report


def _model_ratio_weight(source_model: str, model_ratios: Dict[str, float]) -> float:
    if not model_ratios:
        return 1.0
    normalized_model = str(source_model or "").strip().lower()
    matches = [key for key in model_ratios if key in normalized_model]
    if not matches:
        return 0.0
    return float(model_ratios[max(matches, key=len)])


def _fill_speech_shortage_from_verified(
    selected: List[Dict[str, Any]],
    selected_by_bucket: Dict[Tuple[Any, ...], List[Dict[str, Any]]],
    selected_source_ids: set,
    verified_candidates: List[Tuple[int, Dict[str, Any]]],
    speech_requested: Dict[Tuple[Any, ...], int],
    model_ratios: Dict[str, float],
    use_role_ratios: bool,
    keep_label: bool,
) -> int:
    missing_by_label = Counter()
    for key, requested_count in speech_requested.items():
        selected_count = len(selected_by_bucket[key])
        if selected_count >= requested_count:
            continue
        missing_by_label[bool(key[1])] += requested_count - selected_count

    if not missing_by_label:
        return 0

    ranked_candidates = [
        (source_id, row)
        for source_id, row in verified_candidates
        if source_id not in selected_source_ids and row.get("_kto_bucket") == "meeting_speech"
    ]
    ranked_candidates.sort(
        key=lambda item: (
            _model_ratio_weight(item[1].get("source_model", ""), model_ratios),
            str(item[1].get("source_model", "")),
        ),
        reverse=True,
    )

    added = 0
    for source_id, row in ranked_candidates:
        label = bool(row["label"])
        if keep_label and missing_by_label[label] <= 0:
            continue
        if not keep_label and sum(missing_by_label.values()) <= 0:
            break

        key = normalize_bucket_key(_sample_key(row, model_ratios, use_role_ratios))
        selected_by_bucket[key].append(row)
        selected.append(row)
        selected_source_ids.add(source_id)
        added += 1

        if keep_label:
            missing_by_label[label] -= 1
        else:
            missing_label = max(missing_by_label, key=lambda item: missing_by_label[item])
            missing_by_label[missing_label] -= 1

    if added:
        remaining = {label: count for label, count in missing_by_label.items() if count > 0}
        print(
            "[kto] filled speech shortage from verified fallback "
            f"added={added} remaining_by_label={remaining}",
            flush=True,
        )
    return added


def _kto_bucket_name(phase: str, category: str) -> str:
    if phase == "Task phase":
        return "task_action"
    if phase == "Meeting phase" and category == "VOTE":
        return "meeting_vote"
    if phase == "Meeting phase" and category == "SPEAK":
        return "meeting_speech"
    return "other"


def _dataset_from_kto_rows(rows: List[Dict[str, Any]]) -> Dataset:
    columns = {
        "prompt": [],
        "completion": [],
        "label": [],
        "phase": [],
        "source_model": [],
        "actor_role": [],
        "kto_bucket": [],
        "task_action_type": [],
        "sample_source": [],
        "action_text": [],
    }
    for row in rows:
        columns["prompt"].append(row["prompt"])
        columns["completion"].append(row["completion"])
        columns["label"].append(row["label"])
        columns["phase"].append(row["phase"])
        columns["source_model"].append(row.get("source_model", ""))
        columns["actor_role"].append(row.get("actor_role", ""))
        kto_bucket = row.get("kto_bucket", row.get("_kto_bucket", ""))
        columns["kto_bucket"].append(kto_bucket)
        if kto_bucket == "task_action":
            columns["task_action_type"].append(row.get("task_action_type", row.get("_task_action_type", "")))
            columns["sample_source"].append(row.get("sample_source", ""))
            columns["action_text"].append(row.get("action_text", ""))
        else:
            columns["task_action_type"].append("")
            columns["sample_source"].append("")
            columns["action_text"].append("")
    return Dataset.from_dict(columns)


def save_dataset(dataset: Dataset, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dataset.save_to_disk(out_dir)


def dataset_summary(dataset: Dataset) -> Dict[str, Any]:
    labels = Counter(bool(label) for label in dataset["label"])
    phases = Counter(dataset["phase"])
    phase_labels = Counter(zip(dataset["phase"], map(bool, dataset["label"])))
    models = Counter(dataset["source_model"]) if "source_model" in dataset.column_names else Counter()
    actor_roles = Counter(dataset["actor_role"]) if "actor_role" in dataset.column_names else Counter()
    buckets = Counter(dataset["kto_bucket"]) if "kto_bucket" in dataset.column_names else Counter()
    task_action_types = Counter(dataset["task_action_type"]) if "task_action_type" in dataset.column_names else Counter()
    sample_sources = Counter(dataset["sample_source"]) if "sample_source" in dataset.column_names else Counter()
    return {
        "size": len(dataset),
        "positive": labels[True],
        "negative": labels[False],
        "phases": dict(phases),
        "phase_labels": {f"{phase}:{label}": count for (phase, label), count in phase_labels.items()},
        "source_models": dict(models),
        "actor_roles": dict(actor_roles),
        "kto_buckets": dict(buckets),
        "task_action_types": dict(task_action_types),
        "sample_sources": dict(sample_sources),
    }
