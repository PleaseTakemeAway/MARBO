"""
Build a role-prediction SFT dataset from belief-enabled compact logs.

Label policy:
    - After at least two meeting discussion rounds are visible in the context,
      actors receive oracle labels for every roster player.
    - Before that point, actors receive conservative evidence labels: self role,
      known Impostor teammates when privately available, and observed KILL/VENT
      actors as Impostor. Other players remain unknown unless previously
      established by the same deterministic rule state.

The output schema follows chat-style SFT datasets:
    {
        "prompt": [{"role": "system", "content": ...},
                   {"role": "user", "content": ...}],
        "completion": [{"role": "assistant", "content": ...}],
        "phase": str,
        "actor_role": str,
        "label_policy": str,
        "state_point": str,
        "state_timing": str,
        "event_key": str,
    }
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from datasets import Dataset

from . import sampling_config
from .dataset_sampling import (
    add_model_ratios_to_weights,
    append_model_ratio_key,
    resolve_model_ratios,
    sample_dataset,
)
from .progress import iter_progress, progress_bar


BeliefMap = Dict[int, str]
RoleMap = Dict[int, str]
RELATIONAL_BELIEF_LABELS = ("Relational Belief", "Belief State")
RELATIONAL_BELIEF_ENABLED_KEYS = ("Relational Belief Enabled", "Belief State Enabled")
PREVIOUS_RELATIONAL_BELIEF_KEYS = ("Previous Relational Belief", "Previous Belief State")


PLAYER_RE = re.compile(r"Player\s+(\d+)(?:\s*:\s*[A-Za-z]+)?", re.IGNORECASE)
KNOWN_IMPOSTORS_RE = re.compile(r"Known Impostors:\s*(\[[^\n]+\])", re.IGNORECASE)
ACTION_KILL_RE = re.compile(
    r"Timestep\s+\d+:\s*\[[^\]]+\]\s*"
    r"(?P<actor>Player\s+\d+(?:\s*:\s*[A-Za-z]+)?)\s+"
    r"KILL\s+"
    r"(?P<victim>Player\s+\d+(?:\s*:\s*[A-Za-z]+)?)\b",
    re.IGNORECASE,
)
ACTION_VENT_RE = re.compile(
    r"Timestep\s+\d+:\s*\[[^\]]+\]\s*"
    r"(?P<actor>Player\s+\d+(?:\s*:\s*[A-Za-z]+)?)\s+"
    r"VENT\b",
    re.IGNORECASE,
)
ACTION_PREFIX_RE = re.compile(
    r"\b(COMPLETE FAKE TASK|COMPLETE TASK|REPORT DEAD BODY|CALL MEETING|VIEW MONITOR|KILL|VENT|MOVE|VOTE|SPEAK)\b",
    re.IGNORECASE,
)
CURRENT_DISCUSSION_ROUND_RE = re.compile(
    r"Discussion round\s*\(\s*(?P<round>\d+)\s*/\s*(?P<total>\d+)\s*\)",
    re.IGNORECASE,
)
ACTION_HISTORY_DISCUSSION_ROUND_RE = re.compile(
    r"\[meeting phase\s*-\s*round\s+(?P<round>\d+)\]",
    re.IGNORECASE,
)
ORACLE_DISCUSSION_ROUND_THRESHOLD = 2


def normalize_role(raw_role: Any) -> str:
    role = str(raw_role or "").strip().lower()
    if role == "crewmate":
        return "Crewmate"
    if role == "impostor":
        return "Impostor"
    return "unknown"


def player_number(raw_name: Any) -> Optional[int]:
    match = PLAYER_RE.search(str(raw_name or ""))
    return int(match.group(1)) if match else None


def get_prompt(entry: Dict[str, Any]) -> Dict[str, Any]:
    prompt = entry.get("interaction", {}).get("prompt", {})
    return prompt if isinstance(prompt, dict) else {}


def get_all_info(entry: Dict[str, Any]) -> str:
    return str(get_prompt(entry).get("All Info", ""))


def get_phase(entry: Dict[str, Any]) -> str:
    return str(get_prompt(entry).get("Phase", "")).strip()


def get_actor_role(entry: Dict[str, Any]) -> str:
    return normalize_role(entry.get("player", {}).get("identity", ""))


def get_actor_name(entry: Dict[str, Any]) -> str:
    return str(entry.get("player", {}).get("name", "")).strip()


def get_model(entry: Dict[str, Any]) -> str:
    return str(entry.get("player", {}).get("model", "")).strip()


def is_role_prediction_entry(entry: Dict[str, Any]) -> bool:
    if entry.get("type") == "role_prediction":
        return True
    prompt = get_prompt(entry)
    response = entry.get("interaction", {}).get("response")
    return (
        any(bool(prompt.get(key)) for key in RELATIONAL_BELIEF_ENABLED_KEYS)
        and isinstance(response, dict)
        and any(label in response for label in RELATIONAL_BELIEF_LABELS)
        and "Action" not in response
        and "Thinking Process" not in response
    )


def roster_numbers(entry: Dict[str, Any]) -> List[int]:
    nums: List[int] = []
    for raw_player in get_prompt(entry).get("Player Roster", []) or []:
        num = player_number(raw_player)
        if num is not None:
            nums.append(num)
    return nums


def known_impostor_numbers(entry: Dict[str, Any]) -> List[int]:
    system_prompt = str(entry.get("interaction", {}).get("system_prompt", ""))
    match = KNOWN_IMPOSTORS_RE.search(system_prompt)
    if not match:
        return []
    return [int(player_match.group(1)) for player_match in PLAYER_RE.finditer(match.group(1))]


def context_text(entry: Dict[str, Any], include_previous_event_memory: bool = False) -> str:
    prompt = get_prompt(entry)
    parts = [str(prompt.get("All Info", ""))]
    if include_previous_event_memory:
        parts.append(str(prompt.get("Previous Event Memory", "")))
    return "\n".join(parts)


def parse_hard_evidence(
    entry: Dict[str, Any],
    task_label_mode: str = "strict_observable",
    include_previous_event_memory: bool = False,
) -> List[Tuple[str, int]]:
    """Return hard role facts as (role, player_num).

    Speech claims are intentionally ignored here. The KILL/VENT regexes only
    match action-history style lines such as:

        Timestep 3: [task] Player 2: blue KILL Player 5: red

    Only the actor of KILL/VENT is labeled as Impostor. Victims and available
    KILL targets are left unknown because they are not reliable public belief
    labels before enough discussion context is available.
    """

    _ = task_label_mode
    text = context_text(entry, include_previous_event_memory=include_previous_event_memory)
    evidence: List[Tuple[str, int]] = []

    for match in ACTION_KILL_RE.finditer(text):
        actor = player_number(match.group("actor"))
        if actor is not None:
            evidence.append(("Impostor", actor))

    for match in ACTION_VENT_RE.finditer(text):
        actor = player_number(match.group("actor"))
        if actor is not None:
            evidence.append(("Impostor", actor))

    return evidence


def complete_role_map_for_roster(
    roster: List[int],
    role_map: RoleMap,
    num_impostors: Optional[int] = 2,
) -> RoleMap:
    completed = {num: role_map[num] for num in roster if role_map.get(num) in {"Crewmate", "Impostor"}}
    missing = [num for num in roster if num not in completed]
    if not missing or num_impostors is None:
        return completed

    known_impostors = sum(1 for role in completed.values() if role == "Impostor")
    known_crewmates = sum(1 for role in completed.values() if role == "Crewmate")
    if known_impostors == num_impostors:
        for num in missing:
            completed[num] = "Crewmate"
    elif known_crewmates == len(roster) - num_impostors:
        for num in missing:
            completed[num] = "Impostor"
    return completed


def oracle_label(
    entry: Dict[str, Any],
    role_map: RoleMap,
    num_impostors: Optional[int] = 2,
) -> BeliefMap:
    roster = roster_numbers(entry)
    completed_role_map = complete_role_map_for_roster(roster, role_map, num_impostors=num_impostors)
    return {num: completed_role_map.get(num, "unknown") for num in roster}


def seed_private_known_roles(entry: Dict[str, Any], label: BeliefMap) -> None:
    actor_num = player_number(get_actor_name(entry))
    actor_role = get_actor_role(entry)
    if actor_num in label:
        label[actor_num] = actor_role

    if actor_role == "Impostor":
        for known_num in known_impostor_numbers(entry):
            if known_num in label:
                label[known_num] = "Impostor"


def current_discussion_round(entry: Dict[str, Any]) -> Optional[int]:
    text = context_text(entry, include_previous_event_memory=False)
    current_match = CURRENT_DISCUSSION_ROUND_RE.search(text)
    if current_match:
        return int(current_match.group("round"))

    rounds = [
        int(match.group("round"))
        for match in ACTION_HISTORY_DISCUSSION_ROUND_RE.finditer(text)
    ]
    return max(rounds) if rounds else None


def has_enough_discussion_context(
    entry: Dict[str, Any],
    threshold: int = ORACLE_DISCUSSION_ROUND_THRESHOLD,
) -> bool:
    if get_phase(entry) != "Meeting phase":
        return False
    round_idx = current_discussion_round(entry)
    return round_idx is not None and round_idx >= threshold


def update_rule_state(
    entry: Dict[str, Any],
    prior: Optional[BeliefMap],
    task_label_mode: str = "strict_observable",
    include_previous_event_memory: bool = False,
) -> BeliefMap:
    label = {num: (prior or {}).get(num, "unknown") for num in roster_numbers(entry)}

    seed_private_known_roles(entry, label)

    for role, target_num in parse_hard_evidence(
        entry,
        task_label_mode=task_label_mode,
        include_previous_event_memory=include_previous_event_memory,
    ):
        if target_num in label:
            label[target_num] = role

    return label


def label_policy_for_entry(entry: Dict[str, Any]) -> str:
    if has_enough_discussion_context(entry):
        return "oracle"
    return "evidence_rule"


def target_label_for_entry(
    entry: Dict[str, Any],
    role_map: RoleMap,
    rule_prior: Optional[BeliefMap],
    task_label_mode: str = "strict_observable",
    include_previous_event_memory: bool = False,
    num_impostors: Optional[int] = 2,
    force_oracle: bool = False,
) -> Tuple[BeliefMap, BeliefMap, str]:
    """Return (target_label, updated_rule_state, label_policy)."""

    updated_rule = update_rule_state(
        entry,
        prior=rule_prior,
        task_label_mode=task_label_mode,
        include_previous_event_memory=include_previous_event_memory,
    )
    policy = "oracle" if force_oracle else label_policy_for_entry(entry)
    if policy == "oracle":
        return oracle_label(entry, role_map, num_impostors=num_impostors), updated_rule, policy
    return updated_rule, updated_rule, policy


def format_belief_state(label: BeliefMap) -> str:
    lines = ["[Relational Belief]"]
    for num in sorted(label):
        lines.append(f"- Player {num}: Role={normalize_role(label[num])}")
    return "\n".join(lines)


def format_roster(entry: Dict[str, Any]) -> str:
    return "\n".join(f"- {player}" for player in get_prompt(entry).get("Player Roster", []) or [])


def role_prediction_system_prompt(entry: Dict[str, Any]) -> str:
    base_prompt = str(entry.get("interaction", {}).get("system_prompt", "")).strip()
    policy = (
        "\n\nRole prediction SFT labeling policy:\n"
        "- Before at least two discussion rounds are visible, keep other players unknown unless hard evidence fixes their role.\n"
        "- Hard evidence means an observed KILL or VENT; mark only the actor of that action as Impostor.\n"
        "- If you are an Impostor, mark yourself and known Impostor teammates as Impostor; keep others unknown before the discussion threshold.\n"
        "- After at least two discussion rounds are visible, use the full meeting context to predict every player's true role.\n"
        "Return only [Relational Belief]."
    )
    return f"{base_prompt}{policy}" if base_prompt else policy.strip()


def previous_relational_belief(prompt: Dict[str, Any]) -> Any:
    for key in PREVIOUS_RELATIONAL_BELIEF_KEYS:
        if key in prompt:
            return prompt.get(key, "")
    return ""


def format_role_prediction_user_prompt(entry: Dict[str, Any]) -> str:
    prompt = get_prompt(entry)
    return (
        f"Phase: {prompt.get('Phase', '')}\n\n"
        f"Player roster:\n{format_roster(entry)}\n\n"
        f"Previous Relational Belief:\n{previous_relational_belief(prompt)}\n\n"
        f"Previous Event Memory:\n{prompt.get('Previous Event Memory', '')}\n\n"
        f"Current observations and messages:\n{prompt.get('All Info', '')}\n\n"
        "Update only the relational belief over every player's role."
    )


def normalize_model_filters(include_models: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if not include_models:
        return ()
    filters: List[str] = []
    for raw_model in include_models:
        filters.extend(model.strip().lower() for model in str(raw_model).split(",") if model.strip())
    return tuple(filters)


def matches_model_filter(entry: Dict[str, Any], model_filters: Tuple[str, ...]) -> bool:
    if not model_filters:
        return True
    model = get_model(entry).lower()
    return any(model_filter in model for model_filter in model_filters)


def resolve_log_paths(log_paths: Iterable[str], exclude_sorted: bool = True) -> List[str]:
    resolved: List[str] = []
    for raw_path in log_paths:
        path = os.path.abspath(os.path.expanduser(str(raw_path)))
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                if "agent-logs-compact.json" not in files:
                    continue
                compact_path = os.path.join(root, "agent-logs-compact.json")
                if exclude_sorted and "sorted" in compact_path.split(os.sep):
                    continue
                resolved.append(compact_path)
        else:
            if exclude_sorted and "sorted" in path.split(os.sep):
                continue
            resolved.append(path)
    return sorted(dict.fromkeys(resolved))


def iter_log_entries(path: str) -> Iterator[Dict[str, Any]]:
    abs_path = os.path.abspath(path)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry["_log_path"] = abs_path
            yield entry


def count_log_entries(log_paths: Iterable[str]) -> int:
    total = 0
    paths = list(log_paths)
    for path in iter_progress(paths, desc="[sft] counting log entries", unit="file"):
        with open(path, "r") as f:
            total += sum(1 for line in f if line.strip())
    return total


def collect_role_maps(
    log_paths: Iterable[str],
    total_entries: Optional[int] = None,
) -> Dict[Tuple[str, str], RoleMap]:
    role_maps: Dict[Tuple[str, str], RoleMap] = defaultdict(dict)
    paths = list(log_paths)
    with progress_bar(total=total_entries, desc="[sft] collecting role maps", unit="entry") as progress:
        for path in paths:
            for entry in iter_log_entries(path):
                progress.update(1)
                game_index = str(entry.get("game_index", ""))
                actor_num = player_number(get_actor_name(entry))
                actor_role = get_actor_role(entry)
                if game_index and actor_num is not None and actor_role in {"Crewmate", "Impostor"}:
                    role_maps[(entry["_log_path"], game_index)][actor_num] = actor_role
    return role_maps


def event_key(entry: Dict[str, Any]) -> str:
    return "::".join(
        [
            str(entry.get("_log_path", "")),
            str(entry.get("game_index", "")),
            str(entry.get("step", "")),
            get_actor_name(entry),
            str(entry.get("timestamp", "")),
        ]
    )


def state_reconstruction_point(entry: Dict[str, Any], label_policy: str) -> str:
    phase = get_phase(entry)
    actor_role = get_actor_role(entry)
    if phase == "Task phase" and label_policy == "evidence_rule" and actor_role == "Crewmate":
        return "task_crewmate_rule"
    if phase == "Task phase" and label_policy == "evidence_rule" and actor_role == "Impostor":
        return "task_impostor_rule"
    if phase == "Meeting phase" and label_policy == "evidence_rule" and actor_role == "Crewmate":
        return "meeting_crewmate_rule"
    if phase == "Meeting phase" and label_policy == "evidence_rule" and actor_role == "Impostor":
        return "meeting_impostor_rule"
    if phase == "Meeting phase" and label_policy == "oracle" and actor_role == "Crewmate":
        return "meeting_crewmate_oracle"
    if phase == "Meeting phase" and label_policy == "oracle" and actor_role == "Impostor":
        return "meeting_impostor_oracle"
    return f"{phase}:{label_policy}:{actor_role}"


def state_reconstruction_timing(entry: Dict[str, Any]) -> str:
    phase = get_phase(entry)
    if phase == "Task phase":
        return "task_phase"

    if phase != "Meeting phase":
        return "unknown_phase"

    action_prefix = latest_visible_action_prefix(get_all_info(entry))
    if not action_prefix:
        return "meeting_start"
    if action_prefix == "SPEAK":
        return "meeting_after_speech"
    if action_prefix == "VOTE":
        return "meeting_after_vote"
    if action_prefix in {"REPORT DEAD BODY", "CALL MEETING"}:
        return "meeting_start"
    return "meeting_after_other"


def latest_visible_action_prefix(all_info: str) -> str:
    for line in reversed(str(all_info or "").splitlines()):
        match = ACTION_PREFIX_RE.search(line)
        if match:
            return match.group(1).upper()
    return ""


def iter_role_prediction_sft_rows(
    log_paths: Iterable[str],
    role_maps: Dict[Tuple[str, str], RoleMap],
    include_models: Optional[Iterable[str]] = None,
    task_label_mode: str = "strict_observable",
    include_previous_event_memory: bool = False,
    num_impostors: Optional[int] = 2,
    total_entries: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    model_filters = normalize_model_filters(include_models)
    rule_states: Dict[Tuple[str, str, str], BeliefMap] = {}
    oracle_unlocked_games: Dict[Tuple[str, str], bool] = {}

    paths = list(log_paths)
    with progress_bar(total=total_entries, desc="[sft] building rows", unit="entry") as progress:
        for path in paths:
            for entry in iter_log_entries(path):
                progress.update(1)
                if not is_role_prediction_entry(entry):
                    continue
                if not matches_model_filter(entry, model_filters):
                    continue

                game_index = str(entry.get("game_index", ""))
                actor_name = get_actor_name(entry)
                game_key = (entry["_log_path"], game_index)
                if has_enough_discussion_context(entry):
                    oracle_unlocked_games[game_key] = True
                role_map = role_maps.get((entry["_log_path"], game_index), {})
                state_key = (entry["_log_path"], game_index, actor_name)
                target_label, updated_rule, policy = target_label_for_entry(
                    entry,
                    role_map=role_map,
                    rule_prior=rule_states.get(state_key),
                    task_label_mode=task_label_mode,
                    include_previous_event_memory=include_previous_event_memory,
                    num_impostors=num_impostors,
                    force_oracle=oracle_unlocked_games.get(game_key, False),
                )
                rule_states[state_key] = updated_rule
                state_point = state_reconstruction_point(entry, policy)
                state_timing = state_reconstruction_timing(entry)

                yield {
                    "prompt": [
                        {"role": "system", "content": role_prediction_system_prompt(entry)},
                        {"role": "user", "content": format_role_prediction_user_prompt(entry)},
                    ],
                    "completion": [{"role": "assistant", "content": format_belief_state(target_label)}],
                    "phase": get_phase(entry),
                    "source_model": get_model(entry),
                    "actor_role": get_actor_role(entry),
                    "label_policy": policy,
                    "state_point": state_point,
                    "state_timing": state_timing,
                    "task_label_mode": task_label_mode,
                    "event_key": event_key(entry),
                }


def build_role_prediction_sft_dataset(
    log_paths: Iterable[str],
    include_models: Optional[Iterable[str]] = None,
    task_label_mode: str = "strict_observable",
    include_previous_event_memory: bool = False,
    num_impostors: Optional[int] = 2,
    apply_sampling: bool = True,
) -> Dataset:
    resolved_paths = list(log_paths)
    total_log_entries = count_log_entries(resolved_paths)
    role_maps = collect_role_maps(resolved_paths, total_entries=total_log_entries)
    rows = list(
        iter_role_prediction_sft_rows(
            resolved_paths,
            role_maps,
            include_models=include_models,
            task_label_mode=task_label_mode,
            include_previous_event_memory=include_previous_event_memory,
            num_impostors=num_impostors,
            total_entries=total_log_entries,
        )
    )
    if rows:
        dataset = Dataset.from_list(rows)
    else:
        dataset = Dataset.from_dict(
            {
                "prompt": [],
                "completion": [],
                "phase": [],
                "source_model": [],
                "actor_role": [],
                "label_policy": [],
                "state_point": [],
                "state_timing": [],
                "task_label_mode": [],
                "event_key": [],
            }
        )
    if apply_sampling and sampling_config.SFT_SAMPLING_ENABLED:
        model_ratios = resolve_model_ratios(
            getattr(sampling_config, "SFT_MODEL_RATIOS", None),
            include_models,
        )
        weights = add_model_ratios_to_weights(
            sampling_config.SFT_STATE_POINT_RATIOS,
            model_ratios,
        )
        dataset, _ = sample_dataset(
            dataset,
            key_fn=lambda row: append_model_ratio_key(
                sft_ratio_key(row),
                row.get("source_model", ""),
                model_ratios,
            ),
            weights=weights,
            total_size=sampling_config.SFT_TOTAL_SIZE,
            seed=sampling_config.SAMPLING_SEED,
            allow_oversample=sampling_config.SFT_ALLOW_OVERSAMPLE,
            fill_shortage=sampling_config.SFT_FILL_SHORTAGE,
        )
    return dataset


def sft_ratio_key(row: Dict[str, Any]) -> Any:
    fields = getattr(sampling_config, "SFT_RATIO_FIELDS", None)
    if fields is None:
        field = getattr(sampling_config, "SFT_RATIO_FIELD", "state_point")
        return row[field]
    if isinstance(fields, str):
        return row[fields]
    values = tuple(row[field] for field in fields)
    return values[0] if len(values) == 1 else values


def dataset_summary(dataset: Dataset) -> Dict[str, Any]:
    phases = Counter(dataset["phase"])
    models = Counter(dataset["source_model"]) if "source_model" in dataset.column_names else Counter()
    actor_roles = Counter(dataset["actor_role"])
    policies = Counter(dataset["label_policy"])
    state_points = Counter(dataset["state_point"]) if "state_point" in dataset.column_names else Counter()
    state_timings = Counter(dataset["state_timing"]) if "state_timing" in dataset.column_names else Counter()
    phase_policies = Counter(zip(dataset["phase"], dataset["label_policy"]))
    return {
        "size": len(dataset),
        "phases": dict(phases),
        "source_models": dict(models),
        "actor_roles": dict(actor_roles),
        "label_policies": dict(policies),
        "state_points": dict(state_points),
        "state_timings": dict(state_timings),
        "phase_policies": {f"{phase}:{policy}": count for (phase, policy), count in phase_policies.items()},
    }


def save_dataset(dataset: Dataset, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dataset.save_to_disk(out_dir)
