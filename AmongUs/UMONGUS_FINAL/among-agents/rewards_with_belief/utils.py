import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .common_utils import *  # noqa: F401,F403
from . import common_utils as base_utils


BELIEF_LINE_RE = re.compile(
    r"^\s*-\s*(?P<player>Player\s+\d+(?:\s*:\s*[A-Za-z]+)?)\s*:\s*Role\s*=\s*(?P<role>Crewmate|Impostor|unknown)\b",
    re.IGNORECASE | re.MULTILINE,
)
PLAYER_NUM_RE = re.compile(r"Player\s+(\d+)", re.IGNORECASE)
RELATIONAL_BELIEF_LABELS = ("Relational Belief", "Belief State")
RELATIONAL_BELIEF_ENABLED_KEYS = ("Relational Belief Enabled", "Belief State Enabled")
PREVIOUS_RELATIONAL_BELIEF_KEYS = ("Previous Relational Belief", "Previous Belief State")


def normalize_belief_role(raw_role: str) -> str:
    role = str(raw_role or "").strip().lower()
    if role == "crewmate":
        return "Crewmate"
    if role == "impostor":
        return "Impostor"
    return "unknown"


def normalize_player_name(raw_name: str) -> str:
    name = re.sub(r"\s+", " ", str(raw_name or "").strip())
    name = re.sub(r"\s*:\s*", ": ", name)
    return name


def player_number(name: Optional[str]) -> Optional[int]:
    match = PLAYER_NUM_RE.search(str(name or ""))
    return int(match.group(1)) if match else None


def parse_belief_text(text: str) -> Dict[str, str]:
    """Parse current categorical role beliefs from Relational Belief text."""

    beliefs: Dict[str, str] = {}
    for match in BELIEF_LINE_RE.finditer(text or ""):
        player = normalize_player_name(match.group("player"))
        beliefs[player] = normalize_belief_role(match.group("role"))
    return beliefs


def belief_text_from_entry(entry: Dict[str, Any]) -> str:
    prompt = base_utils.get_prompt(entry)
    response = entry.get("interaction", {}).get("response")

    if isinstance(prompt, dict):
        memory = prompt.get("Memory")
        if isinstance(memory, str) and any(label in memory for label in RELATIONAL_BELIEF_LABELS):
            return memory

    if isinstance(response, dict):
        belief_state = next(
            (response.get(label) for label in RELATIONAL_BELIEF_LABELS if isinstance(response.get(label), str)),
            "",
        )
        if isinstance(belief_state, str) and belief_state.strip():
            return belief_state

    if isinstance(prompt, dict):
        previous = next(
            (
                prompt.get(key)
                for key in PREVIOUS_RELATIONAL_BELIEF_KEYS
                if isinstance(prompt.get(key), str)
            ),
            "",
        )
        if isinstance(previous, str) and previous.strip():
            return previous

    return ""


def is_belief_update_entry(entry: Dict[str, Any]) -> bool:
    prompt = base_utils.get_prompt(entry)
    response = entry.get("interaction", {}).get("response")
    return (
        isinstance(prompt, dict)
        and any(bool(prompt.get(key)) for key in RELATIONAL_BELIEF_ENABLED_KEYS)
        and isinstance(response, dict)
        and any(label in response for label in RELATIONAL_BELIEF_LABELS)
        and "Action" not in response
        and "Thinking Process" not in response
    )


def is_clean_belief_action_entry(entry: Dict[str, Any]) -> bool:
    return (
        base_utils.is_belief_enabled_entry(entry)
        and not base_utils.is_fallback_or_random_entry(entry)
        and base_utils.is_action_entry(entry)
    )


def _sort_key(entry: Dict[str, Any]) -> Tuple[int, str, int, str]:
    kind_order = 0 if is_belief_update_entry(entry) else 1
    return (
        int(entry.get("step", -1)),
        str(entry.get("timestamp", "")),
        kind_order,
        base_utils.get_player_name(entry),
    )


def event_is_before(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return _sort_key(a) < _sort_key(b)


def latest_belief_before_entry(
    entry: Dict[str, Any],
    game_context: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    current_player = base_utils.get_player_name(entry)
    current_key = _sort_key(entry)
    latest: Dict[str, str] = {}

    for item in sorted(game_context or [], key=_sort_key):
        if item is entry or _sort_key(item) >= current_key:
            break
        if base_utils.get_player_name(item) != current_player:
            continue
        if not is_belief_update_entry(item):
            continue
        parsed = parse_belief_text(belief_text_from_entry(item))
        if parsed:
            latest = parsed
    return latest


def actor_belief(
    entry: Dict[str, Any],
    game_context: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    parsed = parse_belief_text(belief_text_from_entry(entry))
    if parsed:
        return parsed
    return latest_belief_before_entry(entry, game_context)


def belief_role_for_player(beliefs: Dict[str, str], player_name: Optional[str]) -> str:
    if not player_name:
        return ""

    normalized = normalize_player_name(player_name)
    if normalized in beliefs:
        return beliefs[normalized]

    wanted_num = player_number(normalized)
    if wanted_num is None:
        return ""

    matches = [
        role
        for belief_player, role in beliefs.items()
        if player_number(belief_player) == wanted_num
    ]
    return matches[0] if len(matches) == 1 else ""


def believed_impostor_targets(
    entry: Dict[str, Any],
    game_context: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    actor = normalize_player_name(base_utils.get_player_name(entry))
    return [
        player
        for player, role in actor_belief(entry, game_context).items()
        if role == "Impostor" and normalize_player_name(player) != actor
    ]


def action_target_belief_role(
    entry: Dict[str, Any],
    game_context: Optional[List[Dict[str, Any]]] = None,
) -> str:
    action = base_utils.extract_action(entry)
    target = base_utils.extract_action_target(action)
    return belief_role_for_player(actor_belief(entry, game_context), target)


def _target_tokens(names: Iterable[str]) -> Dict[str, List[str]]:
    tokens: Dict[str, List[str]] = {}
    for name in names:
        normalized = normalize_player_name(name)
        values = []
        pnum = player_number(normalized)
        if pnum is not None:
            values.append(f"player {pnum}")
        if ":" in normalized:
            color = normalized.split(":", 1)[1].strip().lower()
            if color:
                values.append(color)
        tokens[normalized] = values
    return tokens


def accusation_targets(
    speech: str,
    game_context: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Return players explicitly accused in speech by simple auditable regexes."""

    role_map = base_utils.build_role_map(game_context)
    if not speech or not role_map:
        return []

    lower = speech.lower()
    accused = set()
    for player, tokens in _target_tokens(role_map.keys()).items():
        escaped = [re.escape(token) for token in tokens]
        if not escaped:
            continue
        token_alt = r"(?:%s)" % "|".join(escaped)
        patterns = (
            rf"\b(?:vote|eject|suspect|accuse|accusing|saw|seen|watched|caught)\s+{token_alt}\b",
            rf"\b(?:it's|it is|impostor is|imposter is)\s+{token_alt}\b",
            rf"\b{token_alt}\s+(?:is|was|seems|looks|acted|killed|kill|vented|vent|lied|lying|deflect|faked|fake)\b",
            rf"\b{token_alt}\s+.*?\b(?:suspicious|sus|impostor|imposter)\b",
        )
        if any(re.search(pattern, lower) for pattern in patterns):
            accused.add(player)
    return sorted(accused, key=lambda name: player_number(name) or 999)


def _opponent_role(role: str) -> str:
    if role == "Crewmate":
        return "Impostor"
    if role == "Impostor":
        return "Crewmate"
    return ""


def relevant_listener_roles_for_speaker(role: str) -> Tuple[str, ...]:
    """
    Listener side used for speech-induced belief measurement.

    Crewmates win by helping the crew identify Impostors, so Crewmate speech is
    evaluated against Crewmate listeners. Impostors win by misleading
    Crewmates, so Impostor speech is also evaluated against Crewmate listeners.
    """

    if role in {"Crewmate", "Impostor"}:
        return ("Crewmate",)
    return ()


def _has_intervening_meeting_action(
    start: Dict[str, Any],
    end: Dict[str, Any],
    game_context: Optional[List[Dict[str, Any]]],
) -> bool:
    start_key = _sort_key(start)
    end_key = _sort_key(end)
    for item in game_context or []:
        item_key = _sort_key(item)
        if item_key <= start_key or item_key >= end_key:
            continue
        if is_belief_update_entry(item):
            continue
        if base_utils.get_phase(item) != "Meeting phase":
            continue
        if base_utils.is_action_entry(item):
            return True
    return False


def speech_event_id(entry: Dict[str, Any]) -> str:
    prompt = base_utils.get_prompt(entry)
    value = prompt.get("Speech Event Id") if isinstance(prompt, dict) else ""
    return str(value or "").strip()


def belief_update_trigger(entry: Dict[str, Any]) -> str:
    prompt = base_utils.get_prompt(entry)
    value = prompt.get("Belief Update Trigger") if isinstance(prompt, dict) else ""
    return str(value or "").strip()


def listener_belief_transitions_after_speech(
    entry: Dict[str, Any],
    game_context: Optional[List[Dict[str, Any]]] = None,
    listener_roles: Optional[Iterable[str]] = None,
    require_same_step: bool = True,
    require_no_intervening_action: bool = True,
    include_unchanged_pairs: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compare listener-side logged beliefs immediately before and after a speech.

    Prefer explicit Speech Event Id pairs when available. Older logs fall back
    to timestamp/order matching, where the "after" belief update must be the
    observer's first belief update after the speech.
    """

    if base_utils.get_phase(entry) != "Meeting phase":
        return []
    if base_utils.action_category(base_utils.extract_action(entry)) != "SPEAK":
        return []

    speaker_role = base_utils.get_player_role(entry)
    speaker = normalize_player_name(base_utils.get_player_name(entry))
    allowed_listener_roles = set(listener_roles or relevant_listener_roles_for_speaker(speaker_role))
    if not allowed_listener_roles:
        return []

    transitions: List[Dict[str, Any]] = []
    role_map = base_utils.build_role_map(game_context)

    def add_transition(
        observer: str,
        before: Dict[str, str],
        after: Dict[str, str],
        after_entry: Dict[str, Any],
        pairing_mode: str,
    ) -> None:
        changes = []
        for target in sorted(role_map.keys(), key=lambda name: player_number(name) or 999):
            before_role = belief_role_for_player(before, target)
            after_role = belief_role_for_player(after, target)
            if not before_role or not after_role or before_role == after_role:
                continue
            true_role = base_utils.role_of(target, role_map)
            if true_role not in {"Crewmate", "Impostor"}:
                continue
            changes.append(
                {
                    "target": normalize_player_name(target),
                    "true_role": true_role,
                    "before": before_role,
                    "after": after_role,
                }
            )
        if changes or include_unchanged_pairs:
            transitions.append(
                {
                    "observer": observer,
                    "observer_role": base_utils.get_player_role(after_entry),
                    "after_step": after_entry.get("step"),
                    "after_timestamp": after_entry.get("timestamp"),
                    "pairing_mode": pairing_mode,
                    "speech_event_id": speech_event_id(entry),
                    "changes": changes,
                }
            )

    event_id = speech_event_id(entry)
    if event_id:
        before_by_observer: Dict[str, Dict[str, str]] = {}
        first_after_by_observer: Dict[str, Tuple[Dict[str, str], Dict[str, Any]]] = {}
        for item in sorted(game_context or [], key=_sort_key):
            if not is_belief_update_entry(item):
                continue
            if speech_event_id(item) != event_id:
                continue
            if base_utils.get_player_role(item) not in allowed_listener_roles:
                continue
            observer = normalize_player_name(base_utils.get_player_name(item))
            if observer == speaker:
                continue
            parsed = parse_belief_text(belief_text_from_entry(item))
            if not parsed:
                continue
            trigger = belief_update_trigger(item)
            if trigger == "pre_speech":
                before_by_observer[observer] = parsed
            elif trigger == "post_speech" and observer not in first_after_by_observer:
                first_after_by_observer[observer] = (parsed, item)

        for observer, before in before_by_observer.items():
            if observer not in first_after_by_observer:
                continue
            after, after_entry = first_after_by_observer[observer]
            add_transition(observer, before, after, after_entry, "speech_event_id")
        return transitions

    speech_key = _sort_key(entry)
    step = entry.get("step")
    latest_before: Dict[str, Dict[str, str]] = {}
    first_after: Dict[str, Tuple[Dict[str, str], Dict[str, Any]]] = {}

    for item in sorted(game_context or [], key=_sort_key):
        if not is_belief_update_entry(item):
            continue
        if base_utils.get_player_role(item) not in allowed_listener_roles:
            continue
        if require_same_step and item.get("step") != step:
            continue

        observer = normalize_player_name(base_utils.get_player_name(item))
        if observer == speaker:
            continue
        parsed = parse_belief_text(belief_text_from_entry(item))
        if not parsed:
            continue

        item_key = _sort_key(item)
        if item_key < speech_key:
            latest_before[observer] = parsed
            continue

        if item_key > speech_key and observer not in first_after:
            if require_no_intervening_action and _has_intervening_meeting_action(entry, item, game_context):
                continue
            first_after[observer] = (parsed, item)

    for observer, before in latest_before.items():
        if observer not in first_after:
            continue
        after, after_entry = first_after[observer]
        add_transition(observer, before, after, after_entry, "legacy_order")
    return transitions


def opponent_belief_transitions_after_speech(
    entry: Dict[str, Any],
    game_context: Optional[List[Dict[str, Any]]] = None,
    require_same_step: bool = True,
    require_no_intervening_action: bool = True,
) -> List[Dict[str, Any]]:
    """Backward-compatible name for listener belief transitions."""

    return listener_belief_transitions_after_speech(
        entry,
        game_context,
        require_same_step=require_same_step,
        require_no_intervening_action=require_no_intervening_action,
    )


def _change_score_for_speaker(change: Dict[str, str], speaker_role: str) -> int:
    true_role = change.get("true_role")
    before = change.get("before")
    after = change.get("after")
    if true_role not in {"Crewmate", "Impostor"}:
        return 0

    before_correct = before == true_role
    after_correct = after == true_role
    before_wrong = before in {"Crewmate", "Impostor"} and before != true_role
    after_wrong = after in {"Crewmate", "Impostor"} and after != true_role

    if speaker_role == "Crewmate":
        if not before_correct and after_correct:
            return 1
        if before_correct and not after_correct:
            return -1
        if not before_wrong and after_wrong:
            return -1
        return 0

    if speaker_role == "Impostor":
        if before_correct and not after_correct:
            return 1
        if not before_correct and after_correct:
            return -1
        if not before_wrong and after_wrong:
            return 1
        return 0

    return 0


def _score_direction(score: int) -> str:
    if score > 0:
        return "favorable"
    if score < 0:
        return "unfavorable"
    return "neutral"


def speech_belief_shift_effect(
    entry: Dict[str, Any],
    game_context: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Summarize whether a speech moves listener beliefs toward the speaker's
    role objective.

    For Crewmate speakers, favorable means Crewmate listeners become more
    accurate about true roles. For Impostor speakers, favorable means Crewmate
    listeners become less accurate or move into a wrong decided role.
    """

    speaker_role = base_utils.get_player_role(entry)
    event_id = speech_event_id(entry)
    transitions = listener_belief_transitions_after_speech(entry, game_context)
    favorable = 0
    unfavorable = 0
    neutral = 0
    changed_predictions = 0
    listener_deltas: List[float] = []
    annotated = []

    for transition in transitions:
        annotated_changes = []
        scores = []
        for change in transition["changes"]:
            changed_predictions += 1
            score = _change_score_for_speaker(change, speaker_role)
            direction = _score_direction(score)
            if direction == "favorable":
                favorable += 1
            elif direction == "unfavorable":
                unfavorable += 1
            else:
                neutral += 1
            scores.append(score)
            annotated_changes.append({**change, "score": score, "direction": direction})
        listener_delta = sum(scores) / len(scores) if scores else 0.0
        listener_deltas.append(listener_delta)
        annotated.append({**transition, "listener_delta": listener_delta, "changes": annotated_changes})

    aggregate_shift: Optional[float]
    if listener_deltas:
        aggregate_shift = sum(listener_deltas) / len(listener_deltas)
    else:
        aggregate_shift = None

    label: Optional[bool]
    reason: str
    if aggregate_shift is None:
        label = None
        reason = "no_paired_listener_belief_measurement" if event_id else "no_measured_listener_belief_shift"
    elif aggregate_shift > 0:
        label = True
        reason = "listener_belief_shift_favorable"
    else:
        label = False
        reason = "listener_belief_shift_not_favorable"

    return {
        "label": label,
        "reason": reason,
        "speech_event_id": event_id,
        "pairing_mode": transitions[0].get("pairing_mode") if transitions else ("speech_event_id" if event_id else "legacy_order"),
        "measured_listeners": len(transitions),
        "changed_predictions": changed_predictions,
        "aggregate_shift": aggregate_shift,
        "favorable": favorable,
        "unfavorable": unfavorable,
        "neutral": neutral,
        "transitions": annotated,
    }


def opponent_belief_transition_effect(
    entry: Dict[str, Any],
    game_context: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper for the updated speech belief-shift reward."""

    return speech_belief_shift_effect(entry, game_context)
