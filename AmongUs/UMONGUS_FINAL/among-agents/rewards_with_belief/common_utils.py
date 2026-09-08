import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


KNOWN_ACTION_PREFIXES = (
    "COMPLETE FAKE TASK",
    "COMPLETE TASK",
    "REPORT DEAD BODY",
    "CALL MEETING",
    "VIEW MONITOR",
    "KILL",
    "VENT",
    "MOVE",
    "VOTE",
    "SPEAK",
)


PLAYER_RE = re.compile(r"Player\s+\d+:\s+[A-Za-z]+")
RELATIONAL_BELIEF_ENABLED_KEYS = ("Relational Belief Enabled", "Belief State Enabled")
PREVIOUS_RELATIONAL_BELIEF_KEYS = ("Previous Relational Belief", "Previous Belief State")


@dataclass(frozen=True)
class PlayerRef:
    name: str
    dead: bool = False


@dataclass(frozen=True)
class TaskInfo:
    name: str
    completed: bool
    path: str


@dataclass(frozen=True)
class MeetingOutcome:
    vote_counts: Counter
    total_votes: int
    ejected_player: Optional[str]
    ejected_role: Optional[str]


def get_prompt(entry: Dict[str, Any]) -> Dict[str, Any]:
    prompt = entry.get("interaction", {}).get("prompt", {})
    return prompt if isinstance(prompt, dict) else {}


def get_all_info(entry: Dict[str, Any]) -> str:
    return str(get_prompt(entry).get("All Info", ""))


def get_phase(entry: Dict[str, Any]) -> str:
    return str(get_prompt(entry).get("Phase", "")).strip()


def get_player_name(entry: Dict[str, Any]) -> str:
    return str(entry.get("player", {}).get("name", "")).strip()


def get_player_role(entry: Dict[str, Any]) -> str:
    return str(entry.get("player", {}).get("identity", "")).strip()


def get_model(entry: Dict[str, Any]) -> str:
    return str(entry.get("player", {}).get("model", "")).strip()


def is_belief_enabled_entry(entry: Dict[str, Any]) -> bool:
    prompt = get_prompt(entry)
    player = entry.get("player", {})
    return any(bool(prompt.get(key)) for key in RELATIONAL_BELIEF_ENABLED_KEYS) or player.get("belief_state") is True


def is_fallback_or_random_entry(entry: Dict[str, Any]) -> bool:
    """
    Exclude completions where the saved action was not selected by the model.

    RandomAgent usually does not write compact LLM logs, but deterministic repair
    fallback and human/manual actions can appear in the same format.
    """
    prompt = get_prompt(entry)
    full_response = str(entry.get("interaction", {}).get("full_response", ""))
    model = get_model(entry).lower()
    system_prompt = str(entry.get("interaction", {}).get("system_prompt", "")).lower()

    if "Action Repair Fallback" in prompt:
        return True
    if "deterministic fallback after invalid action repair attempts" in full_response.lower():
        return True
    if "homosapiens" in model or "human agent" in system_prompt:
        return True
    if model in {"random", "randomagent", "random-agent"}:
        return True
    return False


def is_action_entry(entry: Dict[str, Any]) -> bool:
    prompt = get_prompt(entry)
    if not prompt:
        return False
    if any(key in prompt for key in PREVIOUS_RELATIONAL_BELIEF_KEYS) or "Previous Event Memory" in prompt:
        return False
    if not prompt.get("Phase"):
        return False
    return bool(extract_action(entry))


def is_clean_non_belief_entry(entry: Dict[str, Any]) -> bool:
    return (
        not is_belief_enabled_entry(entry)
        and not is_fallback_or_random_entry(entry)
        and is_action_entry(entry)
    )


def extract_action(entry: Dict[str, Any]) -> str:
    raw_action = _extract_raw_action(entry)
    if not raw_action:
        return ""
    return normalize_action(raw_action, entry)


def _extract_raw_action(entry: Dict[str, Any]) -> str:
    response = entry.get("interaction", {}).get("response")
    if isinstance(response, dict):
        action = response.get("Action")
        if isinstance(action, str) and action.strip():
            return first_line(action.strip())

        thinking_process = response.get("Thinking Process")
        if isinstance(thinking_process, dict):
            action = thinking_process.get("action")
            if isinstance(action, str) and action.strip():
                return first_line(action.strip())

    full_response = str(entry.get("interaction", {}).get("full_response", ""))
    match = re.search(r"\[Action\]\s*(.*)", full_response, re.DOTALL)
    if match:
        return first_line(match.group(1).strip())
    return ""


def normalize_action(raw_action: str, entry: Dict[str, Any]) -> str:
    """
    Recover the action that the environment most likely executed.

    LLMAgent matches non-SPEAK actions by checking whether an exact available
    action string appears anywhere in the model's [Action] text. Some logs
    therefore contain malformed text such as:

        SPEAK: ... MOVE from Cafeteria to Admin

    even though the executed action was the embedded MOVE. Reproduce that match
    behavior here so rewards are assigned to the final valid action.
    """
    raw_action = first_line(raw_action)
    available = available_actions(get_all_info(entry))

    for action in available:
        if action.startswith("SPEAK:"):
            continue
        if action in raw_action:
            return action

    return raw_action


def first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text else ""


def action_category(action: str) -> Optional[str]:
    for prefix in KNOWN_ACTION_PREFIXES:
        if action.startswith(prefix):
            return prefix
    return None


def extract_player_name(text: str) -> Optional[str]:
    match = PLAYER_RE.search(text or "")
    return match.group(0).strip() if match else None


def extract_action_target(action: str) -> Optional[str]:
    return extract_player_name(action)


def current_location(all_info: str) -> str:
    match = re.search(r"Current Location:\s*(.*)", all_info)
    return match.group(1).strip() if match else ""


def current_room_players(all_info: str) -> List[PlayerRef]:
    match = re.search(r"Players in [^:]+:\s*(.*?)(?:\n\n|$)", all_info, re.DOTALL)
    if not match:
        return []

    players: List[PlayerRef] = []
    for raw in match.group(1).replace("\n", " ").split(","):
        raw = raw.strip()
        if not raw:
            continue
        dead = "(dead)" in raw
        clean = raw.replace("(dead)", "").strip()
        name = extract_player_name(clean)
        if name:
            players.append(PlayerRef(name=name, dead=dead))
    return players


def available_actions(all_info: str) -> List[str]:
    if "Available actions:" not in all_info:
        return []
    block = all_info.split("Available actions:", 1)[1]
    actions: List[str] = []
    for line in block.splitlines():
        match = re.match(r"\s*\d+\.\s*(.+?)\s*$", line)
        if match:
            actions.append(match.group(1).strip())
    return actions


def has_available_action(all_info: str, prefix: str) -> bool:
    return any(action.startswith(prefix) for action in available_actions(all_info))


def parse_tasks(all_info: str) -> List[TaskInfo]:
    if "Your Assigned Tasks:" not in all_info or "Available actions:" not in all_info:
        return []
    block = all_info.split("Your Assigned Tasks:", 1)[1].split("Available actions:", 1)[0]
    pattern = re.compile(
        r"\d+\.\s*[^:]+:\s*(?P<name>.*?)\s*(?P<done>\[completed\])?\s*\n"
        r"Path:\s*(?P<path>.*?)(?=\n\d+\.|\Z)",
        re.DOTALL,
    )
    tasks: List[TaskInfo] = []
    for match in pattern.finditer(block):
        tasks.append(
            TaskInfo(
                name=match.group("name").strip(),
                completed=bool(match.group("done")),
                path=" ".join(match.group("path").strip().split()),
            )
        )
    return tasks


def observation_block(all_info: str) -> str:
    match = re.search(r"Observation history:\n(.*?)\n\nAction history:", all_info, re.DOTALL)
    return match.group(1).strip() if match else ""


def action_history_block(all_info: str) -> str:
    match = re.search(r"Action history:\n(.*?)\n\nYour Assigned Tasks:", all_info, re.DOTALL)
    return match.group(1).strip() if match else ""


def memory_text(entry: Dict[str, Any]) -> str:
    return str(get_prompt(entry).get("Memory", ""))


def has_observed_hard_evidence(entry: Dict[str, Any]) -> bool:
    text = "\n".join(
        [
            observation_block(get_all_info(entry)),
            action_history_block(get_all_info(entry)),
            memory_text(entry),
        ]
    )
    evidence_terms = (
        " KILL ",
        "KILL Player",
        " VENT ",
        "VENT from",
        "dead body",
        "reported",
        "REPORT DEAD BODY",
    )
    return any(term in text for term in evidence_terms)


def has_visible_dead_in_current_room(entry: Dict[str, Any]) -> bool:
    return any(player.dead for player in current_room_players(get_all_info(entry)))


def build_role_map(game_context: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, Optional[str]]:
    roles: Dict[str, Optional[str]] = {}
    if not game_context:
        return roles

    for item in game_context:
        name = get_player_name(item)
        role = get_player_role(item)
        if not name or role not in {"Crewmate", "Impostor"}:
            continue
        if name in roles and roles[name] != role:
            roles[name] = None
        else:
            roles[name] = role
    return roles


def role_of(name: Optional[str], role_map: Dict[str, Optional[str]]) -> Optional[str]:
    if not name:
        return None
    return role_map.get(name)


def sorted_game_entries(game_context: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not game_context:
        return []
    return sorted(
        game_context,
        key=lambda item: (
            item.get("step", -1),
            str(item.get("timestamp", "")),
            get_player_name(item),
        ),
    )


def previous_player_action(
    entry: Dict[str, Any],
    game_context: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    current_name = get_player_name(entry)
    ordered = sorted_game_entries(game_context)
    previous: Optional[Dict[str, Any]] = None
    for item in ordered:
        if item is entry:
            return previous
        if item.get("timestamp") == entry.get("timestamp") and item.get("step") == entry.get("step"):
            return previous
        if get_player_name(item) == current_name and is_action_entry(item):
            previous = item
    return previous


def kill_witness_count(
    entry: Dict[str, Any],
    role_map: Dict[str, Optional[str]],
    target_name: Optional[str],
) -> Optional[int]:
    if not target_name:
        return None

    actor = get_player_name(entry)
    witnesses = 0
    for player in current_room_players(get_all_info(entry)):
        if player.dead or player.name in {actor, target_name}:
            continue
        if role_of(player.name, role_map) == "Crewmate":
            witnesses += 1
    return witnesses


def safe_kill_available(
    entry: Dict[str, Any],
    role_map: Dict[str, Optional[str]],
) -> bool:
    all_info = get_all_info(entry)
    actor = get_player_name(entry)
    for available in available_actions(all_info):
        if not available.startswith("KILL"):
            continue
        target = extract_action_target(available)
        if role_of(target, role_map) != "Crewmate":
            continue
        witnesses = 0
        for player in current_room_players(all_info):
            if player.dead or player.name in {actor, target}:
                continue
            if role_of(player.name, role_map) == "Crewmate":
                witnesses += 1
        if witnesses == 0:
            return True
    return False


def previous_repeated_condition_count(
    entry: Dict[str, Any],
    game_context: Optional[List[Dict[str, Any]]],
    predicate,
) -> int:
    count = 0
    for item in sorted_game_entries(game_context):
        if item is entry:
            break
        if item.get("timestamp") == entry.get("timestamp") and item.get("step") == entry.get("step"):
            break
        if get_player_name(item) == get_player_name(entry) and predicate(item):
            count += 1
    return count


def meeting_votes(
    entry: Dict[str, Any],
    game_context: Optional[List[Dict[str, Any]]],
    role_map: Dict[str, Optional[str]],
) -> MeetingOutcome:
    step = entry.get("step")
    vote_counts: Counter = Counter()

    for item in game_context or []:
        if get_phase(item) != "Meeting phase" or item.get("step") != step:
            continue
        action = extract_action(item)
        if not action.startswith("VOTE"):
            continue
        target = extract_action_target(action)
        if target:
            vote_counts[target] += 1

    if not vote_counts:
        return MeetingOutcome(vote_counts=vote_counts, total_votes=0, ejected_player=None, ejected_role=None)

    max_votes = max(vote_counts.values())
    top_targets = [target for target, votes in vote_counts.items() if votes == max_votes]
    if len(top_targets) != 1:
        return MeetingOutcome(vote_counts=vote_counts, total_votes=sum(vote_counts.values()), ejected_player=None, ejected_role=None)

    ejected = top_targets[0]
    return MeetingOutcome(
        vote_counts=vote_counts,
        total_votes=sum(vote_counts.values()),
        ejected_player=ejected,
        ejected_role=role_of(ejected, role_map),
    )


def speech_text(action: str) -> str:
    if action.startswith("SPEAK:"):
        return action.split("SPEAK:", 1)[1].strip()
    if action.startswith("SPEAK"):
        return action[len("SPEAK") :].lstrip(": ").strip()
    return ""
