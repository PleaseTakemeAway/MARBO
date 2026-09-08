import re
from typing import Any, Dict, List, Optional

from . import utils


ROOM_NAMES = (
    "Cafeteria",
    "Weapons",
    "Navigation",
    "O2",
    "Shields",
    "Communications",
    "Storage",
    "Admin",
    "Electrical",
    "Lower Engine",
    "Security",
    "Reactor",
    "Upper Engine",
    "Medbay",
)


class FactVerifier:
    """
    Conservative verifier for explicit conflicts with prompt-visible facts.

    It only returns False for direct contradictions. Otherwise it returns True,
    meaning no conflict was detected by these rules.
    """

    def label(self, entry: Dict[str, Any], speech: str) -> Optional[bool]:
        if not speech:
            return None

        if self._conflicts_with_current_location(entry, speech):
            return False
        if self._conflicts_with_task_status(entry, speech):
            return False
        if self._conflicts_with_death_state(entry, speech):
            return False
        if self._unsupported_direct_witness_claim(entry, speech):
            return False
        return True

    def _conflicts_with_current_location(self, entry: Dict[str, Any], speech: str) -> bool:
        current = utils.current_location(utils.get_all_info(entry))
        if not current:
            return False
        for room in ROOM_NAMES:
            patterns = (
                rf"\bI(?: am|'m)\s+(?:currently\s+)?(?:in|at)\s+(?:the\s+)?{re.escape(room)}\s+right now\b",
                rf"\bcurrently\s+(?:in|at)\s+(?:the\s+)?{re.escape(room)}\b",
            )
            if any(re.search(pattern, speech, re.IGNORECASE) for pattern in patterns):
                return room != current
        return False

    def _conflicts_with_task_status(self, entry: Dict[str, Any], speech: str) -> bool:
        lowered = speech.lower()
        if "completed" not in lowered and "finished" not in lowered:
            return False

        for task in utils.parse_tasks(utils.get_all_info(entry)):
            task_name = task.name.lower()
            if task_name and task_name in lowered and not task.completed:
                completion_claim = re.search(
                    rf"\b(?:completed|finished|did)\b[^.?!\n]*{re.escape(task_name)}",
                    lowered,
                )
                if completion_claim:
                    return True
        return False

    def _conflicts_with_death_state(self, entry: Dict[str, Any], speech: str) -> bool:
        lowered = speech.lower()
        claims_no_death = any(
            phrase in lowered
            for phrase in (
                "no one died",
                "nobody died",
                "no deaths",
                "no dead bodies",
            )
        )
        if not claims_no_death:
            return False

        visible_facts = "\n".join(
            [
                utils.get_all_info(entry),
                utils.memory_text(entry),
            ]
        ).lower()
        return "(dead)" in visible_facts or "dead body" in visible_facts or "was voted out" in visible_facts

    def _unsupported_direct_witness_claim(self, entry: Dict[str, Any], speech: str) -> bool:
        lowered = speech.lower()
        direct_claim = re.search(r"\b(?:i saw|i witnessed)\b[^.?!\n]*(?:kill|vent)", lowered)
        if not direct_claim:
            return False

        evidence_text = "\n".join(
            [
                utils.observation_block(utils.get_all_info(entry)),
                utils.action_history_block(utils.get_all_info(entry)),
                utils.memory_text(entry),
            ]
        ).lower()
        return "kill" not in evidence_text and "vent" not in evidence_text


class MeetingSpeechReward:
    """Staged-voting plus fact-verifier labels for meeting speech."""

    def __init__(self, use_fact_verifier: bool = True, fact_verifier: Optional[Any] = None) -> None:
        if fact_verifier is not None:
            self.fact_verifier = fact_verifier
        else:
            self.fact_verifier = FactVerifier() if use_fact_verifier else None

    def label(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[bool]:
        if utils.get_phase(entry) != "Meeting phase":
            return None

        action = utils.extract_action(entry)
        if utils.action_category(action) != "SPEAK":
            return None

        speech = utils.speech_text(action)
        fact_label = self.fact_verifier.label(entry, speech) if self.fact_verifier else None
        if fact_label is False:
            return False

        staged_label = self._staged_voting_label(entry, game_context)
        if staged_label is not None:
            return staged_label

        return fact_label

    def _staged_voting_label(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]],
    ) -> Optional[bool]:
        role_map = utils.build_role_map(game_context)
        outcome = utils.meeting_votes(entry, game_context, role_map)
        if outcome.total_votes == 0:
            return None

        speaker = utils.get_player_name(entry)
        speaker_role = utils.get_player_role(entry)
        speaker_votes = outcome.vote_counts.get(speaker, 0)
        speaker_has_majority = speaker_votes > outcome.total_votes / 2

        if speaker_role == "Impostor":
            if outcome.ejected_player == speaker or speaker_has_majority:
                return False
            if outcome.ejected_role == "Impostor":
                return False
            if outcome.ejected_role == "Crewmate":
                return True
            return True

        if speaker_role == "Crewmate":
            if outcome.ejected_role == "Crewmate":
                return False
            if speaker_has_majority:
                return False
            if outcome.ejected_role == "Impostor":
                return True
            if speaker_votes == 0:
                return True

        return None
