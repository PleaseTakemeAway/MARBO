from typing import Any, Dict, List, Optional, Protocol

from .base import RewardDecision
from .fact_verifier import FactVerifier
from . import utils


class FactVerifierLike(Protocol):
    def label(self, entry: Dict[str, Any], speech: str) -> Optional[bool]:
        ...


class BeliefMeetingSpeechReward:
    """
    Speech labels split into validity and listener belief-shift feedback.

    A speech is negative if it conflicts with visible facts. Otherwise, logged
    listener belief updates immediately before/after the speech are aggregated.
    Missing belief-shift measurements are skipped by default.
    """

    def __init__(
        self,
        use_fact_verifier: bool = True,
        fact_verifier: Optional[FactVerifierLike] = None,
        use_accusation_fallback: bool = False,
    ) -> None:
        if fact_verifier is not None:
            self.fact_verifier = fact_verifier
        else:
            self.fact_verifier = FactVerifier() if use_fact_verifier else None
        self.use_accusation_fallback = use_accusation_fallback

    def explain(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]] = None,
    ) -> RewardDecision:
        if utils.get_phase(entry) != "Meeting phase":
            return RewardDecision(None, "meeting_speech", "not_meeting_phase")

        action = utils.extract_action(entry)
        if utils.action_category(action) != "SPEAK":
            return RewardDecision(None, "meeting_speech", "not_speech")

        speech = utils.speech_text(action)
        fact_label = self.fact_verifier.label(entry, speech) if self.fact_verifier else None
        fact_details = getattr(self.fact_verifier, "last_result", None) if self.fact_verifier else None
        if fact_label is False:
            return RewardDecision(
                False,
                "meeting_speech",
                "fact_contradiction",
                {"speech": speech, "fact_verifier": fact_details},
            )

        shift_effect = utils.speech_belief_shift_effect(entry, game_context)
        if shift_effect["label"] is not None:
            return RewardDecision(
                shift_effect["label"],
                "meeting_speech",
                shift_effect["reason"],
                {
                    "speech": speech,
                    "fact_label": fact_label,
                    "fact_verifier": fact_details,
                    "listener_belief_shift": shift_effect,
                },
            )

        if not self.use_accusation_fallback:
            return RewardDecision(
                None,
                "meeting_speech",
                shift_effect["reason"],
                {
                    "speech": speech,
                    "fact_label": fact_label,
                    "fact_verifier": fact_details,
                    "listener_belief_shift": shift_effect,
                },
            )

        role_map = utils.build_role_map(game_context)
        beliefs = utils.actor_belief(entry, game_context)
        speaker = utils.get_player_name(entry)
        speaker_role = utils.get_player_role(entry)
        targets = utils.accusation_targets(speech, game_context)

        target_details = []
        for target in targets:
            target_details.append(
                {
                    "target": target,
                    "target_role": utils.role_of(target, role_map),
                    "belief_role": utils.belief_role_for_player(beliefs, target),
                }
            )

        details = {
            "speech": speech,
            "targets": target_details,
            "fact_label": fact_label,
            "fact_verifier": fact_details,
            "listener_belief_shift": shift_effect,
        }

        if not targets:
            return RewardDecision(None, "meeting_speech", "no_explicit_accusation", details)

        if speaker_role == "Crewmate":
            for target in targets:
                target_role = utils.role_of(target, role_map)
                belief_role = utils.belief_role_for_player(beliefs, target)
                if target_role == "Crewmate":
                    return RewardDecision(False, "meeting_speech", "crewmate_accused_true_crewmate", details)
                if target_role == "Impostor" and belief_role == "Impostor":
                    return RewardDecision(True, "meeting_speech", "crewmate_accused_believed_true_impostor", details)
            return RewardDecision(None, "meeting_speech", "crewmate_accusation_without_belief_support", details)

        if speaker_role == "Impostor":
            for target in targets:
                target_role = utils.role_of(target, role_map)
                if utils.player_number(target) == utils.player_number(speaker):
                    return RewardDecision(False, "meeting_speech", "impostor_self_accusation", details)
                if target_role == "Crewmate":
                    return RewardDecision(True, "meeting_speech", "impostor_accused_true_crewmate", details)
            return RewardDecision(None, "meeting_speech", "impostor_accused_teammate_or_unknown_target", details)

        return RewardDecision(None, "meeting_speech", "unknown_speaker_role", details)

    def label(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[bool]:
        return self.explain(entry, game_context).label
