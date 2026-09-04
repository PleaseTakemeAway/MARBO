from typing import Any, Dict, List, Optional

from .base import RewardDecision, RewardFunction
from .meeting_speech_reward import BeliefMeetingSpeechReward
from .meeting_vote_reward import BeliefMeetingVoteReward
from .task_phase_reward import BeliefTaskPhaseReward
from . import utils


class BeliefAwareGameReward(RewardFunction):
    """
    Reward dispatcher for KTO labels on belief-enabled Among Us agent logs.

    This uses the current categorical Relational Belief:

        Role in {Crewmate, Impostor, unknown}

    The reward is intentionally conservative. It combines hard observable game
    rules with actor belief-action consistency and skips ambiguous strategic
    transition cases.
    """

    def __init__(
        self,
        task_reward: Optional[BeliefTaskPhaseReward] = None,
        vote_reward: Optional[BeliefMeetingVoteReward] = None,
        speech_reward: Optional[BeliefMeetingSpeechReward] = None,
    ) -> None:
        self.task_reward = task_reward or BeliefTaskPhaseReward()
        self.vote_reward = vote_reward or BeliefMeetingVoteReward()
        self.speech_reward = speech_reward or BeliefMeetingSpeechReward()

    def explain(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]] = None,
    ) -> RewardDecision:
        if not utils.is_clean_belief_action_entry(entry):
            return RewardDecision(None, "dispatcher", "not_clean_belief_action_entry")

        phase = utils.get_phase(entry)
        if phase == "Task phase":
            return self.task_reward.explain(entry, game_context=game_context)

        if phase == "Meeting phase":
            vote_decision = self.vote_reward.explain(entry, game_context=game_context)
            if vote_decision.label is not None:
                return vote_decision
            return self.speech_reward.explain(entry, game_context=game_context)

        return RewardDecision(None, "dispatcher", "unknown_phase")

    def label(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[bool]:
        return self.explain(entry, game_context).label


# Compatibility alias for code that expects GameOutcomeReward.
GameOutcomeReward = BeliefAwareGameReward
