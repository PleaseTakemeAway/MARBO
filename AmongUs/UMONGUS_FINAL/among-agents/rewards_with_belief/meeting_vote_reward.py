from typing import Any, Dict, List, Optional

from .base import RewardDecision
from . import utils


class BeliefMeetingVoteReward:
    """Meeting vote labels using true role plus actor categorical belief."""

    def explain(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]] = None,
    ) -> RewardDecision:
        if utils.get_phase(entry) != "Meeting phase":
            return RewardDecision(None, "meeting_vote", "not_meeting_phase")

        action = utils.extract_action(entry)
        if utils.action_category(action) != "VOTE":
            return RewardDecision(None, "meeting_vote", "not_vote")

        actor = utils.get_player_name(entry)
        voter_role = utils.get_player_role(entry)
        target = utils.extract_action_target(action)
        target_role = utils.role_of(target, utils.build_role_map(game_context))
        belief_role = utils.belief_role_for_player(utils.actor_belief(entry, game_context), target)
        believed_impostors = utils.believed_impostor_targets(entry, game_context)

        details = {
            "action": action,
            "target": target,
            "target_role": target_role,
            "belief_role_for_target": belief_role,
            "believed_impostors": believed_impostors,
        }

        if target and utils.player_number(target) == utils.player_number(actor):
            return RewardDecision(False, "meeting_vote", "self_vote", details)

        if target_role not in {"Crewmate", "Impostor"}:
            return RewardDecision(None, "meeting_vote", "unknown_target_role", details)

        if voter_role == "Crewmate":
            if target_role == "Crewmate":
                return RewardDecision(False, "meeting_vote", "crewmate_voted_crewmate", details)
            if belief_role == "Impostor":
                return RewardDecision(True, "meeting_vote", "crewmate_voted_true_impostor_consistent_with_belief", details)
            if believed_impostors:
                return RewardDecision(False, "meeting_vote", "crewmate_ignored_believed_impostor_vote_target", details)
            return RewardDecision(None, "meeting_vote", "crewmate_voted_true_impostor_without_belief_support", details)

        if voter_role == "Impostor":
            if target_role == "Crewmate":
                return RewardDecision(True, "meeting_vote", "impostor_voted_crewmate", details)
            return RewardDecision(None, "meeting_vote", "impostor_voted_teammate_distancing_ambiguous", details)

        return RewardDecision(None, "meeting_vote", "unknown_voter_role", details)

    def label(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[bool]:
        return self.explain(entry, game_context).label
