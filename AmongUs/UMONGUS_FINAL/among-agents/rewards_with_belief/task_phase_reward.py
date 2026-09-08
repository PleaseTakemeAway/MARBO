from typing import Any, Dict, List, Optional

from .base import RewardDecision
from .hard_task_phase_reward import TaskPhaseReward


class BeliefTaskPhaseReward:
    """
    Task-phase labels for belief-enabled agents.

    v1 keeps the hard observable rules used by the no-belief baseline. Movement,
    investigation, camouflage, and belief-transition effects remain skipped
    unless the observable hard rule can label them.
    """

    def __init__(self, hard_reward: Optional[TaskPhaseReward] = None) -> None:
        self.hard_reward = hard_reward or TaskPhaseReward()

    def explain(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]] = None,
    ) -> RewardDecision:
        label = self.hard_reward.label(entry, game_context=game_context)
        if label is None:
            return RewardDecision(None, "task_phase", "hard_rule_no_label")
        return RewardDecision(label, "task_phase", "hard_observable_rule")

    def label(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[bool]:
        return self.explain(entry, game_context).label
