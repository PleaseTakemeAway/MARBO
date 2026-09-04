from typing import Any, Dict, List, Optional

from . import utils


class TaskPhaseReward:
    """
    Heuristic labels for task-phase actions in non-belief Among Us logs.

    The rules intentionally avoid weak movement labels. Movement is only labeled
    when it follows a prior kill for an Impostor, or when it is part of a repeated
    clear avoidance pattern.
    """

    def __init__(
        self,
        penalize_repeated_task_ignore: bool = True,
        penalize_repeated_safe_kill_avoidance: bool = True,
    ) -> None:
        self.penalize_repeated_task_ignore = penalize_repeated_task_ignore
        self.penalize_repeated_safe_kill_avoidance = penalize_repeated_safe_kill_avoidance

    def label(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[bool]:
        if utils.get_phase(entry) != "Task phase":
            return None

        action = utils.extract_action(entry)
        category = utils.action_category(action)
        if category is None:
            return None
        if category == "SPEAK":
            return None

        role = utils.get_player_role(entry)
        role_map = utils.build_role_map(game_context)

        if role == "Crewmate":
            return self._label_crewmate(entry, game_context, category)
        if role == "Impostor":
            return self._label_impostor(entry, game_context, category, role_map)
        return None

    def _label_crewmate(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]],
        category: str,
    ) -> Optional[bool]:
        if category == "COMPLETE TASK":
            return True

        if category == "REPORT DEAD BODY":
            return True

        if category == "CALL MEETING":
            if utils.has_observed_hard_evidence(entry):
                return True
            # Cafeteria dead markers can be ejected players, so do not use them
            # as positive body evidence. No evidence means this is an emergency
            # button call without support from the observable prompt.
            return False

        if self._ignored_visible_body(entry, category):
            return False

        if self.penalize_repeated_task_ignore and self._repeatedly_ignored_available_task(entry, game_context, category):
            return False

        return None

    def _label_impostor(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]],
        category: str,
        role_map: Dict[str, Optional[str]],
    ) -> Optional[bool]:
        action = utils.extract_action(entry)

        if category == "KILL":
            target = utils.extract_action_target(action)
            if utils.role_of(target, role_map) != "Crewmate":
                return None
            witness_count = utils.kill_witness_count(entry, role_map, target)
            if witness_count is None:
                return None
            return witness_count == 0

        if category == "VENT":
            if self._vented_in_front_of_crewmate(entry, role_map):
                return False
            return None

        if category in {"MOVE", "VENT"} and self._moved_after_own_kill(entry, game_context):
            return True

        if self.penalize_repeated_safe_kill_avoidance and self._repeatedly_avoided_safe_kill(
            entry,
            game_context,
            category,
            role_map,
        ):
            return False

        return None

    def _ignored_visible_body(self, entry: Dict[str, Any], category: str) -> bool:
        if category in {"CALL MEETING", "REPORT DEAD BODY"}:
            return False
        if utils.current_location(utils.get_all_info(entry)) == "Cafeteria":
            return False
        return utils.has_visible_dead_in_current_room(entry) and utils.has_available_action(
            utils.get_all_info(entry),
            "REPORT DEAD BODY",
        )

    def _repeatedly_ignored_available_task(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]],
        category: str,
    ) -> bool:
        if category == "COMPLETE TASK" or utils.has_observed_hard_evidence(entry):
            return False
        if not utils.has_available_action(utils.get_all_info(entry), "COMPLETE TASK"):
            return False

        def previous_task_ignore(item: Dict[str, Any]) -> bool:
            if utils.get_phase(item) != "Task phase" or utils.get_player_role(item) != "Crewmate":
                return False
            previous_category = utils.action_category(utils.extract_action(item))
            return (
                previous_category != "COMPLETE TASK"
                and utils.has_available_action(utils.get_all_info(item), "COMPLETE TASK")
                and not utils.has_observed_hard_evidence(item)
            )

        return utils.previous_repeated_condition_count(entry, game_context, previous_task_ignore) >= 1

    def _vented_in_front_of_crewmate(
        self,
        entry: Dict[str, Any],
        role_map: Dict[str, Optional[str]],
    ) -> bool:
        actor = utils.get_player_name(entry)
        for player in utils.current_room_players(utils.get_all_info(entry)):
            if player.dead or player.name == actor:
                continue
            if utils.role_of(player.name, role_map) == "Crewmate":
                return True
        return False

    def _moved_after_own_kill(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]],
    ) -> bool:
        previous = utils.previous_player_action(entry, game_context)
        if not previous:
            return False
        return utils.action_category(utils.extract_action(previous)) == "KILL"

    def _repeatedly_avoided_safe_kill(
        self,
        entry: Dict[str, Any],
        game_context: Optional[List[Dict[str, Any]]],
        category: str,
        role_map: Dict[str, Optional[str]],
    ) -> bool:
        if category == "KILL":
            return False
        if not utils.safe_kill_available(entry, role_map):
            return False

        def previous_safe_non_kill(item: Dict[str, Any]) -> bool:
            if utils.get_phase(item) != "Task phase" or utils.get_player_role(item) != "Impostor":
                return False
            previous_category = utils.action_category(utils.extract_action(item))
            return previous_category != "KILL" and utils.safe_kill_available(item, role_map)

        return utils.previous_repeated_condition_count(entry, game_context, previous_safe_non_kill) >= 1
