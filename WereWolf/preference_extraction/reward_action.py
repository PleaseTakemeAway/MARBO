import argparse
import json
import os
import re
from collections import defaultdict

from MARBO.wolf.preference_extraction.reward import (
    DEFAULT_OUTPUT_FILE_NAMES,
    add_common_reward_args,
    build_model_from_args,
    build_night_self_role_map,
    build_night_state_by_day,
    normalize_role_label,
    phase_day,
    to_text,
)

_GOD_ROLES_SET = {"Seer", "Witch", "Guard", "Hunter"}
_GOOD_ROLES_SET = {"Villager", "Seer", "Witch", "Guard", "Hunter"}
_NO_TARGET_TOKENS = {"", "0", "no", "none", "null", "-1", "pass"}
_DEAD_MARKERS = ("dead", "deceased", "killed", "eliminated", "out")


def parse_args():
    parser = argparse.ArgumentParser(description="Export action reward good/bad phase maps.")
    add_common_reward_args(parser)
    parser.add_argument(
        "--good_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE_NAMES["action"]["good"],
        help="output file name for good action phases",
    )
    parser.add_argument(
        "--bad_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE_NAMES["action"]["bad"],
        help="output file name for bad action phases",
    )
    return parser.parse_args()


def _include_entry_by_gen_times(entry, max_gen_times: int = 3) -> bool:

    value = entry.get("gen_times")
    if value is None:
        return True
    try:
        return int(value) <= max_gen_times
    except (TypeError, ValueError):
        return True


def _check_recon_alignment(phase, target_str, reconstructed_role_label):

    if not isinstance(reconstructed_role_label, dict) or not target_str:
        return None

    predicted = normalize_role_label(reconstructed_role_label.get(str(target_str)))
    if predicted is None or predicted == "Unknown":
        return None

    if "night_skill_wolf" in phase:
        return predicted in _GOD_ROLES_SET
    elif "night_skill_seer" in phase:
        return predicted == "Werewolf"
    elif "night_skill_witch" in phase:
        return predicted == "Werewolf"
    elif "night_skill_guard" in phase:
        return predicted != "Werewolf"  # villager-side
    return None


def _normalize_target_id(raw_target):
    if raw_target is None:
        return None
    text = str(raw_target).strip()
    if not text:
        return None
    if text.lower() in _NO_TARGET_TOKENS:
        return None
    if text.isdigit():
        return text
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits if digits else text


def _target_role(role_mapping, target_id):
    if target_id is None:
        return None
    role = role_mapping.get(str(target_id), "")
    normalized = normalize_role_label(role)
    return normalized if normalized else None


def _load_action_response(entry):
    try:
        from MARBO.wolf.adversarial_data_extraction.utils_english import _parse_response_json
    except ModuleNotFoundError:
        from MARBO.wolf.adversarial_data_extraction.utils_english import _parse_response_json

    for field in ("response", "action"):
        text = to_text(entry.get(field)).strip()
        if not text:
            continue
        parsed = _parse_response_json(text)
        if isinstance(parsed, dict):
            return parsed
    return None


def _get_reconstructed_raw_label(reconstructed_role_label, target_id):
    if not isinstance(reconstructed_role_label, dict) or target_id is None:
        return None

    candidates = [str(target_id)]
    match = re.search(r"\d+", str(target_id))
    if match:
        pid = match.group()
        candidates.extend([pid, f"Player {pid}", f"Player {int(pid)}"])

    for key in candidates:
        if key in reconstructed_role_label:
            return reconstructed_role_label.get(key)
    return None


def _predicted_target_role(reconstructed_role_label, target_id):
    return normalize_role_label(_get_reconstructed_raw_label(reconstructed_role_label, target_id))


def _is_dead_like_label(raw_label):
    if raw_label is None:
        return False
    lowered = str(raw_label).strip().lower()
    return any(token in lowered for token in _DEAD_MARKERS)


def _target_marked_dead(reconstructed_role_label, target_id):
    return _is_dead_like_label(_get_reconstructed_raw_label(reconstructed_role_label, target_id))


def _iter_reconstructed_predictions(reconstructed_role_label):
    if not isinstance(reconstructed_role_label, dict):
        return
    for key, raw_value in reconstructed_role_label.items():
        match = re.search(r"\d+", str(key))
        if not match:
            continue
        pid = match.group()
        yield pid, raw_value, normalize_role_label(raw_value), _is_dead_like_label(raw_value)


def _has_live_predicted_role(reconstructed_role_label, target_role):
    for _, raw_value, normalized_role, is_dead in _iter_reconstructed_predictions(reconstructed_role_label):
        if is_dead:
            continue
        if normalized_role == target_role:
            return True
    return False


def _live_predicted_ids_by_role(reconstructed_role_label, target_role):
    ids = set()
    for pid, _, normalized_role, is_dead in _iter_reconstructed_predictions(reconstructed_role_label):
        if is_dead:
            continue
        if normalized_role == target_role:
            ids.add(str(pid))
    return ids


def build_night_outcome_by_day(game_path, role_mapping):
    game_log_file = os.path.join(game_path, "game_log.json")
    if not os.path.exists(game_log_file):
        return {}

    try:
        with open(game_log_file, "r", encoding="utf-8") as f:
            game_log = json.load(f)
    except Exception:
        return {}

    night_outcome_by_day = {}
    for event in game_log:
        if str(event.get("event", "")) != "end_night":
            continue
        try:
            day = int(event.get("day"))
        except Exception:
            continue

        content = event.get("content") or {}
        dead_list = content.get("dead_list", event.get("target", []))
        if dead_list is None:
            dead_list = []
        if not isinstance(dead_list, list):
            dead_list = [dead_list]

        dead_ids = []
        for raw_target in dead_list:
            normalized = _normalize_target_id(raw_target)
            if normalized is not None:
                dead_ids.append(normalized)

        dead_roles = [_target_role(role_mapping, pid) for pid in dead_ids]
        night_outcome_by_day[day] = {
            "dead_ids": dead_ids,
            "dead_roles": dead_roles,
            "no_one_died": len(dead_ids) == 0,
            "villager_dead": any(role in _GOOD_ROLES_SET for role in dead_roles),
        }

    return night_outcome_by_day


def _classify_seer_action(
    target,
    actual_target_role,
    predicted_target_role,
    target_is_dead,
    known_checked_targets=None,
):
    # Undesirable: no use / check dead player / re-check already known player /
    # check anyone other than Werewolf.
    if target is None or target_is_dead:
        return False, False
    if known_checked_targets is not None and str(target) in known_checked_targets:
        return False, False
    if actual_target_role is None:
        return None, None

    if actual_target_role != "Werewolf":
        return False, False

    if predicted_target_role == "Werewolf":
        process_label = True
    elif predicted_target_role and predicted_target_role != "Unknown":
        process_label = False
    else:
        process_label = None

    return True, process_label


def _classify_guard_action(
    target,
    player_id,
    day,
    actual_target_role,
    predicted_target_role,
    target_is_dead,
    night_outcome,
    reconstructed_role_label,
):
    del player_id
    del day
    del night_outcome

    # Desirable: protect any good-side role.
    # Undesirable: no use / protect Werewolf / invalid target.
    if target is None or target_is_dead:
        return False, False
    if actual_target_role is None:
        return None, None

    if actual_target_role == "Werewolf":
        return False, False

    if predicted_target_role and predicted_target_role != "Unknown":
        process_label = predicted_target_role != "Werewolf"
    else:
        process_label = None

    return True, process_label


def _classify_witch_action(
    response_json,
    actual_target_role,
    predicted_target_role,
    target_is_dead,
    reconstructed_role_label,
    day,
):
    heal_target = _normalize_target_id((response_json or {}).get("heal"))
    poison_target = _normalize_target_id((response_json or {}).get("poison"))
    first_night = day == 0

    # Night 1: desirable if heal used, undesirable if no heal.
    if first_night:
        heal_used = heal_target is not None
        return (True, True) if heal_used else (False, False)

    # Night >= 2: desirable if poison a Werewolf; otherwise undesirable.
    if poison_target is None:
        return False, False

    if target_is_dead:
        return False, False

    if actual_target_role is None:
        return None, None

    if predicted_target_role == "Werewolf":
        process_label = True
    elif predicted_target_role and predicted_target_role != "Unknown":
        process_label = False
    else:
        process_label = None

    if actual_target_role == "Werewolf" and process_label is True:
        return True, True
    if actual_target_role != "Werewolf":
        return False, False
    return True, process_label


def _classify_werewolf_action(target, actual_target_role, predicted_target_role, target_is_dead):
    """
    Werewolf rule (user override):
      - Desirable if killing any non-Werewolf.
      - Undesirable if killing ally / no kill.
    """
    # Undesirable: no kill / invalid target.
    if target is None or target_is_dead:
        return False, False

    if actual_target_role is None:
        return None, None

    if actual_target_role == "Werewolf":
        return False, False

    if predicted_target_role and predicted_target_role != "Unknown":
        process_label = predicted_target_role in _GOOD_ROLES_SET
    else:
        process_label = None

    return True, process_label


def label_action_entry(
    entry,
    role_mapping,
    player_id,           # reserved: kept for API parity with reward_recon.label_recon_entry
    reconstructed_role_label,
    self_reconstructed_role,  # reserved: kept for API parity
    game_type,
    night_outcome_by_day=None,
    known_checked_targets=None,
):
    """
    Backward-compatible wrapper:
      returns only final desirable/undesirable label.
    """
    outcome_label, _ = evaluate_action_entry(
        entry=entry,
        role_mapping=role_mapping,
        player_id=player_id,
        reconstructed_role_label=reconstructed_role_label,
        self_reconstructed_role=self_reconstructed_role,
        game_type=game_type,
        night_outcome_by_day=night_outcome_by_day,
        known_checked_targets=known_checked_targets,
    )
    return outcome_label


def evaluate_action_entry(
    entry,
    role_mapping,
    player_id,           # reserved: kept for API parity with reward_recon.label_recon_entry
    reconstructed_role_label,
    self_reconstructed_role,  # reserved: kept for API parity
    game_type,
    night_outcome_by_day=None,
    known_checked_targets=None,
):
    """
    Evaluate one night action entry and return:
      (final_label, process_label)

    final_label:
      True  → desirable by the role-specific reward table
      False → undesirable by the role-specific reward table
      None  → ambiguous / skipped

    process_label:
      Low-level belief-state/process alignment signal (True / False / None).
    """
    del self_reconstructed_role

    if not _include_entry_by_gen_times(entry):
        return None, None

    phase = str(entry.get("phase", ""))
    if "night_skill" not in phase:
        return None, None

    day = phase_day(phase)
    if day is None:
        return None, None

    try:
        from MARBO.wolf.adversarial_data_extraction.utils_english import extract_night_action_target, label_from_action_response
    except ModuleNotFoundError:
        from MARBO.wolf.adversarial_data_extraction.utils_english import extract_night_action_target, label_from_action_response

    response_json = _load_action_response(entry)
    if not isinstance(response_json, dict):
        return None, None

    response_text = ""
    for field in ("response", "action"):
        candidate = to_text(entry.get(field)).strip()
        if candidate:
            response_text = candidate
            break

    raw_target = extract_night_action_target(phase, response_text)
    if raw_target is None:
        if "night_skill_witch" in phase:
            raw_target = response_json.get("poison")
        elif "night_skill_guard" in phase:
            raw_target = response_json.get("protected_player", response_json.get("guard"))
        elif "night_skill_seer" in phase:
            raw_target = response_json.get("check")
        elif "night_skill_wolf" in phase:
            raw_target = response_json.get("kill")

    target = _normalize_target_id(raw_target)
    if target is not None and str(target) == str(player_id):
        return False, False

    actual_target_role = _target_role(role_mapping, target)
    predicted_target_role = _predicted_target_role(reconstructed_role_label, target)
    target_is_dead = _target_marked_dead(reconstructed_role_label, target)
    night_outcome = {}
    if isinstance(night_outcome_by_day, dict):
        night_outcome = night_outcome_by_day.get(day) or {}

    actor_role = normalize_role_label(role_mapping.get(str(player_id), ""))

    if "night_skill_seer" in phase or actor_role == "Seer":
        result = _classify_seer_action(
            target=target,
            actual_target_role=actual_target_role,
            predicted_target_role=predicted_target_role,
            target_is_dead=target_is_dead,
            known_checked_targets=known_checked_targets,
        )
        if known_checked_targets is not None and target is not None:
            known_checked_targets.add(str(target))
        return result

    if "night_skill_guard" in phase or actor_role == "Guard":
        return _classify_guard_action(
            target=target,
            player_id=player_id,
            day=day,
            actual_target_role=actual_target_role,
            predicted_target_role=predicted_target_role,
            target_is_dead=target_is_dead,
            night_outcome=night_outcome,
            reconstructed_role_label=reconstructed_role_label,
        )

    if "night_skill_witch" in phase or actor_role == "Witch":
        return _classify_witch_action(
            response_json=response_json,
            actual_target_role=actual_target_role,
            predicted_target_role=predicted_target_role,
            target_is_dead=target_is_dead,
            reconstructed_role_label=reconstructed_role_label,
            day=day,
        )

    if "night_skill_wolf" in phase or actor_role == "Werewolf":
        return _classify_werewolf_action(
            target=target,
            actual_target_role=actual_target_role,
            predicted_target_role=predicted_target_role,
            target_is_dead=target_is_dead,
        )

    for field in ("response", "action"):
        text = to_text(entry.get(field)).strip()
        if not text:
            continue

        # Hard rule: wolf team-kill should always be undesirable.
        if "night_skill_wolf" in phase:
            raw_target = extract_night_action_target(phase, text)
            target = _normalize_target_id(raw_target)
            if _target_role(role_mapping, target) == "Werewolf":
                recon_ok = None
                if day >= 1 and target is not None:
                    recon_ok = _check_recon_alignment(phase, target, reconstructed_role_label)
                return False, recon_ok

        outcome_label = label_from_action_response(phase, text, role_mapping, game_type)
        if outcome_label is None:
            continue
        outcome_label = bool(outcome_label)

        # state_recon is tracked separately and does not gate outcome label.
        recon_ok = None
        if day >= 1:
            target = _normalize_target_id(extract_night_action_target(phase, text))
            if target is not None:
                recon_ok = _check_recon_alignment(phase, target, reconstructed_role_label)

        return outcome_label, recon_ok

    return None, None


def split_action_samples(model, args):
    good_map = model.empty_phase_map()
    bad_map = model.empty_phase_map()
    state_recon_meta = defaultdict(lambda: defaultdict(dict))
    reason_meta = defaultdict(lambda: defaultdict(dict))

    for game_path, role_mapping, entries_by_player, werewolf_is_sft, good_is_sft in model.iter_game_contexts(
        args.game_dir,
        random_play=args.random_play,
        self_play=args.self_play,
        max_games=args.max_games,
    ):
        abs_game_path = os.path.abspath(game_path)
        night_outcome_by_day = build_night_outcome_by_day(abs_game_path, role_mapping)
        for player_id, entries in entries_by_player.items():
            player_role = role_mapping.get(str(player_id), "")
            if player_role == "Werewolf" and not werewolf_is_sft:
                continue
            if player_role != "Werewolf" and not good_is_sft:
                continue

            night_self_role_by_day = build_night_self_role_map(entries, player_id)
            night_state_by_day = build_night_state_by_day(entries)
            checked_targets_by_seer = set()

            for entry in entries:
                phase = str(entry.get("phase", ""))
                day = phase_day(phase)

                if "night_skill" not in phase:
                    continue

                label, recon_label = evaluate_action_entry(
                    entry=entry,
                    role_mapping=role_mapping,
                    player_id=player_id,
                    reconstructed_role_label=night_state_by_day.get(day),
                    self_reconstructed_role=night_self_role_by_day.get(day),
                    game_type=model.game_type,
                    night_outcome_by_day=night_outcome_by_day,
                    known_checked_targets=checked_targets_by_seer,
                )
                if label is None:
                    continue
                # Keep bad only when BOTH outcome(result) and process are False.
                if label is False and recon_label is not False:
                    continue

                state_recon_meta[abs_game_path][str(player_id)][phase] = recon_label
                reason_meta[abs_game_path][str(player_id)][phase] = {
                    "outcome": label,
                    "state_recon": recon_label,
                    "final": label,
                }

                if label:
                    model.add_phase_to_map(good_map, abs_game_path, player_id, phase)
                else:
                    model.add_phase_to_map(bad_map, abs_game_path, player_id, phase)

    return good_map, bad_map, state_recon_meta, reason_meta


def main():
    args = parse_args()
    model = build_model_from_args(args)
    good_map, bad_map, state_recon_meta, reason_meta = split_action_samples(model, args)
    out_to = os.path.abspath(args.out_to)
    saved = model.save_good_bad_pair(
        out_to=out_to,
        category="action",
        good_map=good_map,
        bad_map=bad_map,
        good_file_name=args.good_file,
        bad_file_name=args.bad_file,
    )

    meta_file = os.path.join(out_to, "action_state_recon_meta.json")
    meta_plain = {
        game_path: {pid: phases for pid, phases in by_player.items()}
        for game_path, by_player in state_recon_meta.items()
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_plain, f, indent=4, ensure_ascii=False)

    reason_file = os.path.join(out_to, "action_reason_meta.json")
    reason_plain = {
        game_path: {pid: phases for pid, phases in by_player.items()}
        for game_path, by_player in reason_meta.items()
    }
    with open(reason_file, "w", encoding="utf-8") as f:
        json.dump(reason_plain, f, indent=4, ensure_ascii=False)

    print(
        f"[action] good={saved.get('good_count', 0)} -> {saved.get('good_file')}, "
        f"bad={saved.get('bad_count', 0)} -> {saved.get('bad_file')}"
    )
    print(f"[action] state_recon meta -> {meta_file}")
    print(f"[action] reason meta -> {reason_file}")


if __name__ == "__main__":
    main()
