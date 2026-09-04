import argparse
import json
import os
import re
from collections import Counter, defaultdict

from MARBO.wolf.preference_extraction.reward import (
    DEFAULT_OUTPUT_FILE_NAMES,
    GOOD_ROLES,
    add_common_reward_args,
    build_model_from_args,
    normalize_role_label,
    parse_dict_maybe,
    phase_day,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export speech reward good/bad phase maps.")
    add_common_reward_args(parser)
    parser.add_argument(
        "--good_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE_NAMES["speech"]["good"],
        help="output file name for good speech phases",
    )
    parser.add_argument(
        "--bad_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE_NAMES["speech"]["bad"],
        help="output file name for bad speech phases",
    )
    return parser.parse_args()


IDENTITY_KEYS = [
    "Identity to Present",
    "identity to present",
    "identity_to_present",
    "identityToPresent",
    "self_present",
]


def _team_from_role(role):
    normalized = normalize_role_label(role)
    if normalized == "Werewolf":
        return "werewolf"
    if normalized in GOOD_ROLES:
        return "good"
    return None


def _normalize_role_mapping(role_mapping):
    normalized_map = {}
    for pid_raw, role in (role_mapping or {}).items():
        match = re.search(r"\d+", str(pid_raw))
        if not match:
            continue
        pid = match.group()
        normalized_role = normalize_role_label(role)
        if normalized_role:
            normalized_map[pid] = normalized_role
    return normalized_map


_NO_TARGET_TOKENS = {"", "0", "no", "none", "null", "-1", "pass"}


def _normalize_player_token(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in _NO_TARGET_TOKENS:
        return ""
    match = re.search(r"\d+", text)
    return match.group() if match else text


def _build_day_vote_context(log):
    """
    Collect per-day vote counts and expulsion targets from game_log.

    Returns:
      {
        "vote_counts_by_day": {day: Counter({"1": 2, ...})},
        "expelled_by_day": {day: "1" or ""},
      }
    """
    vote_counts_by_day = defaultdict(Counter)
    expelled_by_day = {}

    for event in log or []:
        if not isinstance(event, dict):
            continue

        try:
            day = int(event.get("day"))
        except Exception:
            continue

        event_type = str(event.get("event", ""))
        if event_type in {"vote", "vote_pk"}:
            target = _normalize_player_token(event.get("target"))
            if target:
                vote_counts_by_day[day][target] += 1
            continue

        if event_type != "end_vote":
            continue

        content = event.get("content", {}) or {}
        expelled = _normalize_player_token(
            content.get("expelled", event.get("target"))
        )
        if expelled:
            expelled_by_day[day] = expelled

    return {
        "vote_counts_by_day": vote_counts_by_day,
        "expelled_by_day": expelled_by_day,
    }


def _is_active_witch_action(content):
    if not isinstance(content, dict):
        return False

    for key, value in content.items():
        if str(key).strip().lower() == "pass":
            continue
        token = str(value).strip().lower()
        if token in _NO_TARGET_TOKENS:
            continue
        return True
    return False


def _build_witch_first_use_day(log):
    """
    Return the first day index when the witch used a real potion target.
    None means the witch never used an active potion in the available log.
    """
    first_use_day = None

    for event in log or []:
        if not isinstance(event, dict):
            continue
        if str(event.get("event", "")) != "skill_witch":
            continue

        try:
            day = int(event.get("day"))
        except Exception:
            continue

        content = event.get("content", {}) or {}
        if not _is_active_witch_action(content):
            continue

        if first_use_day is None or day < first_use_day:
            first_use_day = day

    return first_use_day


def _predicted_role(labels, target_id):
    if not isinstance(labels, dict):
        return "Unknown"
    raw = labels.get(str(target_id))
    if raw is None:
        raw = labels.get(f"Player {target_id}")
    normalized = normalize_role_label(raw)
    return normalized or "Unknown"


def _build_phase_specific_state_by_player_day(entries_by_player, phase_keyword):
    state_by_player_day = {}
    for viewer_id, entries in (entries_by_player or {}).items():
        for entry in entries:
            phase = str(entry.get("phase", ""))
            if phase_keyword not in phase:
                continue
            day = phase_day(phase)
            if day is None:
                continue
            parsed = None
            for field in ("response", "action"):
                candidate = parse_dict_maybe(entry.get(field))
                if not isinstance(candidate, dict):
                    continue
                by_num = {}
                for player_key, predicted_role in candidate.items():
                    match = re.search(r"\d+", str(player_key))
                    if not match:
                        continue
                    pid = match.group()
                    normalized_role = normalize_role_label(predicted_role)
                    by_num[pid] = normalized_role or str(predicted_role).strip() or "Unknown"
                if by_num:
                    parsed = by_num
                    break
            if isinstance(parsed, dict):
                state_by_player_day.setdefault(int(viewer_id), {})[day] = parsed
    return state_by_player_day


def _wolf_f1_for_labels(labels, role_map):
    if not isinstance(labels, dict):
        return None

    true_wolves = {pid for pid, role in role_map.items() if normalize_role_label(role) == "Werewolf"}
    if not true_wolves:
        return None

    predicted_wolves = {
        target_pid
        for target_pid in role_map.keys()
        if _predicted_role(labels, target_pid) == "Werewolf"
    }

    true_positive = len(predicted_wolves & true_wolves)
    false_positive = len(predicted_wolves - true_wolves)
    false_negative = len(true_wolves - predicted_wolves)
    denom = (2 * true_positive) + false_positive + false_negative
    if denom == 0:
        return 0.0
    return (2.0 * true_positive) / float(denom)


def _villager_team_wolf_f1(state_by_player_day, target_day, role_map):
    scores = []
    for viewer_pid, viewer_true_role in role_map.items():
        if _team_from_role(viewer_true_role) != "good":
            continue
        labels = state_by_player_day.get(int(viewer_pid), {}).get(target_day)
        if not isinstance(labels, dict):
            continue
        score = _wolf_f1_for_labels(labels, role_map)
        if score is not None:
            scores.append(float(score))
    if not scores:
        return None, 0
    return sum(scores) / len(scores), len(scores)


def _speaker_post_belief_snapshot(speaker_id, role_map, post_state_by_player_day, day):
    pred_by_player = {}
    role_counter = Counter()

    for viewer_pid, viewer_true_role in role_map.items():
        if int(viewer_pid) == int(speaker_id):
            continue
        labels = post_state_by_player_day.get(int(viewer_pid), {}).get(day)
        if not isinstance(labels, dict):
            continue
        predicted = _predicted_role(labels, speaker_id)
        pred_by_player[f"Player {viewer_pid}"] = predicted
        role_counter[predicted] += 1

    if role_counter:
        max_cnt = max(role_counter.values())
        top_roles = [role for role, cnt in role_counter.items() if cnt == max_cnt]
        majority_role = top_roles[0] if len(top_roles) == 1 else "Tie"
    else:
        majority_role = None

    return pred_by_player or None, majority_role, role_counter


def _label_speech_low_level(speaker_role_norm, day, majority_role, role_counter, witch_first_use_day):
    werewolf_count = int(role_counter.get("Werewolf", 0))
    if werewolf_count >= 2:
        return False, "recognized_as_werewolf_by_two_or_more_viewers"

    if speaker_role_norm == "Werewolf":
        if majority_role in {"Seer", "Witch", "Villager"}:
            return True, f"werewolf_recognized_as_{majority_role.lower()}"
        return None, None

    if speaker_role_norm == "Seer":
        if majority_role == "Seer":
            return True, "seer_recognized_as_seer"
        return None, None

    if speaker_role_norm == "Guard":
        if majority_role and majority_role not in {"Guard", "Werewolf", "Tie", "Unknown"}:
            return True, f"guard_not_recognized_as_guard_{majority_role.lower()}"
        return None, None

    if speaker_role_norm == "Witch":
        after_use = witch_first_use_day is not None and day > witch_first_use_day
        expected_role = "Witch" if after_use else "Villager"
        if majority_role == expected_role:
            return True, f"witch_recognized_as_{expected_role.lower()}"
        return None, None

    if speaker_role_norm in GOOD_ROLES:
        if majority_role and majority_role not in {"Werewolf", "Tie", "Unknown"}:
            return True, f"{speaker_role_norm.lower()}_not_recognized_as_werewolf"
        return None, None

    return None, None


def _normalize_player_token(value):
    if value is None:
        return ""
    match = re.search(r"\d+", str(value))
    if match:
        return match.group()
    return str(value).strip()


def _extract_speech_text(entry):
    """
    Best-effort speech text extraction from action/response payload.
    """
    for field in ("action", "response"):
        parsed = parse_dict_maybe(entry.get(field))
        if isinstance(parsed, dict):
            for key in ("Speech", "speech", "发言"):
                value = parsed.get(key)
                if value is not None:
                    text = str(value).strip()
                    if text:
                        return text
    for field in ("action", "response"):
        raw = entry.get(field)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    return ""


def _is_no_speech_text(text):
    normalized = str(text or "").strip().lower()
    if not normalized:
        return True
    if normalized in {"...", "…", "none", "no speech", "empty"}:
        return True
    # punctuation-only output
    if re.fullmatch(r"[.\s!?,;:~\-_=]+", normalized):
        return True
    return False

def label_speech_entry(
    entry,
    speaker_id,
    speaker_role,
    role_mapping,
    pre_state_by_player_day,
    post_state_by_player_day,
    speech_history=None,
    enable_asymmetric_seer_reward=True,
    seer_ally_seer_min_prob=0.5,
    seer_enemy_villager_min_prob=0.5,
    game_log=None,
):

    del pre_state_by_player_day
    del speech_history
    del enable_asymmetric_seer_reward
    del seer_ally_seer_min_prob
    del seer_enemy_villager_min_prob

    day = phase_day(str(entry.get("phase", "")))
    if day is None:
        return None, None, None, None

    speaker_role_norm = normalize_role_label(speaker_role)
    speaker_team = _team_from_role(speaker_role_norm)
    if speaker_team not in {"good", "werewolf"}:
        return None, None, None, None

    role_map = _normalize_role_mapping(role_mapping)
    if not role_map:
        return None, None, None, None

    vote_counts_by_phase = {}
    expelled_by_day = {}
    witch_first_use_day = None
    if game_log is not None:
        vote_counts_by_phase, expelled_by_day = _build_vote_context_by_phase(game_log)
        witch_first_use_day = _build_witch_first_use_day(game_log)

    pred_by_player, majority_role, role_counter = _speaker_post_belief_snapshot(
        speaker_id=speaker_id,
        role_map=role_map,
        post_state_by_player_day=post_state_by_player_day,
        day=day,
    )

    viewer_count = int(sum(role_counter.values()))
    wolf_count = int(role_counter.get("Werewolf", 0))
    guard_count = int(role_counter.get("Guard", 0))
    seer_count = int(role_counter.get("Seer", 0))
    villager_count = int(role_counter.get("Villager", 0))
    speech_text = _extract_speech_text(entry)
    no_speech = _is_no_speech_text(speech_text)
    phase = str(entry.get("phase", ""))
    vote_phase = None
    if "_day_speech_pk" in phase:
        vote_phase = f"{day}_day_vote_pk"
    elif "_day_speech" in phase:
        vote_phase = f"{day}_day_vote"

    speaker_token = _normalize_player_token(speaker_id)
    speaker_vote_count = None
    speaker_expelled = None
    if vote_phase is not None and vote_counts_by_phase:
        speaker_vote_count = int(vote_counts_by_phase.get(vote_phase, {}).get(speaker_token, 0))
        speaker_expelled = expelled_by_day.get(day) == speaker_token

    label, low_level_reason = _label_speech_low_level(
        speaker_role_norm=speaker_role_norm,
        day=day,
        majority_role=majority_role,
        role_counter=role_counter,
        witch_first_use_day=witch_first_use_day,
    )

    meta = {
        "criterion": "role_specific_perception_rule",
        "speaker": int(speaker_id),
        "speaker_role": speaker_role_norm,
        "speaker_team": speaker_team,
        "day": day,
        "phase": phase,
        "viewer_count": viewer_count,
        "villager_viewers": viewer_count,
        "wolf_count": wolf_count,
        "seer_count": seer_count,
        "guard_count": guard_count,
        "villager_count": villager_count,
        "no_speech": no_speech,
        "speaker_vote_count": speaker_vote_count,
        "speaker_expelled": speaker_expelled,
        "witch_first_use_day": witch_first_use_day,
        "speaker_post_belief_majority_role": majority_role,
        "speaker_post_belief_counter": dict(role_counter),
        "is_desirable_low_level": None if label is None else bool(label),
        "low_level_reason": low_level_reason,
    }
    return (None if label is None else bool(label)), pred_by_player, majority_role, meta


# ── Vote-received speech criteria ────────────────────────────────────────────
def _collect_speech_phase_speakers(log):
    """
    Build per-speech-phase speaker sets.
    """
    speakers_by_phase = defaultdict(set)

    for event in log:
        day = event.get("day")
        evt = str(event.get("event", ""))
        if day is None:
            continue

        if evt == "speech":
            speakers_by_phase[f"{day}_day_speech"].add(event.get("source"))
            continue
        if evt == "speech_pk":
            speakers_by_phase[f"{day}_day_speech_pk"].add(event.get("source"))
            continue

    return speakers_by_phase


def _build_vote_context_by_phase(log):

    vote_target_count_by_phase = defaultdict(lambda: defaultdict(int))
    expelled_by_day = {}
    for event in log:
        evt = str(event.get("event", ""))
        day = event.get("day")
        if day is None:
            continue
        try:
            day_num = int(day)
        except (TypeError, ValueError):
            continue

        if evt in {"vote", "vote_pk"}:
            target = _normalize_player_token(event.get("target"))
            if not target:
                continue
            if evt == "vote":
                phase = f"{day_num}_day_vote"
            else:
                phase = f"{day_num}_day_vote_pk"
            vote_target_count_by_phase[phase][str(target)] += 1
            continue

        if evt != "end_vote":
            continue

        content = event.get("content", {}) or {}
        expelled = _normalize_player_token(content.get("expelled", event.get("target")))
        if expelled:
            expelled_by_day[day_num] = expelled

    return vote_target_count_by_phase, expelled_by_day


def _extract_vote_based_speech_sets(log, role_mapping, team_kind):

    normalized_role_map = _normalize_role_mapping(role_mapping)
    good = []
    bad = []
    speakers_by_phase = _collect_speech_phase_speakers(log)
    vote_target_count_by_phase, expelled_by_day = _build_vote_context_by_phase(log)

    for phase, speakers in speakers_by_phase.items():
        if "_day_speech_pk" in phase:
            vote_phase = phase.replace("_day_speech_pk", "_day_vote_pk")
        elif "_day_speech" in phase:
            vote_phase = phase.replace("_day_speech", "_day_vote")
        else:
            continue
        vote_target_count = vote_target_count_by_phase.get(vote_phase, {})
        day = phase_day(phase)
        if day is None:
            continue
        for speaker in sorted(speakers):
            role = normalized_role_map.get(str(speaker), "")
            speaker_team = _team_from_role(role)
            if team_kind == "good" and speaker_team != "good":
                continue
            if team_kind == "werewolf" and speaker_team != "werewolf":
                continue
            received_vote_count = int(vote_target_count.get(str(speaker), 0))
            if team_kind == "good":
                is_desirable = received_vote_count == 0
            else:
                is_desirable = expelled_by_day.get(day) != _normalize_player_token(speaker)
            if is_desirable:
                good.append((int(speaker), phase))
            else:
                bad.append((int(speaker), phase))
    return good, bad


def extract_good_speech_villager(log, role_mapping, game_type):
    good, _ = _extract_vote_based_speech_sets(log, role_mapping, team_kind="good")
    return good


def extract_good_speech_werewolf(log, role_mapping, game_type):
    good, _ = _extract_vote_based_speech_sets(log, role_mapping, team_kind="werewolf")
    return good


def extract_bad_speech_villager(log, role_mapping, game_type):
    _, bad = _extract_vote_based_speech_sets(log, role_mapping, team_kind="good")
    return bad


def extract_bad_speech_werewolf(log, role_mapping, game_type):
    _, bad = _extract_vote_based_speech_sets(log, role_mapping, team_kind="werewolf")
    return bad


def high_level_speech_label(player_id, phase, good_set, bad_set):

    key = (player_id, phase)
    in_good = key in good_set
    in_bad = key in bad_set
    if in_good and in_bad:
        return None
    if in_good:
        return True
    if in_bad:
        return False
    return None


def combine_high_low_speech_label(high_label, low_label):

    if low_label is False:
        return False
    if high_label is True and low_label is True:
        return True
    return None


def split_speech_samples(model, args):

    good_map = model.empty_phase_map()
    bad_map = model.empty_phase_map()
    reason_meta = defaultdict(lambda: defaultdict(dict))

    for game_path, role_mapping, entries_by_player, werewolf_is_sft, good_is_sft in model.iter_game_contexts(
        args.game_dir,
        random_play=args.random_play,
        self_play=args.self_play,
        max_games=args.max_games,
    ):
        # Load raw game_log for vote-based Stage 2 criteria
        game_log_file = os.path.join(game_path, "game_log.json")
        try:
            with open(game_log_file, "r", encoding="utf-8") as f:
                game_log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            game_log = None

        # Pre-compute vote-based good/bad sets once per game
        if game_log is not None:
            good_villager_set = set(extract_good_speech_villager(game_log, role_mapping, model.game_type))
            bad_villager_set = set(extract_bad_speech_villager(game_log, role_mapping, model.game_type))
            good_wolf_set = set(extract_good_speech_werewolf(game_log, role_mapping, model.game_type))
            bad_wolf_set = set(extract_bad_speech_werewolf(game_log, role_mapping, model.game_type))
        else:
            good_villager_set = bad_villager_set = good_wolf_set = bad_wolf_set = set()

        pre_state_by_player_day = _build_phase_specific_state_by_player_day(
            entries_by_player,
            "_day_pre_state_reconstruction",
        )
        post_state_by_player_day = _build_phase_specific_state_by_player_day(
            entries_by_player,
            "_day_post_state_reconstruction",
        )
        speech_history = defaultdict(dict)  # reset per game for trend tracking

        for player_id, entries in entries_by_player.items():
            player_role = role_mapping.get(str(player_id), "")
            if player_role == "Werewolf" and not werewolf_is_sft:
                continue
            if player_role != "Werewolf" and not good_is_sft:
                continue

            if player_role == "Werewolf":
                good_set, bad_set = good_wolf_set, bad_wolf_set
            else:
                good_set, bad_set = good_villager_set, bad_villager_set

            for entry in entries:
                phase = str(entry.get("phase", ""))
                if "_day_speech" not in phase:
                    continue

                # Stage 1: Internal_state_criterion
                internal_label, _, _, _ = label_speech_entry(
                    entry=entry,
                    speaker_id=player_id,
                    speaker_role=player_role,
                    role_mapping=role_mapping,
                    pre_state_by_player_day=pre_state_by_player_day,
                    post_state_by_player_day=post_state_by_player_day,
                    speech_history=speech_history,
                    enable_asymmetric_seer_reward=model.enable_asymmetric_seer_reward,
                    seer_ally_seer_min_prob=model.seer_ally_seer_min_prob,
                    seer_enemy_villager_min_prob=model.seer_enemy_villager_min_prob,
                    game_log=game_log,
                )

                high_label = high_level_speech_label(player_id, phase, good_set, bad_set)
                final_label = combine_high_low_speech_label(high_label, internal_label)
                if final_label is True:
                    model.add_phase_to_map(good_map, os.path.abspath(game_path), player_id, phase)
                    reason_meta[os.path.abspath(game_path)][str(player_id)][phase] = {
                        "outcome": high_label,
                        "state_recon": internal_label,
                        "final": final_label,
                    }
                elif final_label is False:
                    model.add_phase_to_map(bad_map, os.path.abspath(game_path), player_id, phase)
                    reason_meta[os.path.abspath(game_path)][str(player_id)][phase] = {
                        "outcome": high_label,
                        "state_recon": internal_label,
                        "final": final_label,
                    }
                # else: skip

    return good_map, bad_map, reason_meta


def main():
    args = parse_args()
    model = build_model_from_args(args)
    good_map, bad_map, reason_meta = split_speech_samples(model, args)
    saved = model.save_good_bad_pair(
        out_to=os.path.abspath(args.out_to),
        category="speech",
        good_map=good_map,
        bad_map=bad_map,
        good_file_name=args.good_file,
        bad_file_name=args.bad_file,
    )
    reason_file = os.path.join(os.path.abspath(args.out_to), "speech_reason_meta.json")
    reason_plain = {
        game_path: {pid: phases for pid, phases in by_player.items()}
        for game_path, by_player in reason_meta.items()
    }
    with open(reason_file, "w", encoding="utf-8") as f:
        json.dump(reason_plain, f, indent=4, ensure_ascii=False)
    print(
        f"[speech] good={saved.get('good_count', 0)} -> {saved.get('good_file')}, "
        f"bad={saved.get('bad_count', 0)} -> {saved.get('bad_file')}"
    )
    print(f"[speech] reason meta -> {reason_file}")


if __name__ == "__main__":
    main()
