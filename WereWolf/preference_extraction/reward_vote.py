import argparse
import json
import os
import re
from collections import defaultdict

from MARBO.wolf.preference_extraction.reward import (
    DEFAULT_OUTPUT_FILE_NAMES,
    add_common_reward_args,
    build_day_post_state_by_day,
    build_model_from_args,
    normalize_role_label,
    parse_dict_maybe,
    phase_day,
    to_text,
)


_DEAD_MARKERS = ("dead", "deceased", "killed", "eliminated", "out")


def parse_args():
    parser = argparse.ArgumentParser(description="Export vote reward good/bad phase maps.")
    add_common_reward_args(parser)
    parser.add_argument(
        "--good_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE_NAMES["vote"]["good"],
        help="output file name for good vote phases",
    )
    parser.add_argument(
        "--bad_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE_NAMES["vote"]["bad"],
        help="output file name for bad vote phases",
    )
    return parser.parse_args()


# ── Internal-state vote signal helper ──────────────────────────────────────

def _include_entry_by_gen_times(entry, max_gen_times: int = 3) -> bool:
    """
    Exclude entries that appear to come from repeated/random fallback generations.

    By convention, entries with gen_times >= 4 are treated as unreliable and are
    not included in vote reward extraction.
    """
    value = entry.get("gen_times")
    if value is None:
        return True
    try:
        return int(value) <= max_gen_times
    except (TypeError, ValueError):
        return True


def _extract_vote_target_from_entry(entry):
    try:
        from MARBO.wolf.preference_extraction.utils_english import _parse_response_json
    except ModuleNotFoundError:
        from MARBO.wolf.preference_extraction.utils_english import _parse_response_json

    for field in ("response", "action"):
        raw_value = entry.get(field)
        parsed = parse_dict_maybe(raw_value)

        if not isinstance(parsed, dict):
            text = to_text(raw_value).strip()
            if text:
                parsed = _parse_response_json(text)

        if not isinstance(parsed, dict):
            continue

        candidates = [parsed]
        nested = parsed.get("response")
        if isinstance(nested, dict):
            candidates.append(nested)

        for candidate in candidates:
            vote_value = candidate.get("voting_player", candidate.get("Vote", candidate.get("vote")))
            if vote_value is not None:
                return True, _normalize_vote_target(vote_value)

    return False, ""


def _is_predicted_werewolf(target_norm, reconstructed_role_label):
    if not target_norm or not isinstance(reconstructed_role_label, dict):
        return False

    raw = (
        reconstructed_role_label.get(target_norm)
        or reconstructed_role_label.get(str(target_norm))
        or reconstructed_role_label.get(f"Player {target_norm}")
    )
    return normalize_role_label(raw) == "Werewolf"


def _get_reconstructed_label(target_norm, reconstructed_role_label):
    if not target_norm or not isinstance(reconstructed_role_label, dict):
        return ""
    raw = (
        reconstructed_role_label.get(target_norm)
        or reconstructed_role_label.get(str(target_norm))
        or reconstructed_role_label.get(f"Player {target_norm}")
    )
    normalized = normalize_role_label(raw)
    if normalized:
        return normalized
    return str(raw).strip() if raw is not None else ""


def _is_dead_vote_attempt(entry, alive_player_ids=None, fallback_vote_target=None):
    parsed_vote, vote_target_norm = _extract_vote_target_from_entry(entry)
    if (not parsed_vote) and fallback_vote_target is not None:
        parsed_vote = True
        vote_target_norm = _normalize_vote_target(fallback_vote_target)

    if not parsed_vote or not vote_target_norm:
        return False

    alive_player_set = {str(pid) for pid in (alive_player_ids or []) if str(pid).strip()}
    if not alive_player_set:
        return False
    return str(vote_target_norm) not in alive_player_set


def _is_predicted_dead_vote_attempt(entry, reconstructed_role_label, fallback_vote_target=None):
    if not isinstance(reconstructed_role_label, dict):
        return False

    parsed_vote, vote_target_norm = _extract_vote_target_from_entry(entry)
    if (not parsed_vote) and fallback_vote_target is not None:
        parsed_vote = True
        vote_target_norm = _normalize_vote_target(fallback_vote_target)

    if not parsed_vote or not vote_target_norm:
        return False

    predicted_label = _get_reconstructed_label(vote_target_norm, reconstructed_role_label)
    return str(predicted_label).strip().lower() == "dead"


def _is_self_vote_attempt(entry, voter_id=None, fallback_vote_target=None):
    parsed_vote, vote_target_norm = _extract_vote_target_from_entry(entry)
    if (not parsed_vote) and fallback_vote_target is not None:
        parsed_vote = True
        vote_target_norm = _normalize_vote_target(fallback_vote_target)

    if not parsed_vote or not vote_target_norm:
        return False

    voter_norm = _normalize_vote_target(voter_id if voter_id is not None else entry.get("player_id"))
    if not voter_norm:
        return False
    return str(voter_norm) == str(vote_target_norm)


def label_vote_entry(
    entry,
    reconstructed_role_label,
    voter_role=None,
    eliminated_target=None,
    game_role_label=None,
    alive_player_ids=None,
    fallback_vote_target=None,
    voter_id=None,
):
    """
    Low-level vote reward from belief state + seer-indicated wolf signal.

    Villager-side desirable:
      - vote for a player reconstructed as Werewolf
      - OR vote for a wolf target pointed by a player reconstructed as Seer

    Villager-side undesirable:
      - if wolf-suspect exists, but voter does not vote any wolf-suspect
      - attempt to vote for a dead / non-alive player

    True  → desirable low-level vote
    False → undesirable low-level vote
    None  → insufficient low-level signal (skip low-level judgment)
    """
    if not isinstance(reconstructed_role_label, dict):
        return None

    parsed_vote, vote_target_norm = _extract_vote_target_from_entry(entry)
    if (not parsed_vote) and fallback_vote_target is not None:
        parsed_vote = True
        vote_target_norm = _normalize_vote_target(fallback_vote_target)

    if not parsed_vote:
        return None

    if _is_self_vote_attempt(entry, voter_id=voter_id, fallback_vote_target=fallback_vote_target):
        return False

    if _is_predicted_dead_vote_attempt(
        entry,
        reconstructed_role_label,
        fallback_vote_target=fallback_vote_target,
    ):
        return False

    alive_player_set = {str(pid) for pid in (alive_player_ids or []) if str(pid).strip()}
    if alive_player_set and str(vote_target_norm) not in alive_player_set:
        return False

    predicted_wolf_ids = _extract_predicted_role_ids(reconstructed_role_label, "Werewolf")
    seer_called_wolf_ids = _extract_seer_called_wolf_ids(entry, reconstructed_role_label)
    wolf_candidate_ids = set(predicted_wolf_ids) | set(seer_called_wolf_ids)

    # No vote while wolf candidates are present is undesirable.
    if not vote_target_norm:
        if wolf_candidate_ids:
            return False
        return None

    if str(vote_target_norm) in wolf_candidate_ids:
        return True

    # Wolf candidates exist but voted elsewhere.
    if wolf_candidate_ids:
        return False

    # No identifiable wolf candidate signal from low-level view.
    return None


# ── Ground-truth/context helpers ─────────────────────────────────────────────

def _normalize_vote_target(raw_target):
    if raw_target is None:
        return ""
    text = str(raw_target).strip()
    if not text:
        return ""
    if text.lower() in ("abstain", "0", "no", "none", "-1", "null", ""):
        return ""
    m = re.search(r"\d+", text)
    return m.group() if m else text


def _is_dead_like_label(raw_label):
    if raw_label is None:
        return False
    lowered = str(raw_label).strip().lower()
    return any(token in lowered for token in _DEAD_MARKERS)


def _extract_predicted_role_ids(reconstructed_role_label, target_role):
    ids = set()
    if not isinstance(reconstructed_role_label, dict):
        return ids
    for raw_pid, raw_role in reconstructed_role_label.items():
        pid = _normalize_vote_target(raw_pid)
        if not pid:
            continue
        role = normalize_role_label(raw_role)
        if role != target_role:
            continue
        if _is_dead_like_label(raw_role):
            continue
        ids.add(str(pid))
    return ids


def _extract_prompt_text(entry):
    prompt = entry.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                text = to_text(msg.get("content")).strip()
                if text:
                    return text
        merged = "\n".join(
            to_text(msg.get("content")).strip()
            for msg in prompt
            if isinstance(msg, dict)
        ).strip()
        return merged
    return to_text(prompt).strip()


def _extract_player_speeches(prompt_text):
    speeches = {}
    if not prompt_text:
        return speeches

    # Markdown block style: **Player 3**：...
    md_pattern = re.compile(
        r"\*\*Player\s*(\d+)\*\*[:：]\s*(.*?)(?=\n\*\*Player\s*\d+\*\*[:：]|\n\nYou are currently|$)",
        flags=re.DOTALL | re.IGNORECASE,
    )
    for m in md_pattern.finditer(prompt_text):
        pid = m.group(1)
        speech = m.group(2).strip()
        if speech:
            speeches[pid] = speech
    if speeches:
        return speeches

    # Fallback style: "3 said at ... : ..."
    plain_pattern = re.compile(
        r"(\d+)\s+said\s+at\s+.*?:\s*(.*?)(?=\d+\s+said\s+at|Please analyze|$)",
        flags=re.DOTALL | re.IGNORECASE,
    )
    for m in plain_pattern.finditer(prompt_text):
        pid = m.group(1)
        speech = m.group(2).strip()
        if speech:
            speeches[pid] = speech
    return speeches


def _extract_wolf_targets_from_speech(speech_text):
    targets = set()
    if not speech_text:
        return targets

    patterns = [
        r"Player\s*(\d+)[^\n]{0,80}?(?:werewolf|wolf|狼人)",
        r"(\d+)号[^\n]{0,40}?狼人",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, speech_text, flags=re.IGNORECASE):
            snippet = m.group(0).lower()
            # Skip likely negations: "not werewolf", "不是狼人"
            if re.search(r"not\s+(?:a\s+)?(?:werewolf|wolf)|不是.{0,6}狼人", snippet):
                continue
            pid = _normalize_vote_target(m.group(1))
            if pid:
                targets.add(str(pid))
    return targets


def _extract_seer_called_wolf_ids(entry, reconstructed_role_label):
    """
    From current vote prompt speeches, extract wolf IDs pointed by players
    reconstructed as Seer.
    """
    predicted_seers = _extract_predicted_role_ids(reconstructed_role_label, "Seer")
    if not predicted_seers:
        return set()

    prompt_text = _extract_prompt_text(entry)
    speeches = _extract_player_speeches(prompt_text)

    wolf_ids = set()
    for seer_pid in predicted_seers:
        wolf_ids.update(_extract_wolf_targets_from_speech(speeches.get(str(seer_pid), "")))
    return wolf_ids


def _build_vote_phase_context(log, role_mapping):

    phase_votes = defaultdict(list)          # phase -> [(source_str, target_norm)]
    logged_vote_target = defaultdict(dict)   # phase -> {source_str: target_norm}
    phase_expelled = {}                      # phase -> expelled_target_norm or ""
    pk_happens = set()
    alive_players_by_phase = {}
    alive_players = {str(pid) for pid in role_mapping.keys()}

    for item in log:
        day = item.get("day")
        event = item.get("event")
        if day is None:
            continue

        if event == "vote":
            phase = f"{day}_day_vote"
            alive_players_by_phase.setdefault(phase, sorted(alive_players))
            src = str(item.get("source"))
            tgt = _normalize_vote_target(item.get("target"))
            phase_votes[phase].append((src, tgt))
            logged_vote_target[phase][src] = tgt
            continue

        if event == "vote_pk":
            phase = f"{day}_day_vote_pk"
            alive_players_by_phase.setdefault(phase, sorted(alive_players))
            src = str(item.get("source"))
            tgt = _normalize_vote_target(item.get("target"))
            phase_votes[phase].append((src, tgt))
            logged_vote_target[phase][src] = tgt
            continue

        if event == "end_night":
            dead_list = (item.get("content", {}) or {}).get("dead_list", []) or []
            for dead_player in dead_list:
                dead_norm = _normalize_vote_target(dead_player)
                if dead_norm:
                    alive_players.discard(dead_norm)
            continue

        if event != "end_vote":
            continue

        content = item.get("content", {}) or {}
        if "expelled" not in content and content.get("vote_outcome") == "draw":
            phase_expelled[f"{day}_day_vote"] = ""
            pk_happens.add(day)
            continue

        if "expelled" in content:
            phase = f"{day}_day_vote_pk" if day in pk_happens else f"{day}_day_vote"
            expelled_norm = _normalize_vote_target(content.get("expelled"))
            phase_expelled[phase] = expelled_norm
            if expelled_norm:
                alive_players.discard(expelled_norm)
            pk_happens.discard(day)

    phase_context = {}
    all_phases = set(phase_votes.keys()) | set(phase_expelled.keys())
    for phase in all_phases:
        expelled_target = phase_expelled.get(phase, "")
        expelled_role = normalize_role_label(
            role_mapping.get(str(expelled_target))
            or role_mapping.get(f"Player {expelled_target}")
        )

        phase_context[phase] = {
            "eliminated_target": expelled_target,
            "wolf_expelled": expelled_role == "Werewolf",
            "alive_players": alive_players_by_phase.get(phase, []),
        }

    return phase_context, logged_vote_target


def _build_vote_phase_context_from_entries(entries_by_player, role_mapping):

    parsed_vote_target = defaultdict(dict)

    for player_id, entries in (entries_by_player or {}).items():
        for entry in entries:
            if not _include_entry_by_gen_times(entry):
                continue
            phase = str(entry.get("phase", ""))
            if "_day_vote" not in phase:
                continue
            parsed_vote, vote_target_norm = _extract_vote_target_from_entry(entry)
            if not parsed_vote:
                continue
            parsed_vote_target[phase][str(player_id)] = vote_target_norm

    phase_context = {}
    day_wolf_expelled = {}

    for phase, by_player in parsed_vote_target.items():
        vote_count = defaultdict(int)
        for target_norm in by_player.values():
            if not target_norm:
                continue
            vote_count[target_norm] += 1

        eliminated_target = ""
        if vote_count:
            max_count = max(vote_count.values())
            top_targets = [target for target, cnt in vote_count.items() if cnt == max_count]
            if len(top_targets) == 1:
                eliminated_target = top_targets[0]

        eliminated_role = normalize_role_label(
            role_mapping.get(str(eliminated_target))
            or role_mapping.get(f"Player {eliminated_target}")
        )
        wolf_expelled_phase = eliminated_role == "Werewolf"
        phase_context[phase] = {
            "eliminated_target": eliminated_target,
            "wolf_expelled_phase": wolf_expelled_phase,
        }

        day = phase_day(phase)
        if day is not None:
            day_wolf_expelled[day] = day_wolf_expelled.get(day, False) or wolf_expelled_phase

    for phase in parsed_vote_target.keys():
        day = phase_day(phase)
        if phase not in phase_context:
            phase_context[phase] = {}
        phase_context[phase]["wolf_expelled"] = day_wolf_expelled.get(day)

    return phase_context, parsed_vote_target


def _label_votes_from_log(log, role_mapping):

    phase_context, logged_vote_target = _build_vote_phase_context(log, role_mapping)
    good = []
    bad = []

    for item in log:
        event = item.get("event")
        if event not in ("vote", "vote_pk"):
            continue

        day = item.get("day")
        if day is None:
            continue

        phase = f"{day}_day_vote_pk" if event == "vote_pk" else f"{day}_day_vote"
        source = str(item.get("source"))
        voter_role = role_mapping.get(source, "")
        if voter_role == "Werewolf":
            continue

        ctx = phase_context.get(phase, {})
        if "wolf_expelled" not in ctx:
            continue
        target_norm = _normalize_vote_target(item.get("target"))
        alive_player_set = {str(pid) for pid in (ctx.get("alive_players") or []) if str(pid).strip()}
        if target_norm and alive_player_set and str(target_norm) not in alive_player_set:
            bad.append((int(source), phase))
            continue
        if bool(ctx.get("wolf_expelled")):
            good.append((int(source), phase))
        else:
            bad.append((int(source), phase))

    return good, bad


def extract_good_vote(log, role_mapping, game_type):
    """
    Compatibility API used by reward.py.
    """
    del game_type
    good, _ = _label_votes_from_log(log, role_mapping)
    return good


def extract_bad_vote(log, role_mapping, game_type):
    """
    Compatibility API used by reward.py.
    """
    del game_type
    _, bad = _label_votes_from_log(log, role_mapping)
    return bad


# ── Main split logic ─────────────────────────────────────────────────────────

def split_vote_samples(model, args):

    good_map = model.empty_phase_map()
    bad_map = model.empty_phase_map()
    # state_recon_meta[game_path][player_id][phase] = True / False / None
    state_recon_meta = defaultdict(lambda: defaultdict(dict))
    reason_meta = defaultdict(lambda: defaultdict(dict))

    for game_path, role_mapping, entries_by_player, werewolf_is_sft, good_is_sft in model.iter_game_contexts(
        args.game_dir,
        random_play=args.random_play,
        self_play=args.self_play,
        max_games=args.max_games,
    ):
        # Load raw game_log for vote context
        # Reconstruct vote context directly from the sampled completions.
        phase_context, logged_vote_target = _build_vote_phase_context_from_entries(entries_by_player, role_mapping)

        abs_game_path = os.path.abspath(game_path)

        for player_id, entries in entries_by_player.items():
            valid_entries = [entry for entry in entries if _include_entry_by_gen_times(entry)]
            player_role = role_mapping.get(str(player_id), "")
            if player_role == "Werewolf" and not werewolf_is_sft:
                continue
            if player_role != "Werewolf" and not good_is_sft:
                continue
            # Vote reward is defined for villager-side voters.
            if player_role == "Werewolf":
                continue

            # Use day_post_state_reconstruction: beliefs formed after speeches, before voting
            day_post_state_by_day = build_day_post_state_by_day(valid_entries)

            for entry in valid_entries:
                phase = str(entry.get("phase", ""))
                day = phase_day(phase)
                if day is None:
                    continue
                if "_day_vote" not in phase:
                    continue

                ctx = phase_context.get(phase, {})
                high_label = ctx.get("wolf_expelled")
                fallback_vote_target = (logged_vote_target.get(phase, {}) or {}).get(str(player_id))

                # Low-level process signal used as state_recon meta too.
                # Dead/non-alive vote attempt is treated as explicit process=False.
                if _is_predicted_dead_vote_attempt(
                    entry,
                    day_post_state_by_day.get(day),
                    fallback_vote_target=fallback_vote_target,
                ):
                    internal_label = False
                else:
                    internal_label = label_vote_entry(
                        entry,
                        day_post_state_by_day.get(day),
                        voter_role=player_role,
                        eliminated_target=ctx.get("eliminated_target"),
                        game_role_label=role_mapping,
                        alive_player_ids=ctx.get("alive_players"),
                        fallback_vote_target=fallback_vote_target,
                        voter_id=player_id,
                    )

                if high_label is None:
                    continue

                # Low-level signal has priority:
                #   False -> bad even if the day outcome was successful.
                if internal_label is False:
                    final_label = False
                elif internal_label is True or high_label is True:
                    final_label = True
                else:
                    final_label = None
                if final_label is None:
                    continue

                if final_label:
                    model.add_phase_to_map(good_map, abs_game_path, player_id, phase)
                else:
                    model.add_phase_to_map(bad_map, abs_game_path, player_id, phase)
                state_recon_meta[abs_game_path][str(player_id)][phase] = internal_label
                reason_meta[abs_game_path][str(player_id)][phase] = {
                    "outcome": high_label,
                    "state_recon": internal_label,
                    "final": final_label,
                }

    return good_map, bad_map, state_recon_meta, reason_meta


def main():
    args = parse_args()
    model = build_model_from_args(args)
    good_map, bad_map, state_recon_meta, reason_meta = split_vote_samples(model, args)
    out_to = os.path.abspath(args.out_to)
    saved = model.save_good_bad_pair(
        out_to=out_to,
        category="vote",
        good_map=good_map,
        bad_map=bad_map,
        good_file_name=args.good_file,
        bad_file_name=args.bad_file,
    )

    # Save state_recon companion: {game_path: {player_id: {phase: True/False/None}}}
    meta_file = os.path.join(out_to, "vote_state_recon_meta.json")
    # defaultdict → plain dict for JSON serialization
    meta_plain = {gp: {pid: phases for pid, phases in by_player.items()}
                  for gp, by_player in state_recon_meta.items()}
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_plain, f, indent=4, ensure_ascii=False)

    reason_file = os.path.join(out_to, "vote_reason_meta.json")
    reason_plain = {gp: {pid: phases for pid, phases in by_player.items()}
                    for gp, by_player in reason_meta.items()}
    with open(reason_file, "w", encoding="utf-8") as f:
        json.dump(reason_plain, f, indent=4, ensure_ascii=False)

    print(
        f"[vote] good={saved.get('good_count', 0)} -> {saved.get('good_file')}, "
        f"bad={saved.get('bad_count', 0)} -> {saved.get('bad_file')}"
    )
    print(f"[vote] state_recon meta -> {meta_file}")
    print(f"[vote] reason meta -> {reason_file}")


if __name__ == "__main__":
    main()
