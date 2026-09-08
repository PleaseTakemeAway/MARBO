import argparse
import os

from MARBO.wolf.preference_extraction.reward import (
    DEFAULT_OUTPUT_FILE_NAMES,
    add_common_reward_args,
    build_model_from_args,
    build_night_self_role_map,
    build_night_state_by_day,
    extract_state_reconstruction_map,
    normalize_role_label,
    phase_day,
)

# Roles that must be correctly predicted for a Good label.
# Villagers are excluded — their identity is not strategically critical.
_SPECIAL_ROLES = {"Werewolf", "Seer", "Witch", "Guard", "Hunter"}

# Only these phases are labeled; the others are skipped.
_SKIP_PHASE_KEYWORDS = ("_night_state_reconstruction", "_day_pre_state_reconstruction")


def parse_args():
    parser = argparse.ArgumentParser(description="Export state reconstruction reward good/bad phase maps.")
    add_common_reward_args(parser)
    parser.add_argument(
        "--good_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE_NAMES["state_recon"]["good"],
        help="output file name for good state reconstruction phases",
    )
    parser.add_argument(
        "--bad_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE_NAMES["state_recon"]["bad"],
        help="output file name for bad state reconstruction phases",
    )
    return parser.parse_args()


def label_recon_entry(
    entry,
    role_mapping,
    player_id,           
    reconstructed_role_label,  
    self_reconstructed_role,  
    game_type,           
):

    phase = str(entry.get("phase", ""))
    if "state_reconstruction" not in phase:
        return None

    # Skip _night_state_reconstruction and _day_pre_state_reconstruction on day 0 only
    day = phase_day(phase)
    if day == 0:
        for kw in _SKIP_PHASE_KEYWORDS:
            if kw in phase:
                return None

    # Parse the predicted role map from the entry
    predicted = extract_state_reconstruction_map(entry)
    if not isinstance(predicted, dict) or not predicted:
        return None

    # Compare predictions against ground truth for special-role players only
    has_special = False
    for pid, actual_role in role_mapping.items():
        actual_norm = normalize_role_label(actual_role)
        if actual_norm not in _SPECIAL_ROLES:
            continue
        has_special = True
        pred_norm = normalize_role_label(predicted.get(str(pid)))
        if pred_norm != actual_norm:
            return False  # at least one special role mis-predicted → Bad

    if not has_special:
        return None  # no special-role players found → skip

    return True  # all special-role players correctly predicted → Good


def split_recon_samples(model, args):
    good_map = model.empty_phase_map()
    bad_map = model.empty_phase_map()

    for game_path, role_mapping, entries_by_player, werewolf_is_sft, good_is_sft in model.iter_game_contexts(
        args.game_dir,
        random_play=args.random_play,
        self_play=args.self_play,
        max_games=args.max_games,
    ):
        for player_id, entries in entries_by_player.items():
            player_role = role_mapping.get(str(player_id), "")
            if player_role == "Werewolf" and not werewolf_is_sft:
                continue
            if player_role != "Werewolf" and not good_is_sft:
                continue

            night_self_role_by_day = build_night_self_role_map(entries, player_id)
            night_state_by_day = build_night_state_by_day(entries)

            for entry in entries:
                phase = str(entry.get("phase", ""))
                if "state_reconstruction" not in phase:
                    continue

                day = phase_day(phase)

                label = label_recon_entry(
                    entry=entry,
                    role_mapping=role_mapping,
                    player_id=player_id,
                    reconstructed_role_label=night_state_by_day.get(day),
                    self_reconstructed_role=night_self_role_by_day.get(day),
                    game_type=model.game_type,
                )
                if label is None:
                    continue
                if label:
                    model.add_phase_to_map(good_map, os.path.abspath(game_path), player_id, phase)
                else:
                    model.add_phase_to_map(bad_map, os.path.abspath(game_path), player_id, phase)

    return good_map, bad_map


def main():
    args = parse_args()
    model = build_model_from_args(args)
    good_map, bad_map = split_recon_samples(model, args)
    saved = model.save_good_bad_pair(
        out_to=os.path.abspath(args.out_to),
        category="state_recon",
        good_map=good_map,
        bad_map=bad_map,
        good_file_name=args.good_file,
        bad_file_name=args.bad_file,
    )
    print(
        f"[state_recon] good={saved.get('good_count', 0)} -> {saved.get('good_file')}, "
        f"bad={saved.get('bad_count', 0)} -> {saved.get('bad_file')}"
    )


if __name__ == "__main__":
    main()
