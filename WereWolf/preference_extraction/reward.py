import argparse
import ast
import hashlib
import json
import os
import re
import sys
from collections import defaultdict


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)


ROLE_TO_EN = {
    "Villager": "Villager",
    "Werewolf": "Werewolf",
    "Seer": "Seer",
    "Witch": "Witch",
    "Guard": "Guard",
    "Hunter": "Hunter",
    "村民": "Villager",
    "狼人": "Werewolf",
    "预言家": "Seer",
    "女巫": "Witch",
    "守卫": "Guard",
    "猎人": "Hunter",
    "Unknown": "Unknown",
}

ROLE_ALIAS_TO_EN = {
    "villager": "Villager",
    "ordinary villager": "Villager",
    "simple villager": "Villager",
    "civilian": "Villager",
    "good person": "Villager",
    "werewolf": "Werewolf",
    "wolf": "Werewolf",
    "seer": "Seer",
    "witch": "Witch",
    "guard": "Guard",
    "hunter": "Hunter",
    "unknown": "Unknown",
}

GOOD_ROLES = {"Villager", "Seer", "Witch", "Guard", "Hunter"}

CATEGORIES = ("speech", "vote", "action", "state_recon")

DEFAULT_OUTPUT_FILE_NAMES = {
    "speech": {"good": "good_speech.json", "bad": "bad_speech.json"},
    "vote": {"good": "good_vote.json", "bad": "bad_vote.json"},
    "action": {"good": "good_action.json", "bad": "bad_action.json"},
    "state_recon": {"good": "good_state_recon.json", "bad": "bad_state_recon.json"},
}


def normalize_role_label(raw_role):
    if raw_role is None:
        return None

    role_str = str(raw_role).strip()
    if not role_str:
        return None

    role_str = ROLE_TO_EN.get(role_str, role_str)
    lowered = role_str.lower().strip()

    if lowered in ROLE_ALIAS_TO_EN:
        return ROLE_ALIAS_TO_EN[lowered]

    if lowered.startswith("unknown"):
        return "Unknown"

    contains_map = [
        ("werewolf", "Werewolf"),
        ("狼人", "Werewolf"),
        ("seer", "Seer"),
        ("预言家", "Seer"),
        ("witch", "Witch"),
        ("女巫", "Witch"),
        ("guard", "Guard"),
        ("守卫", "Guard"),
        ("hunter", "Hunter"),
        ("猎人", "Hunter"),
        ("villager", "Villager"),
        ("村民", "Villager"),
    ]
    for token, normalized in contains_map:
        if token in lowered:
            return normalized

    compact = re.sub(r"[^a-z]", "", lowered)
    compact_map = {
        "villager": "Villager",
        "ordinaryvillager": "Villager",
        "simplevillager": "Villager",
        "civilian": "Villager",
        "goodperson": "Villager",
        "werewolf": "Werewolf",
        "wolf": "Werewolf",
        "seer": "Seer",
        "witch": "Witch",
        "guard": "Guard",
        "hunter": "Hunter",
        "unknown": "Unknown",
    }
    return compact_map.get(compact)


def strip_code_fence(text):
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else stripped


def parse_dict_maybe(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None

    text = strip_code_fence(raw)
    if not text or not text.startswith("{"):
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, SyntaxError):
        return None


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def create_message(role, content):
    return {"role": role, "content": content}


# ── Module-level utility functions ────────────────────────────────────────────
# These are the canonical implementations shared by reward.py and reward_{}.py.
# RewardModel static methods below delegate to these so both call paths stay in sync.

def phase_day(phase):
    """Extract day number from phase string, e.g. '2_day_vote' → 2."""
    try:
        return int(str(phase).split("_")[0])
    except Exception:
        return None


def state_phase_priority(phase):
    if "_day_post_state_reconstruction" in phase:
        return 3
    if "_night_state_reconstruction" in phase:
        return 2
    if "_day_pre_state_reconstruction" in phase:
        return 1
    if "state_reconstruction" in phase:
        return 0
    return -1


def passes_trend_policy(current, previous, min_prob, require_strict_increase):
    if current is None:
        return True
    if previous is None:
        return current >= min_prob
    if require_strict_increase:
        if previous >= 0.999999:
            return current >= previous
        return current > previous
    return current >= previous


def extract_state_reconstruction_map(entry):
    for field in ("response", "action"):
        parsed = parse_dict_maybe(entry.get(field))
        if not isinstance(parsed, dict):
            continue
        by_num = {}
        for player_key, predicted_role in parsed.items():
            match = re.search(r"\d+", str(player_key))
            if not match:
                continue
            pid = match.group()
            normalized_role = normalize_role_label(predicted_role)
            if normalized_role is None:
                normalized_role = str(predicted_role).strip() if predicted_role is not None else "Unknown"
            by_num[pid] = normalized_role
        if by_num:
            return by_num
    return None


def extract_self_role_from_state_entry(entry, player_id):
    player_keys = [f"Player {player_id}", str(player_id)]
    for field in ("response", "action"):
        parsed = parse_dict_maybe(entry.get(field))
        if not isinstance(parsed, dict):
            continue
        for key in player_keys:
            if key in parsed:
                normalized = normalize_role_label(parsed.get(key))
                if normalized:
                    return normalized
    return None


def build_night_self_role_map(entries, player_id):
    mapping = {}
    for entry in entries:
        phase = str(entry.get("phase", ""))
        if "night_state_reconstruction" not in phase:
            continue
        day = phase_day(phase)
        if day is None:
            continue
        self_role = extract_self_role_from_state_entry(entry, player_id)
        if self_role:
            mapping[day] = self_role
    return mapping


def build_night_state_by_day(entries):
    """
    Build a per-day state reconstruction map using ONLY _night_state_reconstruction phases.
    These are the beliefs the player held right before their night action.
    """
    mapping = {}
    for entry in entries:
        phase = str(entry.get("phase", ""))
        if "_night_state_reconstruction" not in phase:
            continue
        day = phase_day(phase)
        if day is None:
            continue
        parsed_state = extract_state_reconstruction_map(entry)
        if isinstance(parsed_state, dict):
            mapping[day] = parsed_state
    return mapping


def build_day_post_state_by_day(entries):
    """
    Build a per-day state reconstruction map using ONLY _day_post_state_reconstruction phases.
    These are the beliefs the player held after the day speeches, right before voting.
    """
    mapping = {}
    for entry in entries:
        phase = str(entry.get("phase", ""))
        if "_day_post_state_reconstruction" not in phase:
            continue
        day = phase_day(phase)
        if day is None:
            continue
        parsed_state = extract_state_reconstruction_map(entry)
        if isinstance(parsed_state, dict):
            mapping[day] = parsed_state
    return mapping


def build_phase_state_by_player_day(entries_by_player, phase_keyword):
    """
    Build a per-viewer, per-day state map for a specific phase keyword.

    Example phase keywords:
      - "_day_pre_state_reconstruction"
      - "_day_post_state_reconstruction"
    """
    state_by_player_day = {}
    for viewer_id, entries in entries_by_player.items():
        for entry in entries:
            phase = str(entry.get("phase", ""))
            if phase_keyword not in phase:
                continue
            day = phase_day(phase)
            if day is None:
                continue
            parsed_state = extract_state_reconstruction_map(entry)
            if isinstance(parsed_state, dict):
                state_by_player_day.setdefault(viewer_id, {})[day] = parsed_state
    return state_by_player_day


def build_best_state_by_player_day(entries_by_player):
    state_by_player_day = {}
    priority_by_player_day = {}
    for viewer_id, entries in entries_by_player.items():
        for entry in entries:
            phase = str(entry.get("phase", ""))
            if "state_reconstruction" not in phase:
                continue
            day = phase_day(phase)
            if day is None:
                continue
            parsed_state = extract_state_reconstruction_map(entry)
            if not isinstance(parsed_state, dict):
                continue
            priority = state_phase_priority(phase)
            prev_priority = priority_by_player_day.get((viewer_id, day), -10)
            if priority < prev_priority:
                continue
            state_by_player_day.setdefault(viewer_id, {})[day] = parsed_state
            priority_by_player_day[(viewer_id, day)] = priority
    return state_by_player_day


class RewardModel:
    """
    Unified reward labeling model for:
      - speech
      - vote
      - action
      - state_reconstruction
    """

    IDENTITY_KEYS = [
        "Identity to Present",
        "identity to present",
        "identity_to_present",
        "identityToPresent",
        "self_present",
    ]

    def __init__(
        self,
        game_type="9p_seer_witch_guard",
        sft_model_regx="sft_agent",
        seer_ally_seer_min_prob=0.5,
        seer_enemy_villager_min_prob=0.5,
        enable_asymmetric_seer_reward=True,
    ):
        try:
            from MARBO.wolf.preference_extraction.utils_english import (
                get_role_assignment,
                get_system_prompt,
                judge_models,
                label_from_action_response,
                label_from_vote_response,
            )
        except ModuleNotFoundError:
            from MARBO.wolf.preference_extraction.utils_english import (
                get_role_assignment,
                get_system_prompt,
                judge_models,
                label_from_action_response,
                label_from_vote_response,
            )

        self.game_type = game_type
        self.sft_model_regx = sft_model_regx
        self.system_prompt = get_system_prompt(game_type)
        self._get_role_assignment = get_role_assignment
        self._judge_models = judge_models
        self._label_from_vote_response = label_from_vote_response
        self._label_from_action_response = label_from_action_response
        self.seer_ally_seer_min_prob = float(seer_ally_seer_min_prob)
        self.seer_enemy_villager_min_prob = float(seer_enemy_villager_min_prob)
        self.enable_asymmetric_seer_reward = bool(enable_asymmetric_seer_reward)

    @staticmethod
    def build_game_uid(game_path):
        normalized = os.path.abspath(str(game_path))
        base = os.path.basename(normalized.rstrip(os.sep)) or "game"
        parent = os.path.basename(os.path.dirname(normalized.rstrip(os.sep)))
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:8]
        return f"{parent}/{base}@{digest}"

    @staticmethod
    def _has_direct_game_dirs(base_dir):
        try:
            names = os.listdir(base_dir)
        except OSError:
            return False
        for name in names:
            path = os.path.join(base_dir, name)
            if os.path.isdir(path) and name.startswith("game_"):
                return True
        return False

    @staticmethod
    def _has_direct_w_dirs(base_dir):
        try:
            names = os.listdir(base_dir)
        except OSError:
            return False
        for name in names:
            path = os.path.join(base_dir, name)
            if os.path.isdir(path) and name.startswith("w-"):
                return True
        return False

    def _resolve_collection_roots(self, base_dir, random_play=False):
        """
        Resolve directories that can be consumed by iter_game_dirs.
        Supports:
          - random layout:   <root>/game_*
          - matchup layout:  <root>/w-*/game_*
          - dataset root containing multiple layout dirs one level below.
        """
        base_dir = os.path.abspath(base_dir)
        if random_play:
            return [base_dir]

        roots = []
        if self._has_direct_game_dirs(base_dir) or self._has_direct_w_dirs(base_dir):
            roots.append(base_dir)
            return roots

        try:
            sub_names = sorted(os.listdir(base_dir))
        except OSError:
            return roots

        for sub_name in sub_names:
            sub_dir = os.path.join(base_dir, sub_name)
            if not os.path.isdir(sub_dir):
                continue
            if self._has_direct_game_dirs(sub_dir) or self._has_direct_w_dirs(sub_dir):
                roots.append(sub_dir)
        return roots

    def iter_game_dirs(self, base_dir, random_play=False, self_play=False):
        roots = self._resolve_collection_roots(base_dir, random_play=random_play)
        if not roots:
            print(f"[WARN] No valid game collection root found under: {base_dir}")
            return

        for root in roots:
            root = os.path.abspath(root)
            has_game_dirs = self._has_direct_game_dirs(root)
            has_w_dirs = self._has_direct_w_dirs(root)

            # Random-style collection (<root>/game_*).
            if has_game_dirs and (random_play or not self_play):
                if root == os.path.abspath(base_dir):
                    print(f"Extract random-layout data from `{root}`...")
                else:
                    print(f"Extract random-layout data from nested root `{root}`...")
                for game_id in sorted(os.listdir(root)):
                    game_id_full = os.path.join(root, game_id)
                    if os.path.isdir(game_id_full) and game_id.startswith("game_"):
                        yield game_id_full, True, True

            # Matchup/self-play collection (<root>/w-*/game_*).
            if not has_w_dirs:
                continue

            for sub_dir in sorted(os.listdir(root)):
                sub_dir_full = os.path.join(root, sub_dir)
                if not os.path.isdir(sub_dir_full):
                    continue

                if self_play and (
                    "self_play" not in sub_dir
                    and f"w-{self.sft_model_regx}_vs_v-{self.sft_model_regx}" not in sub_dir
                ):
                    continue
                if (not self_play) and (not sub_dir.startswith("w-")):
                    continue

                if self_play:
                    werewolf_is_sft = True
                    good_is_sft = True
                else:
                    (_, werewolf_is_sft), (_, good_is_sft) = self._judge_models(sub_dir, self.sft_model_regx)
                    if (not werewolf_is_sft) and (not good_is_sft):
                        continue

                print(f"Extract matchup data from `{sub_dir_full}`...")
                for game_id in sorted(os.listdir(sub_dir_full)):
                    game_id_full = os.path.join(sub_dir_full, game_id)
                    if os.path.isdir(game_id_full) and game_id.startswith("game_"):
                        yield game_id_full, werewolf_is_sft, good_is_sft

    @staticmethod
    def read_player_entries(game_path):
        entries_by_player = {}
        if not os.path.isdir(game_path):
            return entries_by_player

        for filename in sorted(os.listdir(game_path)):
            if not (filename.startswith("Player_") and filename.endswith(".jsonl")):
                continue

            try:
                player_id = int(filename.replace("Player_", "").replace(".jsonl", ""))
            except ValueError:
                continue

            player_file = os.path.join(game_path, filename)
            entries = []
            with open(player_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            entries_by_player[player_id] = entries
        return entries_by_player

    def load_game_context(self, game_path):
        game_log_file = os.path.join(game_path, "game_log.json")
        if not os.path.exists(game_log_file):
            return None, None

        with open(game_log_file, "r", encoding="utf-8") as f:
            game_log = json.load(f)

        role_mapping = self._get_role_assignment(game_log) or {}
        role_mapping = {str(k): ROLE_TO_EN.get(v, v) for k, v in role_mapping.items()}
        entries_by_player = self.read_player_entries(game_path)
        return role_mapping, entries_by_player

    def iter_game_contexts(self, game_dir, random_play=False, self_play=False, max_games=None):
        processed = 0
        for game_path, werewolf_is_sft, good_is_sft in self.iter_game_dirs(
            game_dir, random_play=random_play, self_play=self_play
        ):
            if max_games is not None and processed >= max_games:
                break
            role_mapping, entries_by_player = self.load_game_context(game_path)
            if role_mapping is None or entries_by_player is None:
                continue
            yield game_path, role_mapping, entries_by_player, werewolf_is_sft, good_is_sft
            processed += 1

    @staticmethod
    def _phase_day(phase):
        return phase_day(phase)

    @staticmethod
    def _phase_sort_key(phase):
        day = phase_day(phase)
        day_order = day if day is not None else 10**9
        return (day_order, str(phase))

    @staticmethod
    def _team_from_role(role):
        normalized = normalize_role_label(role)
        if normalized == "Werewolf":
            return "werewolf"
        if normalized in GOOD_ROLES:
            return "good"
        return None

    def _extract_identity_claim_from_entry(self, entry):
        for field in ("action", "response"):
            parsed = parse_dict_maybe(entry.get(field))
            if not isinstance(parsed, dict):
                continue
            for key in self.IDENTITY_KEYS:
                if key in parsed:
                    normalized = normalize_role_label(parsed.get(key))
                    if normalized:
                        return normalized

        text_candidates = []
        for field in ("action", "response"):
            value = entry.get(field)
            if isinstance(value, str):
                text_candidates.append(value)

        patterns = [
            r"\bI\s*am\s*(?:an?\s+|the\s+)?(ordinary\s+villager|villager|werewolf|seer|witch|guard|hunter)\b",
            r"\bI['’]m\s*(?:an?\s+|the\s+)?(ordinary\s+villager|villager|werewolf|seer|witch|guard|hunter)\b",
            r"\bas\s*(?:an?\s+|the\s+)?(ordinary\s+villager|villager|werewolf|seer|witch|guard|hunter)\b",
        ]

        for text in text_candidates:
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if not match:
                    continue
                normalized = normalize_role_label(match.group(1))
                if normalized:
                    return normalized
        return None

    @staticmethod
    def _extract_self_role_from_state_entry(entry, player_id):
        return extract_self_role_from_state_entry(entry, player_id)

    def _build_night_self_role_map(self, entries, player_id):
        return build_night_self_role_map(entries, player_id)

    @staticmethod
    def _extract_state_reconstruction_map(entry):
        return extract_state_reconstruction_map(entry)

    @staticmethod
    def _state_phase_priority(phase):
        return state_phase_priority(phase)

    def _build_best_state_by_player_day(self, entries_by_player):
        return build_best_state_by_player_day(entries_by_player)

    @staticmethod
    def _passes_trend_policy(current, previous, min_prob, require_strict_increase):
        return passes_trend_policy(current, previous, min_prob, require_strict_increase)

    def _label_speech(
        self,
        entry,
        speaker_id,
        speaker_role,
        role_mapping,
        pre_state_by_player_day,
        post_state_by_player_day,
        speech_history=None,
        game_log=None,
    ):
        # Canonical implementation lives in reward_speech.label_speech_entry.
        from MARBO.wolf.preference_extraction.reward_speech import label_speech_entry
        return label_speech_entry(
            entry=entry,
            speaker_id=speaker_id,
            speaker_role=speaker_role,
            role_mapping=role_mapping,
            pre_state_by_player_day=pre_state_by_player_day,
            post_state_by_player_day=post_state_by_player_day,
            speech_history=speech_history,
            enable_asymmetric_seer_reward=self.enable_asymmetric_seer_reward,
            seer_ally_seer_min_prob=self.seer_ally_seer_min_prob,
            seer_enemy_villager_min_prob=self.seer_enemy_villager_min_prob,
            game_log=game_log,
        )

    def _label_vote(self, entry, reconstructed_role_label):
        # Canonical implementation lives in reward_vote.label_vote_entry.
        from MARBO.wolf.preference_extraction.reward_vote import label_vote_entry
        return label_vote_entry(entry=entry, reconstructed_role_label=reconstructed_role_label)

    def _label_action_or_state(
        self,
        entry,
        role_mapping,
        player_id,
        reconstructed_role_label=None,
        self_reconstructed_role=None,
    ):
        phase = str(entry.get("phase", ""))
        if not phase:
            return None
        if "state_reconstruction" in phase:
            # state_recon is sourced from reward_reconstruction.py.
            return None
        from MARBO.wolf.preference_extraction.reward_action import label_action_entry
        return label_action_entry(
            entry=entry,
            role_mapping=role_mapping,
            player_id=player_id,
            reconstructed_role_label=reconstructed_role_label,
            self_reconstructed_role=self_reconstructed_role,
            game_type=self.game_type,
        )

    @staticmethod
    def _phase_to_sample_type(phase):
        if "_day_speech" in phase:
            return "speech"
        if "_day_vote" in phase:
            return "vote"
        if "night_skill" in phase:
            return "action"
        if "state_reconstruction" in phase:
            return "state_recon"
        return None

    @staticmethod
    def _extract_player_id_from_prompt_messages(prompt_msgs):
        if not isinstance(prompt_msgs, list):
            return None
        for msg in prompt_msgs:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = str(msg.get("content", ""))
            m = re.search(r"\bYou are Player\s+(\d+)\b", content, flags=re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    return None
        return None

    def _collect_state_recon_samples_from_reconstruction(
        self,
        game_log,
        role_mapping,
        game_path,
        include_werewolf,
        include_good,
    ):
        """
        Build state_recon samples from reward_reconstruction.py and merge into
        the unified sample schema used by reward.py.
        """
        from MARBO.wolf.preference_extraction.reward_reconstruction import build_state_reconstruction_samples

        raw_samples = build_state_reconstruction_samples(
            log=game_log,
            game_path=game_path,
            system_prompt=self.system_prompt,
        )
        if not raw_samples:
            return []

        game_path_abs = os.path.abspath(game_path)
        game_uid = self.build_game_uid(game_path_abs)
        merged_samples = []
        for raw in raw_samples:
            if not isinstance(raw, dict):
                continue

            player_id = raw.get("player_id")
            if player_id is None:
                player_id = self._extract_player_id_from_prompt_messages(raw.get("prompt"))
            try:
                player_id = int(player_id)
            except (TypeError, ValueError):
                continue

            player_role = role_mapping.get(str(player_id), "")
            if player_role == "Werewolf" and not include_werewolf:
                continue
            if player_role != "Werewolf" and not include_good:
                continue

            sample = dict(raw)
            sample["sample_type"] = "state_recon"
            sample["player_id"] = player_id
            sample["game_path"] = os.path.abspath(sample.get("game_path") or game_path_abs)
            sample["game_uid"] = game_uid
            sample["state_recon"] = None
            sample["label"] = bool(sample.get("label", True))
            sample.setdefault("reconstructed_state_day", phase_day(str(sample.get("phase", ""))))
            sample.setdefault("reconstructed_role_label", None)
            sample.setdefault("self_reconstructed_role", None)
            sample.setdefault("internal_state_role_label", None)
            sample.setdefault("internal_state_werewolf_candidates", None)
            sample.setdefault("speech_asymmetric_meta", None)
            if not isinstance(sample.get("role_label"), dict):
                sample["role_label"] = role_mapping
            merged_samples.append(sample)
        return merged_samples

    @staticmethod
    def _state_recon_kind_from_phase(phase):
        phase = str(phase)
        if "_night_state_reconstruction" in phase:
            return "night"
        if "state_reconstruction" in phase:
            return "day"
        return None

    def _build_state_recon_lookup(self, recon_samples):
        """
        Build lookup: (player_id, day, kind[day|night]) -> bool label.
        Source of truth is reward_reconstruction samples.
        """
        lookup = {}
        for sample in recon_samples:
            if not isinstance(sample, dict):
                continue
            try:
                player_id = int(sample.get("player_id"))
            except (TypeError, ValueError):
                continue
            phase = str(sample.get("phase", ""))
            day = phase_day(phase)
            kind = self._state_recon_kind_from_phase(phase)
            if day is None or kind is None:
                continue
            label = sample.get("label")
            if label is True:
                lookup[(player_id, day, kind)] = True
            elif label is False:
                lookup[(player_id, day, kind)] = False
        return lookup

    @staticmethod
    def _to_player_key_state_map(state_map):
        if not isinstance(state_map, dict):
            return None
        converted = {}
        for key, value in state_map.items():
            match = re.search(r"\d+", str(key))
            if not match:
                continue
            converted[f"Player {match.group()}"] = value
        return converted if converted else None

    @staticmethod
    def _extract_werewolf_candidates(state_map):
        if not isinstance(state_map, dict):
            return None

        candidates = []
        for player_key, role_label in state_map.items():
            if normalize_role_label(role_label) != "Werewolf":
                continue
            match = re.search(r"\d+", str(player_key))
            if not match:
                continue
            candidates.append(int(match.group()))

        if not candidates:
            return []
        return [f"Player {pid}" for pid in sorted(set(candidates))]

    def _build_sample(
        self,
        entry,
        label,
        role_mapping,
        game_path,
        player_id,
        sample_type,
        day,
        reconstructed_role_label,
        self_reconstructed_role,
        speech_asymmetric_meta=None,
        state_recon=None,
        label_reason=None,
    ):
        prompt_raw = entry.get("prompt", "")
        completion_raw = entry.get("response", "")

        prompt_text = prompt_raw if isinstance(prompt_raw, str) else to_text(prompt_raw)
        if isinstance(completion_raw, str):
            completion_text = completion_raw
        else:
            completion_text = to_text(completion_raw)

        if not prompt_text or not completion_text:
            return None

        converted_state = self._to_player_key_state_map(reconstructed_role_label)
        internal_werewolf_candidates = self._extract_werewolf_candidates(converted_state)

        return {
            "prompt": [
                create_message("system", self.system_prompt),
                create_message("user", prompt_text),
            ],
            "completion": [create_message("assistant", completion_text)],
            "label": bool(label),
            "role_label": role_mapping,
            "phase": entry.get("phase", ""),
            "sample_type": sample_type,
            "game_path": game_path,
            "game_uid": self.build_game_uid(game_path),
            "player_id": player_id,
            "reconstructed_state_day": day,
            "reconstructed_role_label": converted_state,
            "self_reconstructed_role": self_reconstructed_role,
            "internal_state_role_label": converted_state,
            "internal_state_werewolf_candidates": internal_werewolf_candidates,
            "speech_asymmetric_meta": (
                speech_asymmetric_meta if sample_type == "speech" else None
            ),
            "state_recon": state_recon,
            "label_reason": label_reason,
        }

    def collect_game_samples(self, game_path, include_werewolf=True, include_good=True):
        game_log_file = os.path.join(game_path, "game_log.json")
        if not os.path.exists(game_log_file):
            print(game_log_file, "not exist")
            return []

        with open(game_log_file, "r", encoding="utf-8") as f:
            game_log = json.load(f)

        role_mapping = self._get_role_assignment(game_log) or {}
        role_mapping = {str(k): ROLE_TO_EN.get(v, v) for k, v in role_mapping.items()}

        entries_by_player = {}
        for filename in sorted(os.listdir(game_path)):
            if not (filename.startswith("Player_") and filename.endswith(".jsonl")):
                continue

            player_id = int(filename.replace("Player_", "").replace(".jsonl", ""))

            player_file = os.path.join(game_path, filename)
            entries = []
            with open(player_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            entries_by_player[player_id] = entries

        day_pre_state_by_player_day = build_phase_state_by_player_day(
            entries_by_player,
            "_day_pre_state_reconstruction",
        )
        day_post_state_by_player_day_all = build_phase_state_by_player_day(
            entries_by_player,
            "_day_post_state_reconstruction",
        )
        speech_history = defaultdict(dict)

        # Pre-compute Stage 2 sets once per game (speech + vote).
        from MARBO.wolf.adversarial_data_extraction.reward_speech import (
            combine_high_low_speech_label,
            extract_bad_speech_villager,
            extract_bad_speech_werewolf,
            extract_good_speech_villager,
            extract_good_speech_werewolf,
            high_level_speech_label,
        )


        from MARBO.wolf.adversarial_data_extraction.reward_vote import extract_bad_vote, extract_good_vote
        good_villager_speech_set = set(extract_good_speech_villager(game_log, role_mapping, self.game_type))
        bad_villager_speech_set = set(extract_bad_speech_villager(game_log, role_mapping, self.game_type))
        good_wolf_speech_set = set(extract_good_speech_werewolf(game_log, role_mapping, self.game_type))
        bad_wolf_speech_set = set(extract_bad_speech_werewolf(game_log, role_mapping, self.game_type))
        good_vote_set = set(extract_good_vote(game_log, role_mapping, self.game_type))
        bad_vote_set = set(extract_bad_vote(game_log, role_mapping, self.game_type))

        recon_samples_for_game = self._collect_state_recon_samples_from_reconstruction(
            game_log=game_log,
            role_mapping=role_mapping,
            game_path=game_path,
            include_werewolf=include_werewolf,
            include_good=include_good,
        )
        state_recon_lookup = self._build_state_recon_lookup(recon_samples_for_game)
        from MARBO.wolf.adversarial_data_extraction.reward_action import build_night_outcome_by_day
        night_outcome_by_day = build_night_outcome_by_day(game_path, role_mapping)

        samples = []
        for player_id, entries in entries_by_player.items():
            player_role = role_mapping.get(str(player_id), "")
            if player_role == "Werewolf" and not include_werewolf:
                continue
            if player_role != "Werewolf" and not include_good:
                continue

            if player_role == "Werewolf":
                good_speech_set, bad_speech_set = good_wolf_speech_set, bad_wolf_speech_set
            else:
                good_speech_set, bad_speech_set = good_villager_speech_set, bad_villager_speech_set

            night_self_role_by_day = self._build_night_self_role_map(entries, player_id)
            # vote uses day_post (after speeches, before voting)
            # action uses night (right before night action)
            day_post_state_by_day = build_day_post_state_by_day(entries)
            night_state_by_day = build_night_state_by_day(entries)

            for entry in entries:
                phase = str(entry.get("phase", ""))
                if not phase:
                    continue
                day = self._phase_day(phase)

                sample_type = self._phase_to_sample_type(phase)
                if sample_type is None:
                    continue
                if sample_type == "state_recon":
                    # State reconstruction samples are sourced from
                    # reward_reconstruction.py and merged later per-game.
                    continue

                current_self_reconstructed_role = night_self_role_by_day.get(day)

                speech_opponent_pred_map = None
                speech_majority_role = None
                speech_asymmetric_meta = None
                vote_state_recon = None
                action_state_recon = None
                vote_process_label = None
                action_process_label = None
                label_reason = None
                if sample_type == "speech":
                    # Unified speech labeling:
                    #   high-level(outcome) + low-level(IG) -> single final label.
                    internal_label, speech_opponent_pred_map, speech_majority_role, speech_asymmetric_meta = self._label_speech(
                        entry=entry,
                        speaker_id=player_id,
                        speaker_role=player_role,
                        role_mapping=role_mapping,
                        pre_state_by_player_day=day_pre_state_by_player_day,
                        post_state_by_player_day=day_post_state_by_player_day_all,
                        speech_history=speech_history,
                        game_log=game_log,
                    )
                    high_label = high_level_speech_label(
                        player_id=player_id,
                        phase=phase,
                        good_set=good_speech_set,
                        bad_set=bad_speech_set,
                    )
                    label = combine_high_low_speech_label(high_label, internal_label)
                    if label is None:
                        continue
                    label_reason = {
                        "outcome": high_label,
                        "state_recon": internal_label,
                        "final": label,
                    }
                elif sample_type == "vote":
                    # Start from the precomputed good/bad vote sets, then let the
                    # process signal override outcome when it is informative.
                    if (player_id, phase) in good_vote_set:
                        label = True
                    elif (player_id, phase) in bad_vote_set:
                        label = False
                    else:
                        continue  # Werewolf voter or not in either set → skip

                    # Process signal from internal-state vote reasoning.
                    vote_process_label = self._label_vote(entry, day_post_state_by_day.get(day))
                    if vote_process_label is False:
                        label = False
                    elif vote_process_label is True:
                        label = True
                    elif label is False:
                        continue
                    label_reason = {
                        "outcome": label,
                        "state_recon": vote_process_label,
                        "final": label,
                    }

                    vote_state_recon = state_recon_lookup.get((player_id, day, "day"))
                    if vote_state_recon is None:
                        vote_state_recon = vote_process_label
                elif sample_type == "action":
                    # Action label uses role-specific outcome/process rules.
                    from MARBO.wolf.adversarial_data_extraction.reward_action import evaluate_action_entry
                    label, action_process_label = evaluate_action_entry(
                        entry=entry,
                        role_mapping=role_mapping,
                        player_id=player_id,
                        reconstructed_role_label=night_state_by_day.get(day),
                        self_reconstructed_role=current_self_reconstructed_role,
                        game_type=self.game_type,
                        night_outcome_by_day=night_outcome_by_day,
                    )
                    # For night_skill bad samples: require BOTH
                    #   result(outcome)=False and process=False.
                    if label is False and action_process_label is not False:
                        continue
                    label_reason = {
                        "outcome": label,
                        "state_recon": action_process_label,
                        "final": label,
                    }

                    action_state_recon = state_recon_lookup.get((player_id, day, "night"))
                    if action_state_recon is None:
                        action_state_recon = action_process_label
                else:
                    continue

                if label is None:
                    continue

                sample = self._build_sample(
                    entry=entry,
                    label=label,
                    role_mapping=role_mapping,
                    game_path=game_path,
                    player_id=player_id,
                    sample_type=sample_type,
                    day=day,
                    reconstructed_role_label=(
                        speech_opponent_pred_map
                        if sample_type == "speech"
                        else night_state_by_day.get(day)
                    ),
                    self_reconstructed_role=(
                        speech_majority_role
                        if sample_type == "speech"
                        else current_self_reconstructed_role
                    ),
                    speech_asymmetric_meta=speech_asymmetric_meta,
                    state_recon=(
                        vote_state_recon
                        if sample_type == "vote"
                        else action_state_recon
                        if sample_type == "action"
                        else None
                    ),
                    label_reason=label_reason,
                )
                if sample is not None:
                    samples.append(sample)

        samples.extend(recon_samples_for_game)

        return samples

    def build_kto_dataset(self, game_dir, random_play=False, self_play=False, max_games=None):
        all_samples = []
        processed = 0
        for game_path, werewolf_is_sft, good_is_sft in self.iter_game_dirs(
            game_dir, random_play=random_play, self_play=self_play
        ):
            if max_games is not None and processed >= max_games:
                break

            game_samples = self.collect_game_samples(
                game_path=game_path,
                include_werewolf=werewolf_is_sft,
                include_good=good_is_sft,
            )
            all_samples.extend(game_samples)
            processed += 1

        return all_samples

    @staticmethod
    def _freeze_phase_map(phase_map):
        out = {}
        for game_path in sorted(phase_map.keys()):
            by_player = phase_map[game_path]
            player_out = {}
            sorted_players = sorted(
                by_player.keys(),
                key=lambda p: int(p) if str(p).isdigit() else str(p),
            )
            for player_id in sorted_players:
                phases = sorted(by_player[player_id], key=RewardModel._phase_sort_key)
                if phases:
                    player_out[str(player_id)] = phases
            if player_out:
                out[game_path] = player_out
        return out

    def split_samples_by_type_label(self, samples):
        phase_maps = {
            category: {
                "good": defaultdict(lambda: defaultdict(set)),
                "bad": defaultdict(lambda: defaultdict(set)),
            }
            for category in CATEGORIES
        }

        for sample in samples:
            category = sample.get("sample_type")
            if category not in phase_maps:
                continue

            label = sample.get("label")
            if label is True:
                label_key = "good"
            elif label is False:
                label_key = "bad"
            else:
                continue

            phase = str(sample.get("phase", ""))
            player_id = sample.get("player_id")
            game_path = os.path.abspath(str(sample.get("game_path", "")))
            if not phase or player_id is None or not game_path:
                continue

            phase_maps[category][label_key][game_path][str(player_id)].add(phase)

        frozen = {}
        for category in CATEGORIES:
            frozen[category] = {
                "good": self._freeze_phase_map(phase_maps[category]["good"]),
                "bad": self._freeze_phase_map(phase_maps[category]["bad"]),
            }
        return frozen

    @staticmethod
    def _phase_count(phase_map):
        return sum(len(phases) for by_player in phase_map.values() for phases in by_player.values())

    @staticmethod
    def empty_phase_map():
        return defaultdict(lambda: defaultdict(set))

    @staticmethod
    def add_phase_to_map(phase_map, game_path, player_id, phase):
        if not game_path or player_id is None or not phase:
            return
        phase_map[str(game_path)][str(player_id)].add(str(phase))

    def save_good_bad_phase_maps(self, out_to, split_maps, categories=None, file_name_map=None):
        os.makedirs(out_to, exist_ok=True)
        categories = categories or CATEGORIES
        file_name_map = file_name_map or DEFAULT_OUTPUT_FILE_NAMES

        saved = {}
        for category in categories:
            if category not in split_maps:
                continue
            if category not in file_name_map:
                continue

            good_data = split_maps[category]["good"]
            bad_data = split_maps[category]["bad"]

            good_file = os.path.join(out_to, file_name_map[category]["good"])
            bad_file = os.path.join(out_to, file_name_map[category]["bad"])

            with open(good_file, "w", encoding="utf-8") as f:
                json.dump(good_data, f, indent=4, ensure_ascii=False)
            with open(bad_file, "w", encoding="utf-8") as f:
                json.dump(bad_data, f, indent=4, ensure_ascii=False)

            saved[category] = {
                "good_file": good_file,
                "bad_file": bad_file,
                "good_count": self._phase_count(good_data),
                "bad_count": self._phase_count(bad_data),
            }
        return saved

    def save_good_bad_pair(
        self,
        out_to,
        category,
        good_map,
        bad_map,
        good_file_name=None,
        bad_file_name=None,
    ):
        os.makedirs(out_to, exist_ok=True)
        name_cfg = DEFAULT_OUTPUT_FILE_NAMES.get(category, {})
        good_file_name = good_file_name or name_cfg.get("good", f"good_{category}.json")
        bad_file_name = bad_file_name or name_cfg.get("bad", f"bad_{category}.json")

        frozen_good = self._freeze_phase_map(good_map)
        frozen_bad = self._freeze_phase_map(bad_map)

        good_file = os.path.join(out_to, good_file_name)
        bad_file = os.path.join(out_to, bad_file_name)
        with open(good_file, "w", encoding="utf-8") as f:
            json.dump(frozen_good, f, indent=4, ensure_ascii=False)
        with open(bad_file, "w", encoding="utf-8") as f:
            json.dump(frozen_bad, f, indent=4, ensure_ascii=False)

        return {
            "good_file": good_file,
            "bad_file": bad_file,
            "good_count": self._phase_count(frozen_good),
            "bad_count": self._phase_count(frozen_bad),
        }


def add_common_reward_args(parser):
    parser.add_argument(
        "--game_dir",
        type=str,
        default="./trial_logs",
        help="path to experiments",
    )
    parser.add_argument(
        "--game_type",
        type=str,
        default="9p_seer_witch_guard",
        help="choose from: 9p_seer_witch_guard, 9p_seer_witch_hunter, 7p_seer_guard",
    )
    parser.add_argument(
        "--sft_model_regx",
        type=str,
        default="sft_agent",
        help="model regex used for side selection in mixed-model directories",
    )
    parser.add_argument(
        "--out_to",
        type=str,
        required=True,
        help="output directory",
    )
    parser.add_argument(
        "--self_play",
        action="store_true",
        help="whether to select from self play",
    )
    parser.add_argument(
        "--random_play",
        action="store_true",
        help="game_dir contains game_N dirs directly (no w- subdirs)",
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=None,
        help="optional max number of games for quick debugging",
    )
    parser.add_argument(
        "--disable_asymmetric_seer_reward",
        action="store_true",
        help="disable asymmetric internal-state reward for Seer speech",
    )
    parser.add_argument(
        "--seer_ally_seer_min_prob",
        type=float,
        default=0.5,
        help="minimum ally-side Seer probability for first observed day",
    )
    parser.add_argument(
        "--seer_enemy_villager_min_prob",
        type=float,
        default=0.5,
        help="minimum enemy-side Villager probability for first observed day",
    )
    return parser


def build_model_from_args(args):
    return RewardModel(
        game_type=args.game_type,
        sft_model_regx=args.sft_model_regx,
        seer_ally_seer_min_prob=args.seer_ally_seer_min_prob,
        seer_enemy_villager_min_prob=args.seer_enemy_villager_min_prob,
        enable_asymmetric_seer_reward=(not args.disable_asymmetric_seer_reward),
    )


def collect_samples_from_args(args):
    model = build_model_from_args(args)
    samples = model.build_kto_dataset(
        game_dir=args.game_dir,
        random_play=args.random_play,
        self_play=args.self_play,
        max_games=args.max_games,
    )
    return model, samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified reward extraction for speech/vote/action/state_recon."
    )
    add_common_reward_args(parser)

    parser.add_argument(
        "--speech_good_file",
        type=str,
        default="adversarial_good_speech.json",
        help="output file name for good speech phases",
    )
    parser.add_argument(
        "--speech_bad_file",
        type=str,
        default="adversarial_bad_speech.json",
        help="output file name for bad speech phases",
    )
    parser.add_argument(
        "--vote_good_file",
        type=str,
        default="villager_good_vote.json",
        help="output file name for good vote phases",
    )
    parser.add_argument(
        "--vote_bad_file",
        type=str,
        default="villager_bad_vote.json",
        help="output file name for bad vote phases",
    )
    parser.add_argument(
        "--action_good_file",
        type=str,
        default="good_actions.json",
        help="output file name for good action phases",
    )
    parser.add_argument(
        "--action_bad_file",
        type=str,
        default="bad_action.json",
        help="output file name for bad action phases",
    )
    parser.add_argument(
        "--recon_good_file",
        type=str,
        default="adversarial_good_state_recon.json",
        help="output file name for good state reconstruction phases",
    )
    parser.add_argument(
        "--recon_bad_file",
        type=str,
        default="adversarial_bad_state_recon.json",
        help="output file name for bad state reconstruction phases",
    )
    parser.add_argument(
        "--good_speech_full_file",
        type=str,
        default="adversarial_good_speech_full.json",
        help="output file name for desirable full speech samples used by conflict filtering",
    )
    parser.add_argument(
        "--state_samples_file",
        type=str,
        default="adversarial_state_samples.json",
        help="output file name for direct state_reconstruction samples (prompt/completion/label)",
    )
    parser.add_argument(
        "--state_pair_samples_file",
        type=str,
        default="adversarial_state_pair_samples.json",
        help="output file name for KTO-style state_reconstruction samples from teacher/original mismatches",
    )
    return parser.parse_args()


def build_good_speech_full(samples):
    out = []
    for x in samples:
        if x.get("sample_type") != "speech" or x.get("label") is not True:
            continue

        prompt_msgs = x.get("prompt", [])
        completion_msgs = x.get("completion", [])
        user_prompt = ""
        response = ""
        system_prompt = ""

        for m in prompt_msgs:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "system" and not system_prompt:
                system_prompt = str(m.get("content", ""))
            if m.get("role") == "user" and not user_prompt:
                user_prompt = str(m.get("content", ""))

        for m in completion_msgs:
            if isinstance(m, dict) and m.get("role") == "assistant":
                response = str(m.get("content", ""))
                break

        if not user_prompt or not response:
            continue

        out.append({
            "game_path": x.get("game_path", ""),
            "game_uid": x.get("game_uid") or RewardModel.build_game_uid(x.get("game_path", "")),
            "player_id": x.get("player_id"),
            "phase": x.get("phase", ""),
            "system_prompt": system_prompt,
            "prompt": user_prompt,
            "completion": response,
            "reconstructed_state_day": x.get("reconstructed_state_day"),
            "reconstructed_role_label": x.get("reconstructed_role_label"),
            "self_reconstructed_role": x.get("self_reconstructed_role"),
            "internal_state_role_label": x.get("internal_state_role_label"),
            "internal_state_werewolf_candidates": x.get("internal_state_werewolf_candidates"),
            "speech_asymmetric_meta": x.get("speech_asymmetric_meta"),
            "label_reason": x.get("label_reason"),
        })
    return out


def build_state_recon_samples(samples):
    out = []
    for x in samples:
        if x.get("sample_type") != "state_recon":
            continue
        phase = str(x.get("phase", ""))
        # Save only day_pre state reconstruction samples to exported files.
        if "_day_pre_state_reconstruction" not in phase:
            continue
        prompt = x.get("prompt")
        completion = x.get("completion")
        if not isinstance(prompt, list) or not prompt:
            continue
        if not isinstance(completion, list) or not completion:
            continue

        out.append({
            "prompt": prompt,
            "completion": completion,
            "label": bool(x.get("label", True)),
            "role_label": x.get("role_label", {}),
            "phase": phase,
            "game_path": x.get("game_path", ""),
            "player_id": x.get("player_id"),
            "state_recon": None,
        })
    return out


def _normalize_completion_messages(raw_completion):
    if isinstance(raw_completion, list):
        msgs = []
        for msg in raw_completion:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip() or "assistant"
            msgs.append(create_message(role, to_text(msg.get("content", ""))))
        if msgs:
            return msgs

    if isinstance(raw_completion, dict):
        return [create_message("assistant", json.dumps(raw_completion, ensure_ascii=False))]

    text = to_text(raw_completion).strip()
    if not text:
        return []
    return [create_message("assistant", text)]


def _extract_assistant_text(messages):
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).strip().lower() != "assistant":
            continue
        return to_text(msg.get("content", "")).strip()
    return ""


def _looks_like_player_role_map(value):
    if not isinstance(value, dict) or not value:
        return False
    player_key_count = 0
    for k in value.keys():
        ks = str(k).strip()
        if re.fullmatch(r"Player\s+\d+", ks) or re.fullmatch(r"\d+", ks):
            player_key_count += 1
    return player_key_count > 0


def _extract_best_json_dict_from_text(text):
    if not text:
        return None

    # 1) Fast path: whole text itself is a dict.
    parsed_whole = parse_dict_maybe(text)
    if isinstance(parsed_whole, dict):
        return parsed_whole

    candidates = []

    # 2) Try fenced blocks first (```json ... ``` or ``` ... ```).
    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL):
        block = match.group(1).strip()
        parsed = parse_dict_maybe(block)
        if isinstance(parsed, dict):
            candidates.append(parsed)

    # 3) Scan raw text for JSON object substrings and decode each object.
    decoder = json.JSONDecoder()
    n = len(text)
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, end_idx = decoder.raw_decode(text[i:])
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        if end_idx <= 0 or i + end_idx > n:
            continue
        candidates.append(parsed)

    if not candidates:
        return None

    # Prefer dicts that look like state-reconstruction player maps.
    player_map_candidates = [c for c in candidates if _looks_like_player_role_map(c)]
    if player_map_candidates:
        return max(
            player_map_candidates,
            key=lambda d: (
                sum(
                    1
                    for k in d.keys()
                    if re.fullmatch(r"Player\s+\d+", str(k).strip()) or re.fullmatch(r"\d+", str(k).strip())
                ),
                len(d),
            ),
        )

    # Fallback: use the largest dict candidate.
    return max(candidates, key=lambda d: len(d))


def _canonical_completion_for_compare(messages):
    text = _extract_assistant_text(messages)
    if not text:
        return ""
    parsed = _extract_best_json_dict_from_text(text)
    if isinstance(parsed, dict):
        try:
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            pass
    return strip_code_fence(text).strip()


def _load_original_state_recon_entries_by_phase(game_path, player_id, cache):
    key = (os.path.abspath(game_path), str(player_id))
    if key in cache:
        return cache[key]

    phase_to_entry = {}
    player_file = os.path.join(key[0], f"Player_{key[1]}.jsonl")
    if os.path.isfile(player_file):
        with open(player_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                phase = str(obj.get("phase", ""))
                if "_day_pre_state_reconstruction" not in phase:
                    continue
                if phase in phase_to_entry:
                    continue
                if obj.get("response") is None:
                    continue
                phase_to_entry[phase] = {
                    "prompt": obj.get("prompt"),
                    "response": obj.get("response"),
                }

    cache[key] = phase_to_entry
    return phase_to_entry


def _normalize_role_map_keys(raw_role_map):
    if not isinstance(raw_role_map, dict):
        return {}
    normalized = {}
    for k, v in raw_role_map.items():
        m = re.search(r"\d+", str(k))
        if not m:
            continue
        role = normalize_role_label(v)
        normalized[m.group()] = role if role is not None else str(v).strip()
    return normalized


def _load_ground_truth_role_label(game_path, cache):
    game_path_abs = os.path.abspath(str(game_path))
    if game_path_abs in cache:
        return cache[game_path_abs]

    game_log_path = os.path.join(game_path_abs, "game_log.json")
    gt_role_label = {}
    if os.path.isfile(game_log_path):
        try:
            with open(game_log_path, "r", encoding="utf-8") as f:
                game_log = json.load(f)
            try:
                from MARBO.wolf.adversarial_data_extraction.utils_english import get_role_assignment
            except ModuleNotFoundError:
                from MARBO.wolf.adversarial_data_extraction.utils_english import get_role_assignment
            gt_role_label = _normalize_role_map_keys(get_role_assignment(game_log))
        except Exception:
            gt_role_label = {}

    cache[game_path_abs] = gt_role_label
    return gt_role_label


def build_state_recon_teacher_original_pairs(samples):
    kto_samples = []
    missing_original = 0
    skipped_identical_completion = 0
    original_cache = {}
    gt_role_label_cache = {}

    for x in samples:
        sample_type = x.get("sample_type")
        if sample_type is not None and sample_type != "state_recon":
            continue

        phase = str(x.get("phase", ""))
        if "_day_pre_state_reconstruction" not in phase:
            continue

        prompt = x.get("prompt")
        teacher_completion = x.get("completion")
        if not isinstance(prompt, list) or not prompt:
            continue
        if not isinstance(teacher_completion, list) or not teacher_completion:
            continue

        game_path = os.path.abspath(str(x.get("game_path", "")))
        if not game_path:
            continue

        player_id = x.get("player_id")
        try:
            player_id = int(player_id)
        except (TypeError, ValueError):
            continue

        phase_to_entry = _load_original_state_recon_entries_by_phase(
            game_path=game_path,
            player_id=player_id,
            cache=original_cache,
        )
        raw_entry = phase_to_entry.get(phase)
        if not raw_entry:
            missing_original += 1
            continue

        raw_prompt_text = to_text(raw_entry.get("prompt", "")).strip()
        original_prompt = prompt
        if raw_prompt_text:
            system_prompt = ""
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    system_prompt = to_text(msg.get("content", "")).strip()
                    if system_prompt:
                        break
            original_prompt = []
            if system_prompt:
                original_prompt.append(create_message("system", system_prompt))
            original_prompt.append(create_message("user", raw_prompt_text))

        original_completion = _normalize_completion_messages(raw_entry.get("response"))
        if not original_completion:
            missing_original += 1
            continue

        teacher_cmp = _canonical_completion_for_compare(teacher_completion)
        original_cmp = _canonical_completion_for_compare(original_completion)
        if teacher_cmp and original_cmp and teacher_cmp == original_cmp:
            skipped_identical_completion += 1
            continue

        is_day1_state_recon = phase.startswith("1_")
        if is_day1_state_recon:
            # For day1 reconstruction pairs, treat the day1 teacher sample as desirable
            # and mismatched original response as undesirable.
            teacher_label = True
            original_label = False
        else:
            teacher_label = bool(x.get("label", True))
            original_label = not teacher_label
        game_uid = x.get("game_uid") or RewardModel.build_game_uid(game_path)
        gt_role_label = _load_ground_truth_role_label(game_path, gt_role_label_cache)
        pair_role_label = gt_role_label if gt_role_label else x.get("role_label", {})
        pair_id = hashlib.sha1(
            f"{game_path}|{player_id}|{phase}|{teacher_cmp}|{original_cmp}".encode("utf-8")
        ).hexdigest()

        # Save as flat KTO-style samples so downstream formatter can read directly.
        kto_samples.append({
            "prompt": original_prompt,
            "completion": original_completion,
            "label": original_label,
            "role_label": pair_role_label,
            "phase": phase,
            "game_path": game_path,
            "game_uid": game_uid,
            "player_id": player_id,
            "state_recon": None,
            "pair_id": pair_id,
            "pair_source": "original_response",
        })
        kto_samples.append({
            "prompt": prompt,
            "completion": teacher_completion,
            "label": teacher_label,
            "role_label": pair_role_label,
            "phase": phase,
            "game_path": game_path,
            "game_uid": game_uid,
            "player_id": player_id,
            "state_recon": None,
            "pair_id": pair_id,
            "pair_source": "teacher_completion",
        })

    return kto_samples, missing_original, skipped_identical_completion


def _filter_state_recon_phase_map_to_day_pre(phase_map):
    """
    Keep only *_day_pre_state_reconstruction phases in exported state_recon phase maps.
    """
    filtered = {}
    for game_path, by_player in (phase_map or {}).items():
        for player_id, phases in (by_player or {}).items():
            kept = [p for p in phases if "_day_pre_state_reconstruction" in str(p)]
            if not kept:
                continue
            filtered.setdefault(game_path, {})[player_id] = kept
    return filtered


def main():
    args = parse_args()
    model = build_model_from_args(args)
    samples = model.build_kto_dataset(
        game_dir=args.game_dir,
        random_play=args.random_play,
        self_play=args.self_play,
        max_games=args.max_games,
    )
    split_maps = model.split_samples_by_type_label(samples)

    file_name_map = {
        "speech": {"good": args.speech_good_file, "bad": args.speech_bad_file},
        "vote": {"good": args.vote_good_file, "bad": args.vote_bad_file},
        "action": {"good": args.action_good_file, "bad": args.action_bad_file},
        "state_recon": {"good": args.recon_good_file, "bad": args.recon_bad_file},
    }
    split_maps_for_save = dict(split_maps)
    if "state_recon" in split_maps_for_save:
        split_maps_for_save["state_recon"] = {
            "good": _filter_state_recon_phase_map_to_day_pre(split_maps["state_recon"]["good"]),
            "bad": _filter_state_recon_phase_map_to_day_pre(split_maps["state_recon"]["bad"]),
        }

    saved = model.save_good_bad_phase_maps(
        out_to=os.path.abspath(args.out_to),
        split_maps=split_maps_for_save,
        categories=("speech", "vote", "action", "state_recon"),
        file_name_map=file_name_map,
    )

    for category in ("speech", "vote", "action", "state_recon"):
        info = saved.get(category, {})
        print(
            f"[{category}] good={info.get('good_count', 0)} -> {info.get('good_file')} | "
            f"bad={info.get('bad_count', 0)} -> {info.get('bad_file')}"
        )

    # Build and save vote/action state_recon companion files for format_training_data_2.py
    vote_state_recon_meta = {}
    action_state_recon_meta = {}
    vote_reason_meta = {}
    action_reason_meta = {}
    speech_reason_meta = {}
    for sample in samples:
        sample_type = sample.get("sample_type")
        game_path = os.path.abspath(str(sample.get("game_path", "")))
        player_id = str(sample.get("player_id", ""))
        phase = str(sample.get("phase", ""))
        if not game_path or not player_id or not phase:
            continue
        if sample_type in ("vote", "action"):
            sr = sample.get("state_recon")
            target_meta = vote_state_recon_meta if sample_type == "vote" else action_state_recon_meta
            target_meta.setdefault(game_path, {}).setdefault(player_id, {})[phase] = sr

        reason = sample.get("label_reason")
        if sample_type == "vote":
            vote_reason_meta.setdefault(game_path, {}).setdefault(player_id, {})[phase] = reason
        elif sample_type == "action":
            action_reason_meta.setdefault(game_path, {}).setdefault(player_id, {})[phase] = reason
        elif sample_type == "speech":
            speech_reason_meta.setdefault(game_path, {}).setdefault(player_id, {})[phase] = reason

    vote_meta_file = os.path.join(os.path.abspath(args.out_to), "vote_state_recon_meta.json")
    with open(vote_meta_file, "w", encoding="utf-8") as f:
        json.dump(vote_state_recon_meta, f, indent=4, ensure_ascii=False)
    print(f"[vote_state_recon_meta] saved -> {vote_meta_file}")

    action_meta_file = os.path.join(os.path.abspath(args.out_to), "action_state_recon_meta.json")
    with open(action_meta_file, "w", encoding="utf-8") as f:
        json.dump(action_state_recon_meta, f, indent=4, ensure_ascii=False)
    print(f"[action_state_recon_meta] saved -> {action_meta_file}")

    vote_reason_file = os.path.join(os.path.abspath(args.out_to), "vote_reason_meta.json")
    with open(vote_reason_file, "w", encoding="utf-8") as f:
        json.dump(vote_reason_meta, f, indent=4, ensure_ascii=False)
    print(f"[vote_reason_meta] saved -> {vote_reason_file}")

    action_reason_file = os.path.join(os.path.abspath(args.out_to), "action_reason_meta.json")
    with open(action_reason_file, "w", encoding="utf-8") as f:
        json.dump(action_reason_meta, f, indent=4, ensure_ascii=False)
    print(f"[action_reason_meta] saved -> {action_reason_file}")

    speech_reason_file = os.path.join(os.path.abspath(args.out_to), "speech_reason_meta.json")
    with open(speech_reason_file, "w", encoding="utf-8") as f:
        json.dump(speech_reason_meta, f, indent=4, ensure_ascii=False)
    print(f"[speech_reason_meta] saved -> {speech_reason_file}")

    good_speech_full = build_good_speech_full(samples)
    good_speech_full_path = os.path.join(os.path.abspath(args.out_to), args.good_speech_full_file)
    with open(good_speech_full_path, "w", encoding="utf-8") as f:
        json.dump(good_speech_full, f, indent=4, ensure_ascii=False)
    print(f"[speech_full] saved={len(good_speech_full)} -> {good_speech_full_path}")

    state_samples = build_state_recon_samples(samples)
    state_samples_path = os.path.join(os.path.abspath(args.out_to), args.state_samples_file)
    with open(state_samples_path, "w", encoding="utf-8") as f:
        json.dump(state_samples, f, indent=4, ensure_ascii=False)
    print(f"[state_samples] saved={len(state_samples)} -> {state_samples_path}")


if __name__ == "__main__":
    main()
