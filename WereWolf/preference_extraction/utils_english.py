import os
import re
import json

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
}

game_description_9p = """In this game, we have 9 players numbered from 1 to 9: 6 Villagers and 3 Werewolves. 
Among the  Villagers, there are special roles, including:
1 Seer:  
- Objective: The Seer’s purpose is to help the Villagers identify the Werewolves. 
- Ability: During  the night phase, the Seer can secretly choose one player and learn their true identity (whether  they are a Werewolf or not) each night.
1 Witch:  
- Objective: The Witch’s purpose is to strategically use her special abilities to help the  Villagers.  
- Abilities: The Witch has one healing potion and one poison potion. Once used, they cannot be  used again in subsequent rounds. The Witch cannot use both the healing potion and the poison  potion on the same night. The healing potion can be used to save a player who was killed by  the Werewolves during the night. The poison potion can eliminate a player who is likely to be a  Werewolf
{god_description} The rest are ordinary Villagers.
"""

game_description_7p = """In this game, we have 7 players numbered from 1 to 7 — 5 villagers and 2 werewolves. 
Among the villagers, there are special roles including:
- 1 Seer:
- Objective: The Seer's purpose is to help villagers identify werewolves.
- Ability: During the night phase, the Seer can secretly choose one player and learn their true identity (whether they are a werewolf or not) each night.。
{god_description} The rest are ordinary Villagers.。"""

guard_description = """- 1 Guard:
    - Objective: The Guard's goal is to strategically use his special ability to assist the Villagers.
    - Ability: The Guard can protect one player each night, preventing them from being attacked by Werewolves. The Guard can choose to protect himself or choose not to protect anyone, but he cannot protect the same player for two consecutive nights.
"""

hunter_description = """- 1 Hunter:
    - Objective: The Hunter's goal is to strategically use his special ability to assist the Villagers in eliminating Werewolves.
    - Ability: When the Hunter is killed by Werewolves or voted out during the day, he can reveal his identity and fire a revenge bullet at any living player, causing that player to die with him. The Hunter can choose not to reveal his identity, but if he does, he must choose a target. (Note: If the Hunter is poisoned by the Witch, he cannot reveal his identity or shoot anyone).
"""

witch_description = """- 1 Witch:
    - Objective: The Witch's goal is to strategically use her special abilities to assist the Villagers.
    - Ability: The Witch has one antidote and one poison. Once used, they cannot be used again. The Witch cannot use both the antidote and the poison in the same night. The antidote can be used to save a player killed by Werewolves at night. The poison can be used to eliminate a player suspected of being a Werewolf.
"""

def get_system_prompt(game_type):
    if game_type == "9p_seer_witch_guard":
        return game_description_9p.format(god_description=guard_description)
    elif game_type == "9p_seer_witch_hunter":
        return game_description_9p.format(god_description=hunter_description)
    elif game_type == "7p_seer_witch":
        return game_description_7p.format(god_description=witch_description)
    elif game_type == "7p_seer_guard":
        return game_description_7p.format(god_description=guard_description)
    return game_description_9p.format(god_description=guard_description)

def get_game_description(game_type):
    game_desc = get_system_prompt(game_type)
    game_desc = game_desc.replace("The rules and instructions for the Werewolf game are as follows:\n", "")

    target_str_en = "You are currently playing a game called 'Werewolf'.\nIn this game,"
    replace_str_en = "The rules and instructions for the Werewolf game are as follows:\n"
    
    game_desc = game_desc.replace(target_str_en, replace_str_en)
    return game_desc


def judge_models(model_setting, sft_model_regx):
    if "_tmp0.5" in model_setting:
        model_setting = model_setting.replace("_tmp0.5", "") # we only consider tmp=0.5,
    werewolf_model_name = model_setting.split("_vs_")[0].split("-")[1]
    good_model_name = model_setting.split("_vs_")[1].split("-")[1]
    
    werewolf_flag = False
    good_flag = False

    for sft_type in sft_model_regx.split("|"):
        if sft_type == werewolf_model_name:
        # if sft_type in werewolf_model_name:
            werewolf_flag = True
        if sft_type == good_model_name:
        # if sft_type in good_model_name:
            good_flag = True
    return (werewolf_model_name, werewolf_flag), (good_model_name, good_flag)


def get_role_assignment(log):
    for i in log:
        event = i["event"]
        if event == "god_view":
            role_assignment = i["content"] # "1"->"Villager"/"Werewolf"/"Seer"/"Witch"/"Guard"/"Hunter"
            return role_assignment
    return None


def create_message(role, content):
    return {
        "role": role,
        "content": content
    }

def extract_prediction(key_target, response):
    key_start_index = response.find(key_target)
    clean_content = None
    if key_start_index != -1:
        value_start_index = response.find("{", key_start_index)
        
        
        if value_start_index != -1:
            bracket_count = 0
            value_end_index = -1
            
            for i, char in enumerate(response[value_start_index:]):
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                
                if bracket_count == 0:
                    value_end_index = value_start_index + i + 1 
                    break
            
            extracted_content = response[value_start_index:value_end_index]


            clean_content = extracted_content.replace('\\"', '"')

    
    return clean_content

def to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts = []
        for item in x:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(x, dict):
        return str(x.get("content", "")) if "content" in x else json.dumps(x, ensure_ascii=False)
    if x is None:
        return ""
    return str(x)

def normalize_role_terms_to_english(text):
    if not isinstance(text, str) or not text:
        return text
    for src, dst in ROLE_TO_EN.items():
        if src in ["Villager", "Werewolf", "Seer", "Witch", "Guard", "Hunter"]:
            continue
        text = text.replace(src, dst)
    return text

def get_assignment_role(game_path, role_cache):
    if game_path in role_cache:
        return role_cache[game_path]
    game_log_path = os.path.join(game_path, "game_log.json")
    if not os.path.exists(game_log_path):
        role_cache[game_path] = {}
        return role_cache[game_path]
    with open(game_log_path, "r", encoding="utf-8") as f:
        log = json.load(f)
    role_map = get_role_assignment(log) or {}
    role_map = {str(k): ROLE_TO_EN.get(v, v) for k, v in role_map.items()}
    role_cache[game_path] = role_map
    return role_map

def resolve_game_path(raw_game_path, fallback_game_path=""):
    gp = (raw_game_path or fallback_game_path or "").strip()
    if gp:
        gp_abs = os.path.abspath(gp)
        if os.path.exists(gp_abs):
            return gp_abs

        game_dir_name = os.path.basename(gp_abs.rstrip(os.sep))
        if game_dir_name.startswith("game_"):
            current_parent = os.path.dirname(gp_abs)
            while True:
                next_parent = os.path.dirname(current_parent)
                if next_parent == current_parent:
                    break
                candidate = os.path.join(next_parent, game_dir_name)
                if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "game_log.json")):
                    return os.path.abspath(candidate)
                current_parent = next_parent

        return gp_abs
    return ""
 
def _parse_response_json(response_text):
    """Try to parse response as JSON, including markdown code block format."""
    try:
        return json.loads(response_text)
    except Exception:
        pass
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass
    return None


def _lookup_role_by_vote_target(game_role_label, vote_target):
    if not isinstance(game_role_label, dict):
        return ""
    token = str(vote_target).strip()
    if not token:
        return ""

    # exact key first
    if token in game_role_label:
        return str(game_role_label.get(token, ""))

    # support "Player N" / "N" mixed formats
    m = re.search(r"\d+", token)
    if m:
        pid = m.group()
        if pid in game_role_label:
            return str(game_role_label.get(pid, ""))
        player_key = f"Player {int(pid)}"
        if player_key in game_role_label:
            return str(game_role_label.get(player_key, ""))
    return ""


def _lookup_role_by_target(game_role_label, target):
    """Lookup role by flexible target token: '3', 'Player 3', etc."""
    return _lookup_role_by_vote_target(game_role_label, target)


def _normalize_vote_target_token(raw_target):
    """Normalize vote target token to comparable id-ish string; empty means abstain/no-target."""
    if raw_target is None:
        return ""
    text = str(raw_target).strip()
    if not text:
        return ""
    if text.lower() in ("abstain", "0", "no", "none", "-1", "null", ""):
        return ""
    m = re.search(r"\d+", text)
    return m.group() if m else text


def label_from_vote_response(
    response_text,
    game_role_label,
    voter_role=None,
    eliminated_target=None,
    seer_vote_target=None,
    villager_majority_target=None,
    has_true_seer=None,
):
    """
    Determine vote label from model response.

    Backward-compatible behavior (when no extra context is passed):
      True  → voted target is predicted as Werewolf in `game_role_label`
      False → voted non-Werewolf or abstained
      None  → parse failed

    Reward-aligned behavior (when context is passed):
      - Abstain => False
      - If true Seer exists: not voting with Seer target => False
      - Else (no Seer): not voting with villager-majority target => False
      - Special roles (Seer/Witch/Guard/Hunter) voting Werewolf => True
      - Villager voting Werewolf and that target is actually expelled && Vote for werewolf => True
      - Villager voting expelled non-Werewolf => False
    """
    try:
        resp = _parse_response_json(response_text)
        if not isinstance(resp, dict):
            return None
        # Support both key formats
        voting_player = str(resp.get("voting_player", resp.get("Vote", resp.get("vote", "")))).strip()
        voting_target_norm = _normalize_vote_target_token(voting_player)
        if not voting_target_norm:
            return False

        # state_recon signal / legacy mode: desirable iff voted target is predicted as Werewolf
        target_role = _lookup_role_by_vote_target(game_role_label, voting_player)
        legacy_mode = (
            voter_role is None
            and eliminated_target is None
            and seer_vote_target is None
            and villager_majority_target is None
            and has_true_seer is None
        )
        if legacy_mode:
            return target_role == "Werewolf"

        # Normalize role & context targets
        voter_role_norm = ROLE_TO_EN.get(str(voter_role).strip(), str(voter_role).strip()) if voter_role is not None else None
        seer_target_norm = _normalize_vote_target_token(seer_vote_target)
        majority_target_norm = _normalize_vote_target_token(villager_majority_target)
        expelled_target_norm = _normalize_vote_target_token(eliminated_target)

        # Coordination rules
        if has_true_seer and seer_target_norm and voting_target_norm != seer_target_norm:
            return False
        if (has_true_seer is False) and majority_target_norm and voting_target_norm != majority_target_norm:
            return False

        # Special roles: desirable if they vote Werewolf
        if voter_role_norm in _GOD_ROLES:
            return target_role == "Werewolf"

        # Villager-specific outcome rules
        if voter_role_norm == "Villager":
            if target_role == "Werewolf" and expelled_target_norm and voting_target_norm == expelled_target_norm:
                return True
            if expelled_target_norm and voting_target_norm == expelled_target_norm and target_role and target_role != "Werewolf":
                return False
            return False

        # Fallback for other non-werewolf roles (if any custom role names appear)
        if target_role == "Werewolf":
            if expelled_target_norm:
                return voting_target_norm == expelled_target_norm
            return True
        return False
    except Exception:
        return None


_GOD_ROLES = {"Seer", "Witch", "Guard", "Hunter"}

_NO_TARGET = ("no", "0", "none", "")


def extract_night_action_target(phase, response_text):
    """
    Parse the target player ID string from a night skill action response.
    Returns the raw target string (e.g. "3"), or None if no target / parse failure.

    Key mapping per phase:
      night_skill_wolf  → "kill"
      night_skill_seer  → "check"
      night_skill_witch → "poison"
      night_skill_guard → "protected_player" / "guard"
      night_skill_hunter → "shoot" / "target"
    """
    try:
        resp = _parse_response_json(response_text)
        if not isinstance(resp, dict):
            return None
        if "night_skill_wolf" in phase:
            t = str(resp.get("kill", "")).strip()
        elif "night_skill_seer" in phase:
            t = str(resp.get("check", "")).strip()
        elif "night_skill_witch" in phase:
            t = str(resp.get("poison", "no")).strip()
        elif "night_skill_guard" in phase:
            t = str(resp.get("protected_player", resp.get("guard", "no"))).strip()
        elif "night_skill_hunter" in phase:
            t = str(resp.get("shoot", resp.get("target", resp.get("kill", "no")))).strip()
        else:
            return None
        return None if t.lower() in _NO_TARGET else t
    except Exception:
        return None


def label_from_action_response(phase, response_text, game_role_label, game_type="9p_seer_witch_guard"):
    """
    Determine label from night skill / state_reconstruction phase response.
    True  → desirable action
    False → undesirable action
    None  → cannot determine (skip, use file-based label)

    Rules per phase (reward-aligned):
      night_skill_wolf:
        day >= 2: kill a special role (Seer/Witch/Guard/Hunter) → True
        day >= 1: kill nobody → False
      night_skill_seer:
        identify a Werewolf → True
      night_skill_witch:
        day == 1: heal someone → True; no heal → False
        day >= 2: poison a Werewolf → True; poison villager-side → False
      night_skill_guard:
        day >= 2: protect special role → True; protect Werewolf → False
      night_skill_hunter:
        eliminate Werewolf → True; eliminate special role → False
      state_reconstruction:
        predicted identity accuracy >= 60% → True; else False
    """
    try:
        # Extract day number from phase prefix (e.g. "2_night_skill_wolf" → 2)
        day = int(phase.split("_")[0])
        resp = _parse_response_json(response_text)

        # ── Wolf ──────────────────────────────────────────────────────────────
        if "night_skill_wolf" in phase:
            if not isinstance(resp, dict):
                return None
            kill_target = str(resp.get("kill", "")).strip()
            if not kill_target or kill_target.lower() in ("no", "0", "none", ""):
                return False if day >= 1 else None
            if day >= 2:
                target_role = _lookup_role_by_target(game_role_label, kill_target)
                return True if target_role in _GOD_ROLES else None
            return None  # day 0 not labeled

        # ── Seer ──────────────────────────────────────────────────────────────
        elif "night_skill_seer" in phase:
            if not isinstance(resp, dict):
                return None
            check_target = str(resp.get("check", "")).strip()
            if not check_target or check_target.lower() in ("no", "0", "none", ""):
                return None
            target_role = _lookup_role_by_target(game_role_label, check_target)
            return True if target_role == "Werewolf" else None

        # ── Witch ─────────────────────────────────────────────────────────────
        elif "night_skill_witch" in phase:
            if not isinstance(resp, dict):
                return None
            heal = str(resp.get("heal", "no")).strip().lower()
            poison = str(resp.get("poison", "no")).strip().lower()
            if day == 1:
                # Good: used healing potion on someone
                return heal not in ("no", "0", "none", "")
            else:
                # Good: poisoned a Werewolf
                if poison not in ("no", "0", "none", ""):
                    target_role = _lookup_role_by_target(game_role_label, poison)
                    if target_role == "Werewolf":
                        return True
                    if target_role:
                        return False
                    return None
                return None  # did nothing — ambiguous

        # ── Guard ─────────────────────────────────────────────────────────────
        elif "night_skill_guard" in phase:
            if not isinstance(resp, dict):
                return None
            protect = str(resp.get("protected_player", resp.get("guard", "no"))).strip().lower()
            if day < 2:
                return None
            if protect in ("no", "0", "none", ""):
                return None
            target_role = _lookup_role_by_target(game_role_label, protect)
            if not target_role:
                return None
            if target_role == "Werewolf":
                return False
            if target_role in _GOD_ROLES:
                return True
            return None

        # ── Hunter ────────────────────────────────────────────────────────────
        elif "night_skill_hunter" in phase:
            if not isinstance(resp, dict):
                return None
            shoot = str(resp.get("shoot", resp.get("target", resp.get("kill", "no")))).strip().lower()
            if shoot in ("no", "0", "none", ""):
                return None
            target_role = _lookup_role_by_target(game_role_label, shoot)
            if not target_role:
                return None
            if target_role == "Werewolf":
                return True
            if target_role in _GOD_ROLES:
                return False
            return None

        # ── State reconstruction ───────────────────────────────────────────────
        elif "state_reconstruction" in phase:
            if resp is None:
                return None
            # resp may be a dict of {player_key: role} or list → dict
            if isinstance(resp, list):
                return None
            if not isinstance(resp, dict):
                return None
            correct = 0
            total = 0
            for player_key, predicted in resp.items():
                # Normalize key: "Player 1" or "1" → "1"
                num = str(player_key).replace("Player", "").strip()
                actual = game_role_label.get(num, "")
                if not actual:
                    continue
                total += 1
                pred_norm = ROLE_TO_EN.get(str(predicted).strip(), str(predicted).strip())
                if pred_norm == actual:
                    correct += 1
            if total == 0:
                return None
            return (correct / total) >= 0.6

    except Exception:
        return None

    return None
