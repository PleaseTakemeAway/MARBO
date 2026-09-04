ROLE_NAMES = ["Villager", "Werewolf", "Seer", "Witch", "Guard", "Hunter", "Unknown"]


def _json_schema_response(name, schema):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
        },
    }


def _player_value(description):
    return {
        "type": "string",
        "description": description,
    }


def _choice_value(choices, description):
    return {
        "type": "string",
        "enum": choices,
        "description": description,
    }


def _string_field(description):
    return {
        "type": "string",
        "description": description,
    }


def _speech_schema():
    return {
        "type": "object",
        "properties": {
            "Identity to Present": _string_field("The identity the player wants to present publicly."),
            "Identity Labels": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Player identity labels, keyed by player name such as Player 1.",
            },
            "Vote": _string_field("The final vote direction, usually a player number or abstain."),
            "Speech": _string_field("The exact speech to say in the current round."),
        },
        "required": ["Identity to Present", "Vote", "Speech"],
        "additionalProperties": False,
    }

def _vote_schema():
    return {
        "type": "object",
        "properties": {
            "notes": _string_field("Private notes and current-round analysis."),
            "voting reason": _string_field("Reason for the vote choice."),
            "voting player": _string_field('Player number as a string, or "abstain".'),
        },
        "required": ["notes", "voting reason", "voting player"],
        "additionalProperties": False,
    }


def _state_reconstruction_schema(player_count):
    player_count = int(player_count or 9)
    properties = {
        f"Player {idx}": {
            "type": "string",
            "enum": ROLE_NAMES,
        }
        for idx in range(1, player_count + 1)
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def _action_values(valid_action, action_type):
    values = [
        str(action_value)
        for action_name, action_value in valid_action or []
        if action_name == action_type and action_value != 0
    ]
    return sorted(set(values), key=lambda value: int(value) if value.isdigit() else value)


def _witch_action_pairs(valid_action):
    pairs = []
    seen = set()
    for action_name, action_value in valid_action or []:
        if action_name == "witch_pass":
            pair = ("no", "no")
        elif action_name == "witch_heal" and action_value != 0:
            pair = (str(action_value), "no")
        elif action_name == "witch_poison" and action_value != 0:
            pair = ("no", str(action_value))
        else:
            continue

        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)
    return pairs


def _witch_pair_schema(heal, poison):
    return {
        "type": "object",
        "properties": {
            "heal": {"const": heal},
            "poison": {"const": poison},
        },
        "required": ["heal", "poison"],
        "additionalProperties": False,
    }


def _skill_schema(identity, valid_action=None):
    identity = str(identity or "").lower()
    action_pair_schemas = []
    if identity == "werewolf":
        properties = {
            "kill": _player_value('Player number as a string, or "no".'),
            "reason": _string_field("Reason for the kill choice."),
        }
        required = ["kill", "reason"]
    elif identity == "seer":
        properties = {
            "check": _player_value("Player number as a string."),
            "reason": _string_field("Reason for the check choice."),
        }
        required = ["check", "reason"]
    elif identity == "guard":
        properties = {
            "guard": _player_value('Player number as a string, or "no".'),
            "reason": _string_field("Reason for the guard choice."),
        }
        required = ["guard", "reason"]
    elif identity == "witch":
        heal_choices = ["no"] + _action_values(valid_action, "witch_heal")
        poison_choices = ["no"] + _action_values(valid_action, "witch_poison")
        properties = {
            "heal": _choice_value(
                heal_choices,
                'Player number to heal as a string, or "no". Do not output "yes"; do not heal and poison in the same action.',
            ),
            "poison": _choice_value(
                poison_choices,
                'Player number to poison as a string, or "no". Do not heal and poison in the same action.',
            ),
        }
        required = ["heal", "poison"]
        action_pair_schemas = [
            _witch_pair_schema(heal, poison)
            for heal, poison in _witch_action_pairs(valid_action)
        ]
    elif identity == "hunter":
        properties = {
            "shoot": _player_value('Player number as a string, or "no".'),
            "reason": _string_field("Reason for the shoot choice."),
        }
        required = ["shoot", "reason"]
    else:
        return None

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    if action_pair_schemas:
        schema["oneOf"] = action_pair_schemas
    return schema


def get_response_format(phase, identity=None, player_count=9, valid_action=None):
    phase = str(phase or "").lower()
    if "speech" in phase:
        return _json_schema_response("werewolf-speech", _speech_schema())
    if "vote" in phase:
        return _json_schema_response("werewolf-vote", _vote_schema())
    if "state_reconstruction" in phase:
        return _json_schema_response(
            "werewolf-state-reconstruction",
            _state_reconstruction_schema(player_count),
        )
    if "skill" in phase:
        schema = _skill_schema(identity, valid_action)
        if schema is not None:
            return _json_schema_response(f"werewolf-{str(identity).lower()}-skill", schema)
    return None
