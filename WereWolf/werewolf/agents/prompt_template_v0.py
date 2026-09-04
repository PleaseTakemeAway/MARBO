class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.%s" % name)
        self.__dict__[name] = value

CON = Const()

# =============================================================================
# Game Descriptions
# =============================================================================

CON.game_description = """You are now playing a game called ’Werewolf’ (also known as ’Mafia’). 
In this game, players are typically divided into two factions: Werewolves and Villagers. 
Different roles in the Werewolf game have different objectives: 
- The Villagers’ goal is to identify the Werewolves and eliminate them through voting. 
- For the Werewolves, their main objective is to hide their true identities, mislead others during discussions to avoid being voted out, and hunt down as many Villagers as possible. 
Here are some basic rules: 
- Identity: Players’ identities are secretly assigned. Werewolves know each other’s identities, while Villagers only know their own. 
- Day and Night Cycles: The game alternates between day and night phases. At night, Werewolves secretly choose a Villager to eliminate. During the day, all players discuss and vote on who they believe is a Werewolf, and the player with the most votes is eliminated. 
- Special Roles: There are some roles with special abilities in the game, such as the ’Seer’ who can learn players’ identities. 
- Winning Conditions: The game ends when one group achieves its winning conditions. If all Werewolves are eliminated, the Villagers win. If the Werewolves kill all ordinary Villagers or all special roles, the Werewolves win.
"""

CON.game_description_9p = """In this game, we have 9 players numbered from 1 to 9: 6 Villagers and 3 Werewolves. 
Among the  Villagers, there are special roles, including:
1 Seer:  
- Objective: The Seer’s purpose is to help the Villagers identify the Werewolves. 
- Ability: During  the night phase, the Seer can secretly choose one player and learn their true identity (whether  they are a Werewolf or not) each night.
1 Witch:  
- Objective: The Witch’s purpose is to strategically use her special abilities to help the  Villagers.  
- Abilities: The Witch has one healing potion and one poison potion. Once used, they cannot be  used again in subsequent rounds. The Witch cannot use both the healing potion and the poison  potion on the same night. The healing potion can be used to save a player who was killed by  the Werewolves during the night. The poison potion can eliminate a player who is likely to be a  Werewolf
{god_description} The rest are ordinary Villagers.
"""

CON.game_description_7p = """In this game, we have 7 players numbered from 1 to 7 — 5 villagers and 2 werewolves. 
Among the villagers, there are special roles including:
- 1 Seer:
- Objective: The Seer's purpose is to help villagers identify werewolves.
- Ability: During the night phase, the Seer can secretly choose one player and learn their true identity (whether they are a werewolf or not) each night.。
{god_description} The rest are ordinary Villagers.。"""

CON.guard_description="""- 1 Guard:  
    - Objective: The Guard’s purpose is to strategically use his special ability to help the Villagers.  
    - Ability: The Guard can protect one player each night from Werewolf attacks. The Guard can choose  to protect himself or choose not to protect anyone, but he cannot protect the same player for  two consecutive nights.
"""

CON.hunter_description = """- 1 Hunter:
    - Objective: The Hunter's purpose is to strategically use his special ability to help villagers eliminate werewolves.
    - Ability: When the Hunter is killed by werewolves or voted out during the day, they can reveal their identity card and shoot a revenge bullet at any living player, taking that player down with them. The Hunter can choose not to reveal their card, but once they reveal it, they must take someone with them (Note: if the Hunter is killed by the Witch's poison, they cannot reveal their card and take someone with them).
"""

CON.witch_description = """- 1 Witch:
    - Objective: The Witch's purpose is to strategically use her special abilities to help villagers.
    - Ability: The Witch has one healing potion and one poison potion. Once used, they cannot be used in subsequent rounds. The Witch cannot use both the healing potion and poison potion in the same night. The healing potion can save a player who was killed by werewolves during the night. The poison potion can eliminate a player who is likely to be a werewolf.  
"""

# =============================================================================
# Role Metadata
# =============================================================================

CON.identity_chinese = {
    "Seer": "预言家",
    "Witch": "女巫",
    "Guard": "守卫",
    "Villager": "村民",
    "Werewolf": "狼人",
    "Hunter": "猎人"
}

CON.identity_abilities = {
    "Seer": "Each night, you can check if one player is a werewolf. Your goal is to use this information to help others find and eliminate all werewolves.",
    "Witch": "You have one healing potion and one poison potion. Your goal is to use these abilities strategically to protect villagers or eliminate werewolves.",
    "Guard": "Each night, you can protect one player from being killed by werewolves. Your goal is to use this ability wisely to reduce villager casualties.",
    "Villager": "You have no special abilities, but your goal is to observe, discuss, and vote to reveal werewolves' identities while trying to survive.",
    "Werewolf": "You need to cooperate with other werewolves to choose a villager to kill each night. Your goal is to hide your identity and mislead other players until the werewolves achieve victory.",
    "Hunter": "When killed by werewolves or voted out during the day, you can choose to reveal your identity card and take one surviving player with you. You can choose not to reveal your card, but if you do reveal it, you must take someone with you. Your goal is to use this ability wisely to help villagers eliminate werewolves."
}

CON.player_identity_info = """
You are Player {player_idx}.
Your identity is: {identity}.
{identity_ability}"""

# =============================================================================
# Shared Prompt Blocks
# =============================================================================

_INFO_BLOCK = """In this game, you currently have the following information:
1. Role setting：
{player_identity_info}
2. Objective Information：
{objective_info}
3. Subjective Information：
{subjective_info}
"""

_SPEECH_INSTRUCTIONS = """Analyze the situation using all information above, then write only the public message you want other players to hear.
Treat role setting, hidden-team knowledge, and private actions as private strategic context; reveal, imply, hide, or fabricate role-related information only when it helps your public position.
Make the public message sound grounded in public discussion, observed behavior, voting logic, and plausible inference, not in private notes or system-provided labels.
Output one JSON object with exactly the keys: "Identity to present", "Vote", "Speech".
"""

_SPEECH_WITH_BELIEF_INSTRUCTIONS = """Analyze the situation using all information above, then write only the public message you want other players to hear.
Treat role setting, hidden-team knowledge, private actions, and [Belief Update] notes as private strategic context; reveal, imply, hide, or fabricate role-related information only when it helps your public position.
Make the public message sound grounded in public discussion, observed behavior, voting logic, and plausible inference, not in private notes or system-provided labels.
When making any role-related claim, consider timing, credibility, survival risk, and whether the claim helps your side.
Output one JSON object with exactly the keys: "Identity to present", "Vote", "Speech".
"""

_VOTE_INSTRUCTIONS = """Use objective information, player statements, voting records, and your private role information as strategic context. Treat subjective information as possibly deceptive.
Choose a currently living player whose elimination best serves your role's objective; output "abstain" if voting is strategically worse than waiting.
Do not explicitly mention private or system-provided labels in the public voting reason.
Output one JSON object with the keys "notes", "voting reason", and "voting player". For "voting player", output a player number or "abstain".
"""

_VOTE_WITH_BELIEF_INSTRUCTIONS = """Use objective information, player statements, voting records, private role information, and [Belief Update] notes as strategic context. Treat subjective information as possibly deceptive.
Choose a currently living player whose elimination best serves your role's objective; output "abstain" if voting is strategically worse than waiting.
Do not explicitly mention private belief-state predictions in the public voting reason.
Output one JSON object with the keys "notes", "voting reason", and "voting player". For "voting player", output a player number or "abstain".
"""

_STATE_RECONSTRUCTION_INSTRUCTIONS = """{your_role}
Please predict the identity labels for all players based on the role setup, objective information, and subjective information (note that objective information is always true, while subjective information may contain deceptive speeches).
The Subjective Information may contain [Observed Notes] from prior rounds and new speeches from the current round.
Treat [Observed Notes] as your prior assessment — use it as a starting point, but revise it in light of new objective facts and current speeches. If new evidence contradicts a prior assessment, update your prediction accordingly.
If you don't know about a player's identity, please output "Unknown".
Output role labels only; do not use life-state labels such as "Dead", "Deceased", or "Eliminated".
Please output in JSON format using the keyword "Player N" for each player.
"""

# =============================================================================
# Night Skill Instructions
# =============================================================================

CON.werewolf_skill_prompt_v3 = """Please combine the role settings, objective information, and subjective information (objective information is always true, while subjective information is not necessarily true) to determine the true identities of the good people present. Choose the player you want to kill and output the result in JSON format with the keywords 'kill' and 'reason'. Output the player number directly. If you choose not to kill anyone, output 'no'."""
CON.seer_skill_prompt_v3 = """Please combine the role settings, objective information, and subjective information (objective information is always true, while subjective information contains deceptive content) to analyze potential alliances based on voting patterns and further analyze players' hidden true identities. Choose the player you want to check and output the result in JSON format with the keywords 'check' and 'reason'. Output the player number directly."""
CON.guard_skill_prompt_v3 = """Please combine the role settings, objective information, and subjective information (objective information is always true, while subjective information contains deceptive content) to analyze potential alliances and players' identities, and then choose the player you want to protect. Output the result in JSON format with the keywords 'guard' and 'reason'. Output the player number directly. If you choose not to protect anyone, output 'no'."""
CON.witch_skill_prompt_v3 = """Please combine the role settings, objective information, and subjective information (objective information is always true, while subjective information may contain deceptive content) to choose your night action. Decide whether to use your healing potion and poison potion:
- {wolf_killed_info}Do you want to use the healing potion? If yes, output 'yes', otherwise output 'no'.
- Do you want to use the poison potion? If yes, output the player number directly, otherwise output 'no'. Output the result in JSON format with the keywords 'heal' and 'poison'."""
CON.hunter_skill_prompt_v3 = """Please combine the role settings, objective information, and subjective information (objective information is always true, while subjective information is not necessarily true) to determine the true identities of the players. Choose the player you want to shoot and output the result in JSON format with the keywords 'shoot' and 'reason'. Output the player number directly. If you choose not to shoot anyone, output 'no'."""

# =============================================================================
# Action Prompts
#
# Default prompts are used by API / opensource / Makto baseline agents.
# Belief-aware prompts are used only by makto_sr_ours.
# =============================================================================

CON.skill_prompt = _INFO_BLOCK + """
{your_role}{instruction_prompt}
"""

CON.speech_prompt = _INFO_BLOCK + "\n" + _SPEECH_INSTRUCTIONS
CON.speech_prompt_with_belief = _INFO_BLOCK + "\n" + _SPEECH_WITH_BELIEF_INSTRUCTIONS
CON.vote_prompt = _INFO_BLOCK + "\n" + _VOTE_INSTRUCTIONS
CON.vote_prompt_with_belief = _INFO_BLOCK + "\n" + _VOTE_WITH_BELIEF_INSTRUCTIONS
CON.state_reconstruction_prompt = _INFO_BLOCK + _STATE_RECONSTRUCTION_INSTRUCTIONS

CON.skill_prompt_v3 = CON.skill_prompt
CON.speech_prompt_v3 = CON.speech_prompt
CON.vote_prompt_v3 = CON.vote_prompt
CON.speech_prompt_with_belief_v3 = CON.speech_prompt_with_belief
CON.vote_prompt_with_belief_v3 = CON.vote_prompt_with_belief
CON.state_reconstruction_sample_v3 = CON.state_reconstruction_prompt
