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

CON.werewolf_skill_prompt_v3 = """First update your private belief about every non-werewolf player's likely identity from the objective information, subjective information, and prior [Belief Update] notes. Objective information is always true; subjective information may be false or deceptive. Then choose the kill action that best advances the Werewolf team's win condition. Output the result in JSON format with the keywords 'kill' and 'reason'. Output the player number directly. If you choose not to kill anyone, output 'no'."""
CON.seer_skill_prompt_v3 = """First update your private belief about every player's likely identity from the objective information, subjective information, voting patterns, and prior [Belief Update] notes. Objective information is always true; subjective information may contain deceptive content. Then choose the check action that would most improve your belief state and help the Villagers eliminate Werewolves. Output the result in JSON format with the keywords 'check' and 'reason'. Output the player number directly."""
CON.guard_skill_prompt_v3 = """First update your private belief about every player's likely identity from the objective information, subjective information, and prior [Belief Update] notes. Objective information is always true; subjective information may contain deceptive content. Then choose the guard action that best protects the Villager side according to your updated belief state. Output the result in JSON format with the keywords 'guard' and 'reason'. Output the player number directly. If you choose not to protect anyone, output 'no'."""
CON.witch_skill_prompt_v3 = """First update your private belief about every player's likely identity from the objective information, subjective information, and prior [Belief Update] notes. Objective information is always true; subjective information may contain deceptive content. Then decide whether to use your healing potion and poison potion according to your updated belief state:
- {wolf_killed_info}Do you want to use the healing potion? If yes, output 'yes', otherwise output 'no'.
- Do you want to use the poison potion? If yes, output the player number directly, otherwise output 'no'. Output the result in JSON format with the keywords 'heal' and 'poison'."""
CON.hunter_skill_prompt_v3 = """First update your private belief about every player's likely identity from the objective information, subjective information, and prior [Belief Update] notes. Objective information is always true; subjective information may be false or deceptive. Then choose the shoot action that best advances the Villager side according to your updated belief state. Output the result in JSON format with the keywords 'shoot' and 'reason'. Output the player number directly. If you choose not to shoot anyone, output 'no'."""

CON.player_identity_info = """
You are Player {player_idx}.
Your identity is: {identity}.
{identity_ability}"""

CON.werewolf_team_info = """
Known Werewolf Team Information:
{wolf_team_info}
"""


CON.skill_prompt = """
** Game Description
{game_description}
{player_identity_info}

** Game Logs
{logs}

Based on the game logs, select one action you want to perform from the list below.
{valid_actions}

Please analyze the current situation and output your decision in JSON format using the same schema as the MaKTO Agent.
- If you are a Werewolf, use the keys "kill" and "reason".
- If you are a Seer, use the keys "check" and "reason".
- If you are a Guard, use the keys "guard" and "reason".
- If you are a Witch, use the keys "heal", "poison", and optionally "reason".
- If you are a Hunter, use the keys "shoot" and "reason".
Choose only from the valid actions listed above.

** Output
"""

CON.speech_prompt = """
** Game Description 
{game_description}
{player_identity_info}

** Game Logs
{logs}

Please analyze the current situation based on the game logs, summarize your speaking intentions, and organize your speech for this round.
"""

CON.state_reconstruction_prompt = """
** Game Description
{game_description}
{player_identity_info}
{werewolf_team_info}

** Game Logs
{logs}

Based on the game logs, predict the identity labels for all players using the same format as the MaKTO Agent.
If you are a Werewolf, you must preserve the known identities of your werewolf teammates in your reconstruction.
If you do not know a player's identity, output "Unknown".
Please output in JSON format using the keyword "Player N" for each player.
"""

CON.vote_prompt = """
** Game Description
{game_description}
{player_identity_info}

** Game Logs
* public logs
{logs}

Based on the game logs, select one action you want to perform from the list below.
{valid_actions}

For "voting player", choose only from the valid actions listed above. If abstaining, output "Abstain".
"""

# ======================================================================
# ReCon Agent Prompts
# ======================================================================
# ReCon-style prompts are designed around situation reconstruction,
# perspective-taking, conflict/deception analysis, and strategic action
# selection. They avoid explicit full role-label prediction as a required
# output, so they can be used as a fair ReCon baseline rather than a
# belief-supervised MaKTO/MARBO-style agent.

CON.state_reconstruction_sample_v3 = """In this game, you currently have the following information:

1. Role Setting:
{player_identity_info}

2. Objective Information:
{objective_info}

3. Subjective Information:
{subjective_info}

You are currently {your_role}.

You are a ReCon-style Werewolf agent. Before making a decision, reconstruct the current game situation from multiple perspectives.

Step 1. Public Fact Reconstruction:
Summarize only the objective facts that are publicly confirmed, such as deaths, votes, claims, revealed information, and current alive players.

Step 2. Self-Perspective Reasoning:
Analyze the situation from your own role objective. Identify which players are likely allies, enemies, or uncertain.

Step 3. Other-Perspective Reasoning:
Infer how other players may perceive you and each other based on speeches, votes, claims, and night outcomes.

Step 4. Conflict and Deception Analysis:
Identify suspicious contradictions, fake claims, vote inconsistencies, or possible deception.

Step 5. Strategic Summary:
Summarize the best strategy for the current phase.

Output the result in JSON format using the following keys:
"public_facts", "self_perspective", "other_perspectives", "conflict_analysis", "strategic_summary".
"""

CON.speech_prompt_v3 = """In this game, you currently have the following information:

1. Role Setting:
{player_identity_info}

2. Objective Information:
{objective_info}

3. Subjective Information:
{subjective_info}

You are currently {your_role}.

Before speaking, act as a ReCon-style agent and reconstruct the game situation from multiple perspectives.

First, reason about:
- What is publicly known?
- What do you privately know?
- How are other players likely to interpret your previous behavior?
- Which players are likely to trust or suspect you?
- Which claim, accusation, defense, or uncertainty expression best serves your role objective?

Then generate a speech that is strategically useful, socially plausible, and consistent with the public game context.

Important rules:
- Do not reveal private information unless doing so is strategically beneficial or required by your role.
- If you are a Werewolf, avoid exposing your team and use plausible reasoning to redirect suspicion.
- If you are a Villager-side role, use evidence from claims, votes, deaths, and contradictions to guide discussion.
- Avoid overconfident accusations without evidence.
- Keep the speech natural and persuasive.

Output in JSON format using the following keys:
"reconstruction", "speaking_intention", "Identity to Present", "Vote", "Speech".
"""

CON.vote_prompt_v3 = """In this game, you currently have the following information:

1. Role Setting:
{player_identity_info}

2. Objective Information:
{objective_info}

3. Subjective Information:
{subjective_info}

You are currently {your_role}.

Before voting, act as a ReCon-style agent and reconstruct the situation from both your own perspective and other players' perspectives.

Consider:
- Which players are most suspicious based on speeches, votes, claims, and night outcomes?
- Which vote is most consistent with your public reasoning?
- How will other players interpret your vote?
- Does your vote help your faction achieve its win condition?
- If you are a Werewolf, does the vote help protect the Werewolf team or mislead the Villagers?
- If you are a Villager-side role, does the vote help eliminate a likely Werewolf?

Choose only from the valid actions listed below:
{valid_actions}

If abstaining, output "abstain".

Output in JSON format using the following keys:
"reconstruction", "voting reason", "voting player".
The "voting player" value must be a player number or "abstain".
"""

CON.skill_prompt_v3 = """In this game, you currently have the following information:

1. Role Setting:
{player_identity_info}

2. Objective Information:
{objective_info}

3. Subjective Information:
{subjective_info}

You are currently {your_role}.

Before choosing a night action, act as a ReCon-style agent and reconstruct the situation from multiple perspectives.

Consider:
- What is objectively known from previous rounds?
- What do you privately know because of your role?
- What are other players likely to believe?
- Which players are likely allies, enemies, or uncertain?
- Which action best serves your faction's win condition?

Choose only from the valid actions listed below:
{valid_actions}

Role-specific output schema:
- If you are a Werewolf, output {"reconstruction": "...", "kill": "player_number_or_no", "reason": "..."}.
- If you are a Seer, output {"reconstruction": "...", "check": "player_number", "reason": "..."}.
- If you are a Guard, output {"reconstruction": "...", "guard": "player_number_or_no", "reason": "..."}.
- If you are a Witch, output {"reconstruction": "...", "heal": "yes_or_no", "poison": "player_number_or_no", "reason": "..."}.
- If you are a Hunter, output {"reconstruction": "...", "shoot": "player_number_or_no", "reason": "..."}.

Do not choose actions outside the valid action list.
"""

# Optional aliases for codebases that expect generic ReCon prompt names.
CON.recon_prompt = CON.state_reconstruction_sample_v3
CON.recon_action_prompt = CON.skill_prompt_v3
