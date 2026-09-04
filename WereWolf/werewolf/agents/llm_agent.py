import json
import re
import logging

from MARBO.wolf.werewolf.agents.prompt_template_v1 import CON
from MARBO.wolf.werewolf.agents.base_agent import Agent
from MARBO.wolf.werewolf.helper.log_utils import JsonFormatter, CustomLoggerAdapter


class LLMAgent(Agent):
    def __init__(self,
                 client=None,
                 tokenizer=None,
                 llm=None,
                 temperature=1.0,
                 log_file=None):
        self.client = client
        self.tokenizer = tokenizer
        self.llm = llm
        self.nlp_action_to_env_action = {}
        self.temperature = temperature

        # Makto-style memory channels
        self.observed_notes = {}   # {day: str}
        self.belief_states = {}    # {day: {speech|vote|night: {"Player N": role}}}
        self.policy_memory = {}    # {day: {speech|vote|night: str}}
        self.alive = [f"Player {i}" for i in range(1, 10)]
        self.vote_reason = {}

        if log_file is not None:
            self.has_log = True
            self.handler = logging.FileHandler(log_file)
            self.handler.setLevel(logging.INFO)
            self.handler.setFormatter(JsonFormatter())
            logger = logging.getLogger(log_file.split("/")[-1].replace(".jsonl", ""))
            logger.setLevel(logging.INFO)
            logger.addHandler(self.handler)
            self.logger = CustomLoggerAdapter(logger, extra={})
        else:
            self.has_log = False

    # =========================
    # Memory Helpers
    # =========================
    def _phase_to_key(self, phase: str) -> str:
        phase_l = str(phase).lower()
        if "night" in phase_l or "skill" in phase_l:
            return "night"
        if "vote" in phase_l:
            return "vote"
        return "speech"

    def append_observed_note(self, day: int, note_text: str):
        note_text = (note_text or "").strip()
        if not note_text:
            return
        prev = self.observed_notes.get(day, "").strip()
        if prev:
            self.observed_notes[day] = prev + "\n" + note_text
        else:
            self.observed_notes[day] = note_text

    def parse_state_reconstruction(self, raw_action: str) -> dict:
        if not isinstance(raw_action, str):
            raise ValueError("state reconstruction response is not a string")

        text = raw_action.strip()
        if not text:
            raise ValueError("empty state reconstruction response")

        candidates = [text]
        for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE):
            candidates.append(m.group(1).strip())

        content = None
        for c in candidates:
            try:
                parsed = json.loads(c)
                if isinstance(parsed, dict):
                    content = parsed
                    break
            except Exception:
                continue

        if content is None:
            raise ValueError("no JSON object found in state reconstruction response")

        valid_roles = {
            "Villager", "Werewolf", "Seer", "Witch", "Guard", "Hunter", "Unknown",
            "村民", "狼人", "预言家", "女巫", "守卫", "猎人", "未知",
        }
        role_alias = {
            "村民": "Villager",
            "狼人": "Werewolf",
            "预言家": "Seer",
            "女巫": "Witch",
            "守卫": "Guard",
            "猎人": "Hunter",
            "未知": "Unknown",
        }

        parsed = {}
        for k, v in content.items():
            key = str(k).strip()
            if not key.startswith("Player "):
                continue
            val = str(v).strip()
            val = role_alias.get(val, val)
            if val not in valid_roles:
                val = "Unknown"
            parsed[key] = val

        return parsed

    def _store_belief_state(self, phase: str, parsed_belief: dict):
        day = int(str(phase).split("_")[0])
        phase_key = self._phase_to_key(phase)
        if day not in self.belief_states:
            self.belief_states[day] = {}
        self.belief_states[day][phase_key] = parsed_belief

    def labels_to_belief_lines(self, role_map: dict):
        lines = []

        def sort_key(item):
            m = re.findall(r"\d+", item[0])
            return int(m[0]) if m else 999

        for player, role in sorted(role_map.items(), key=sort_key):
            if role == "Unknown":
                lines.append(f"- {player}: identity remains uncertain.")
            else:
                lines.append(f"- {player}: currently predicted as {role}.")
        return lines

    def build_policy_memory(self, day: int, phase_key: str) -> str:
        observed = self.observed_notes.get(day, "").strip()
        role_map = self.belief_states.get(day, {}).get(phase_key, {})

        lines = [f"Day {day} Summary:"]
        if observed:
            lines.append("[Observed]")
            lines.append(observed)

        if role_map:
            lines.append("[Belief Update]")
            lines.extend(self.labels_to_belief_lines(role_map))

        return "\n".join(lines).strip()

    def update_policy_memory(self, phase: str):
        day = int(str(phase).split("_")[0])
        phase_key = self._phase_to_key(phase)
        if day not in self.policy_memory:
            self.policy_memory[day] = {}
        self.policy_memory[day][phase_key] = self.build_policy_memory(day, phase_key)

    def _get_policy_note_logs(self, current_phase: str) -> str:
        current_day = int(str(current_phase).split("_")[0])
        current_key = self._phase_to_key(current_phase)
        chunks = []

        for day in sorted(self.policy_memory.keys()):
            if day < current_day:
                for phase_key in ["speech", "vote", "night"]:
                    content = self.policy_memory.get(day, {}).get(phase_key, "").strip()
                    if content:
                        chunks.append(content)
                        break
            elif day == current_day:
                content = self.policy_memory.get(day, {}).get(current_key, "").strip()
                if content:
                    chunks.append(content)

        return "\n\n".join(chunks).strip()

    def _get_recon_note_logs(self, current_phase: str) -> str:
        current_day = int(str(current_phase).split("_")[0])
        chunks = []

        for day in sorted(self.observed_notes.keys()):
            if day <= current_day:
                content = self.observed_notes.get(day, "").strip()
                if content:
                    chunks.append(f"Day {day} Observed Notes:\n{content}")

        return "\n\n".join(chunks).strip()

    # =========================
    # Log Parsing Helpers
    # =========================
    def _parse_log_time(self, time_text):
        pattern = r"(?:Day\s+|第)(\d+)(?:\s+|天)(Daytime|Nighttime|白天|夜晚)"
        match = re.search(pattern, str(time_text))
        if not match:
            return None, None
        matched_day = int(match.group(1))
        time_of_day = "day" if match.group(2) in ["Daytime", "白天"] else "night"
        return matched_day, time_of_day

    def parse_vote_info(self, game_log, vote_at_day):
        def sort_candidates(candidates):
            return sorted(candidates, key=lambda candidate: int(re.findall(r"\d+", candidate)[0]))

        vote_info = ""
        pk_player = []
        vote_out_player = None

        for log in game_log:
            log_turn, _ = self._parse_log_time(getattr(log, "time", ""))
            if log_turn != vote_at_day or "vote" not in getattr(log, "event", ""):
                continue

            if log.event == "vote":
                if log.target > 0:
                    vote_info += f"Player {log.source} voted for Player {log.target};\n"
                else:
                    vote_info += f"Player {log.source} abstained;\n"
            elif log.event == "vote_pk":
                if log.target > 0:
                    vote_info += f"PK Phase, Player {log.source} voted for Player {log.target};\n"
                else:
                    vote_info += f"PK Phase, Player {log.source} abstained;\n"
            elif log.event == "end_vote":
                outcome = log.content.get("vote_outcome")
                if outcome == "all abstention":
                    vote_info += "Result: All players abstained, entering night phase directly.\n"
                elif outcome == "all abstention in pk":
                    vote_info += "Result: All players abstained, entering night phase directly.\n"
                elif outcome == "draw":
                    pk_speech_list = ", ".join(f"Player {idx}" for idx in log.content.get("speech_queue", []))
                    pk_vote_list = ", ".join(f"Player {idx}" for idx in log.content.get("vote_queue", []))
                    vote_info += (
                        f"Result: Draw, players {pk_speech_list} will speak again for PK, "
                        f"and players {pk_vote_list} will vote.\n"
                    )
                    pk_player = sort_candidates(pk_speech_list.split(", ")) if pk_speech_list else []
                elif outcome == "draw in pk":
                    vote_info += "Result: Another draw, entering night phase directly.\n"
                elif isinstance(outcome, int):
                    expelled = log.content.get("expelled")
                    vote_info += f"Result: Player {expelled} was voted out.\n"
                    vote_out_player = expelled

        return vote_info, vote_out_player, pk_player

    # =========================
    # Prompt Formatting (Makto-aligned)
    # =========================
    def _process_speaker_order(self, first_speaker, all_speaker):
        first_index = all_speaker.index(f"Player {first_speaker}")
        speak_order = all_speaker[first_index:] + all_speaker[:first_index]
        return speak_order

    def _format_log_with_notes(self, current_phase, game_log):
        if "state_reconstruction" in current_phase:
            note_logs = self._get_recon_note_logs(current_phase)
        else:
            note_logs = self._get_policy_note_logs(current_phase)

        wolf_team_info = ""
        night_log_tmp = ""
        wolf_killed_at_night = ""
        night_action_log_tmp = ""
        speech_log_tmp = ""
        previous_speaker = []
        pk_speech_log_tmp = ""
        previous_speaker_pk = []
        current_turn = int(current_phase.split("_")[0])

        for log in game_log:
            log_turn, _ = self._parse_log_time(getattr(log, "time", ""))
            if log_turn is None:
                continue

            if log.event == "werewolf_team_info" and "Werewolves are: " not in wolf_team_info:
                wolf_team = ""
                for idx in log.content.get("wolf_team", []):
                    wolf_team += f"{idx},"
                wolf_team = wolf_team[:-1]
                if wolf_team:
                    wolf_team_info += f"- Werewolves are: Player {wolf_team}.\n"

            if log.event == "end_night":
                dead_list = ""
                for idx in log.content.get("dead_list", []):
                    dead_list += f"Player {idx}, "
                    if f"Player {idx}" in self.alive:
                        self.alive.remove(f"Player {idx}")
                if len(dead_list) > 0:
                    dead_list = dead_list[:-2]
                    night_log_tmp += f"Round {log_turn + 1}, player {dead_list} died; "
                else:
                    night_log_tmp += f"Round {log_turn + 1} was a peaceful night, no one died; "

            elif log_turn < current_turn or ("night" in current_phase and log_turn == current_turn):
                if log.event == "kill_decision":
                    night_action_log_tmp += (
                        f"Round {log_turn + 1}, the Werewolf faction chose to kill Player {log.target}.\n"
                    )
                elif log.event == "skill_seer":
                    checked_identity = log.content.get("checked_identity", log.content.get("cheked_identity"))
                    if checked_identity == "bad":
                        night_action_log_tmp += (
                            f"Round {log_turn + 1}, the Seer checked Player {log.target}, and "
                            f"Player {log.target} is a Werewolf.\n"
                        )
                    else:
                        night_action_log_tmp += (
                            f"Round {log_turn + 1}, the Seer checked Player {log.target}, and "
                            f"Player {log.target} is not a Werewolf.\n"
                        )
                elif log.event == "skill_guard":
                    if log.target != 0:
                        night_action_log_tmp += f"Round {log_turn + 1}, the Guard chose to protect Player {log.target}.\n"
                    else:
                        night_action_log_tmp += f"Round {log_turn + 1}, the Guard chose not to protect anyone.\n"
                elif log.event == "skill_witch":
                    if "heal" in log.content:
                        if log.target != 0:
                            night_action_log_tmp += f"Round {log_turn + 1}, the Witch used a healing potion on Player {log.target}.\n"
                        else:
                            night_action_log_tmp += f"Round {log_turn + 1}, the Witch did not use a healing potion.\n"
                    elif "poison" in log.content:
                        if log.target != 0:
                            night_action_log_tmp += f"Round {log_turn + 1}, the Witch used a poison potion on Player {log.target}.\n"
                        else:
                            night_action_log_tmp += f"Round {log_turn + 1}, the Witch did not use a poison potion.\n"
                    elif "pass" in log.content:
                        night_action_log_tmp += (
                            f"Round {log_turn + 1}, the Witch did not use a healing potion; "
                            f"the Witch did not use a poison potion.\n"
                        )
                elif log.event == "skill_hunter" and log.target != 0:
                    if log_turn == current_turn:
                        night_action_log_tmp += (
                            f"Player {log.source} is the Hunter; after being killed by werewolves on Night "
                            f"{current_turn}, they chose to shoot Player {log.target}.\n"
                        )
                    else:
                        night_action_log_tmp += (
                            f"Player {log.source} is the Hunter; after being voted out on Day {log_turn}, "
                            f"they chose to shoot Player {log.target}.\n"
                        )

            if "night" in current_phase and log_turn == current_turn:
                if log.event == "skill_wolf":
                    if "Werewolves voted to kill targets: " not in night_log_tmp:
                        night_log_tmp += "Werewolves voted to kill targets: "
                    night_log_tmp += f"Player {log.source} chose to kill Player {log.target}; "
                elif log.event == "kill_decision":
                    wolf_killed_at_night = f"Tonight, Player {log.target} died, "

            if log_turn == current_turn:
                if log.event == "speech":
                    previous_speaker.append(log.source)
                    if len(log.content.get("speech_content", "")) > 0:
                        speech_text = log.content.get("speech_content")
                        if isinstance(speech_text, dict):
                            speech_text = speech_text.get(
                                "content",
                                speech_text.get("Speech", speech_text.get("speech", str(speech_text))),
                            )
                        speech_log_tmp += f"**Player {log.source}**：{str(speech_text).strip()}\n"
                    else:
                        speech_log_tmp += f"**Player {log.source}**： Empty.\n"
                elif log.event == "speech_pk":
                    previous_speaker_pk.append(log.source)
                    if len(log.content.get("speech_content", "")) > 0:
                        speech_text = log.content.get("speech_content")
                        if isinstance(speech_text, dict):
                            speech_text = speech_text.get(
                                "content",
                                speech_text.get("Speech", speech_text.get("speech", str(speech_text))),
                            )
                        pk_speech_log_tmp += f"**Player {log.source}**：{str(speech_text).strip()}\n"
                    else:
                        pk_speech_log_tmp += f"**Player {log.source}**：Empty.\n"

        return (
            note_logs,
            night_log_tmp,
            night_action_log_tmp,
            wolf_team_info,
            wolf_killed_at_night,
            speech_log_tmp,
            previous_speaker,
            pk_speech_log_tmp,
            previous_speaker_pk,
        )

    def _format_objective_info(self, phase, observation, note_logs, night_obs, night_action_log,
                               wolf_team_info, prev_speaker, prev_speaker_pk, vote_info, pk_players):
        day = int(phase.split("_")[0])
        objective_info = ""

        if "skill" in phase:
            objective_info = f"- Game Progress: Round {day + 1}.\n"
            objective_info += f"- Currently alive players: {', '.join(self.alive)}. "

            if observation["identity"] == "Werewolf":
                objective_info += "You can only choose to kill from the players above.\n"
                if len(wolf_team_info) > 0:
                    objective_info += wolf_team_info
                if len(night_obs.strip()) > 0:
                    objective_info += night_obs.strip() + "\n"
                else:
                    objective_info += "You are the first werewolf to act; please choose your target.\n"
            elif observation["identity"] == "Guard":
                objective_info += "You can only choose to protect from the players above.\n"
                if len(night_action_log.strip()) > 0:
                    objective_info += f"- Action Record: {night_action_log}\n"
            elif observation["identity"] == "Seer":
                objective_info += "You can only choose to check from the players above.\n"
                if len(night_action_log.strip()) > 0:
                    objective_info += f"- Action Record: {night_action_log}\n"
            elif observation["identity"] == "Witch":
                if len(night_action_log.strip()) > 0:
                    objective_info += f"\n- Action Record: {night_action_log}\n"

        elif "speech_pk" in phase:
            objective_info = f"- Game Progress: Round {day} PK phase"
            if len(pk_players) > 0:
                objective_info += f", players on PK stage: {', '.join(pk_players)}"
            objective_info += "\n"
            if observation["identity"] == "Werewolf" and len(wolf_team_info) > 0:
                objective_info += wolf_team_info
            objective_info += f"- Currently alive players: {', '.join(self.alive)}.\n"
            if len(night_action_log.strip()) > 0:
                objective_info += f"- Action Record: {night_action_log}\n"
            if len(pk_players) > 0:
                if len(prev_speaker_pk) == 0:
                    speaker_order_pk = self._process_speaker_order(observation["current_act_idx"], pk_players)
                else:
                    first_speaker_pk = prev_speaker_pk[0]
                    speaker_order_pk = self._process_speaker_order(first_speaker_pk, pk_players)
                objective_info += f"- Speaking order in PK phase: {'; '.join(speaker_order_pk)}\n"
            objective_info += f"- Night Information: {night_obs}\n"

        elif "speech" in phase:
            objective_info = f"- Game Progress: Round {day}.\n"
            if observation["identity"] == "Werewolf" and len(wolf_team_info) > 0:
                objective_info += wolf_team_info
            objective_info += f"- Currently alive players: {', '.join(self.alive)}.\n"
            if len(night_action_log.strip()) > 0:
                objective_info += f"- Action Record: {night_action_log}\n"
            if len(prev_speaker) == 0:
                speaker_order = self._process_speaker_order(observation["current_act_idx"], self.alive)
            else:
                first_speaker = prev_speaker[0]
                speaker_order = self._process_speaker_order(first_speaker, self.alive)
            objective_info += f"- Speaking order for this round: {'; '.join(speaker_order)}\n"
            objective_info += f"- Night Information: {night_obs}\n"

        elif "vote" in phase:
            objective_info = f"- Game Progress: Round {day}.\n"
            if observation["identity"] == "Werewolf" and len(wolf_team_info) > 0:
                objective_info += wolf_team_info + "\n"
            objective_info += f"- Currently alive players: {', '.join(self.alive)}.\n"
            if len(night_action_log.strip()) > 0:
                objective_info += f"- Action Record: {night_action_log}\n"
            objective_info += f"- Night Information: {night_obs}\n"

        elif "state_reconstruction" in phase:
            objective_info = f"- Game Progress: Round {day}.\n"
            if observation["identity"] == "Werewolf" and len(wolf_team_info) > 0:
                objective_info += wolf_team_info
            objective_info += f"- Currently alive players: {', '.join(self.alive)}.\n"
            if len(night_action_log.strip()) > 0:
                objective_info += f"- Action Record: {night_action_log}\n"
            objective_info += f"- Night Information: {night_obs}\n"

        if len(vote_info.strip()) > 0:
            objective_info += f"- Voting Status: {vote_info}\n"

        return objective_info

    def _format_subjective_info(self, phase, observation, note_logs, speech_log, pk_speech_log):
        day = int(phase.split("_")[0])
        subjective_info = ""

        if "skill" in phase:
            if day == 0:
                return "None"
            if len(note_logs.strip()) > 0:
                subjective_info += note_logs + "\n"
            subjective_info += f"- Speeches of all players in Round {day}:\n"
            subjective_info += speech_log + "\n"

        elif "speech_pk" in phase:
            if len(note_logs.strip()) > 0:
                subjective_info += note_logs + "\n"
            subjective_info += f"- Speeches of all players in this round (Round {day}):\n{speech_log}\n"
            if len(pk_speech_log.strip()) > 0:
                subjective_info += f"- Speeches of players in current PK phase:\n{pk_speech_log}\n"
            else:
                subjective_info += "- Current PK phase, you are the first to speak.\n"

        elif "speech" in phase:
            if len(note_logs.strip()) > 0:
                subjective_info += note_logs + "\n"
            subjective_info += (
                f"- Currently in Round {day}, speeches of players before you in this round:\n{speech_log}\n"
            )

        elif "vote_pk" in phase:
            if len(note_logs.strip()) > 0:
                subjective_info += note_logs + "\n"
            subjective_info += f"- Speeches of all players in this round (Round {day}):\n{speech_log}\n"
            if len(pk_speech_log.strip()) > 0:
                subjective_info += f"- Speeches of all players in Round {day} PK phase:\n{pk_speech_log}\n"
            if day in self.vote_reason:
                vote_reason_in_normal_vote = self.vote_reason[day]
                subjective_info += f"- Your voting reason in Round {day}:\n{vote_reason_in_normal_vote}\n"

        elif "vote" in phase:
            if len(note_logs.strip()) > 0:
                subjective_info += note_logs + "\n"
            subjective_info += f"- Speeches of all players in this round:\n{speech_log}\n"

        elif "state_reconstruction" in phase:
            if "night" in phase:
                if len(note_logs.strip()) > 0:
                    subjective_info += note_logs + "\n"
                else:
                    subjective_info = "None"
            else:
                if len(note_logs.strip()) > 0:
                    subjective_info += note_logs + "\n"
                if len(speech_log.strip()) > 0:
                    subjective_info += f"- Speeches of all players in this round (Round {day}):\n{speech_log}\n"

        return subjective_info

    def format_observation(self, observation):
        phase = observation["phase"]
        day = int(observation["phase"].split("_")[0]) + 1
        identity = observation["identity"]

        # keep nlp_action_to_env_action mapping updated for agents relying on it (e.g. GPTAgent)
        if "skill" in phase or "vote" in phase:
            valid_actions = observation.get("valid_action", [])
            self.get_valid_actions_str(valid_actions)

        roles = observation.get("roles")
        if isinstance(roles, list) and len(roles) > 0:
            self.alive = [f"Player {i}" for i in range(1, len(roles) + 1)]
        elif not self.alive:
            self.alive = [f"Player {i}" for i in range(1, 10)]

        identity_info = CON.player_identity_info.format(
            player_idx=observation["current_act_idx"],
            identity=identity,
            identity_ability=CON.identity_abilities[identity],
        )
        brief_identity_description = f"You are currently Player {observation['current_act_idx']} ({identity})."

        note_logs, night_obs, night_action_log, wolf_team_info, wolf_killed_at_night, \
            speech_log, prev_speaker, pk_speech_log, prev_speaker_pk = self._format_log_with_notes(
                phase, observation["game_log"]
            )

        vote_info = ""
        pk_players_at_day_i = []

        for day_i in range(1, day - 1):
            vote_info += f"Round {day_i} Voting Records: "
            vote_info_at_day_i, vote_out_at_day_i, pk_players_at_day_i = self.parse_vote_info(
                observation["game_log"], day_i
            )
            vote_info += vote_info_at_day_i

            if vote_out_at_day_i is not None and f"Player {vote_out_at_day_i}" in self.alive:
                self.alive.remove(f"Player {vote_out_at_day_i}")

        if "pk" in phase or "skill" in phase:
            vote_info_last_day, vote_out_last_day, pk_players_at_day_i = self.parse_vote_info(
                observation["game_log"], day - 1
            )
            if vote_info_last_day != "":
                vote_info += f"Round {day - 1} Voting Records: "
            vote_info += vote_info_last_day
            if vote_out_last_day is not None and f"Player {vote_out_last_day}" in self.alive:
                self.alive.remove(f"Player {vote_out_last_day}")

        objective_info = self._format_objective_info(
            phase, observation, note_logs, night_obs, night_action_log,
            wolf_team_info, prev_speaker, prev_speaker_pk, vote_info, pk_players_at_day_i,
        )
        subjective_info = self._format_subjective_info(
            phase, observation, note_logs, speech_log, pk_speech_log,
        )

        if "skill" in phase:
            instruction = ""
            if identity == "Werewolf":
                instruction = CON.werewolf_skill_prompt_v3
            elif identity == "Seer":
                instruction = CON.seer_skill_prompt_v3
            elif identity == "Witch":
                instruction = CON.witch_skill_prompt_v3.format(wolf_killed_info=wolf_killed_at_night)
            elif identity == "Guard":
                instruction = CON.guard_skill_prompt_v3
            elif identity == "Hunter":
                instruction = CON.hunter_skill_prompt_v3

            prompt = CON.skill_prompt_v3.format(
                player_identity_info=identity_info,
                objective_info=objective_info,
                subjective_info=subjective_info,
                your_role=brief_identity_description,
                instruction_prompt=instruction,
            )

        elif "speech" in phase:
            prompt = CON.speech_prompt_v3.format(
                player_identity_info=identity_info,
                objective_info=objective_info,
                subjective_info=subjective_info,
                your_role=brief_identity_description,
            )

        elif "vote" in phase:
            prompt = CON.vote_prompt_v3.format(
                player_identity_info=identity_info,
                objective_info=objective_info,
                subjective_info=subjective_info,
                your_role=brief_identity_description,
            )

        elif "state_reconstruction" in phase:
            prompt = CON.state_reconstruction_sample_v3.format(
                player_identity_info=identity_info,
                objective_info=objective_info,
                subjective_info=subjective_info,
                your_role=brief_identity_description,
            )
        else:
            raise ValueError(f"Unsupported phase: {phase}")

        return prompt

    # =========================
    # Legacy log formatter (kept for compatibility)
    # =========================
    def _translate_time(self, time_str):
        m = re.search(r"第(\d+)天(白天|夜晚)", time_str)
        if m:
            day, period = m.group(1), m.group(2)
            period_en = "Day" if period == "白天" else "Night"
            return f"Day {day} {period_en}"
        return time_str

    def _print_log(self, log):
        print("===============")
        print(log.event)
        print(log.viewer)
        print(log.source)
        print(log.target)
        print(log.content)
        print(log.time)
        print("===============\n")

    def format_log(self, game_log):
        logs = ""
        for log in game_log:
            log_tmp = ""
            t = self._translate_time(log.time)
            if log.event == "game_setting":
                log_tmp = "This game has the following roles and corresponding numbers: \n"
                for key, value in log.content.items():
                    log_tmp += "- {}:{}\n".format(key, value)
            if log.event == "skill_wolf":
                log_tmp = "{} is a wolf, he plans to kill {} at {}".format(log.source, log.target, t)
            elif log.event == "kill_decision":
                log_tmp = "The wolf team killed {} at {}".format(log.target, t)
            elif log.event == "skill_seer":
                checked_identity = log.content.get("checked_identity", log.content.get("cheked_identity"))
                if checked_identity == "bad":
                    checked_result = "wolf"
                elif checked_identity == "good":
                    checked_result = "good"
                else:
                    checked_result = "unknown"
                log_tmp = "{} is a seer, he checked the identity of {} at {}. The result is {}.".format(
                    log.source, log.target, t, checked_result
                )
            elif log.event == "skill_guard":
                log_tmp = "{} is a guard, he protected {} at {}".format(log.source, log.target, t)
            elif log.event == "skill_witch":
                if "heal" in log.content:
                    log_tmp = "{} is a witch, he healed {} at {}".format(log.source, log.target, t)
                elif "poison" in log.content:
                    log_tmp = "{} is a witch, he poisoned {} at {}".format(log.source, log.target, t)
            elif log.event == "skill_hunter":
                log_tmp = "{} is a hunter, he shot {} at {}".format(log.source, log.target, t)
            elif log.event == "speech" or log.event == "speech_pk":
                if len(log.content["speech_content"]) > 0:
                    log_tmp = "{} said at {} : {}".format(log.source, t, log.content["speech_content"])
                else:
                    log_tmp = "{} said at {} : empty".format(log.source, t)
            elif log.event == "vote":
                if log.target > 0:
                    log_tmp = "{} voted for {} at {}".format(log.source, log.target, t)
                else:
                    log_tmp = "{} abstained from voting at {}".format(log.source, t)
            elif log.event == "vote_pk":
                if log.target > 0:
                    log_tmp = "{} voted for {} at {}".format(log.source, log.target, t)
                else:
                    log_tmp = "{} abstained from voting in pk at {}".format(log.source, t)
            elif log.event == "end_game":
                log_tmp = "Game over!\n"
            elif log.event == "end_night":
                dead_list = ""
                for idx in log.content["dead_list"]:
                    dead_list += "{}, ".format(idx)
                if len(dead_list) > 0:
                    dead_list = dead_list[:-1]
                    log_tmp = "{} death players are {}".format(t, dead_list)
                else:
                    log_tmp = "{} no one died".format(t)
            elif log.event == "end_vote":
                if log.content["vote_outcome"] == "all abstention":
                    log_tmp = "{} all players abstained voting, directly enter the night.".format(t)
                elif log.content["vote_outcome"] == "all abstention in pk":
                    log_tmp = "{} again spoke, all players abstained voting, directly enter the night.".format(t)
                elif log.content["vote_outcome"] == "draw":
                    pk_speech_list = ""
                    for idx in log.content["speech_queue"]:
                        pk_speech_list += "{}, ".format(idx)
                    pk_speech_list = pk_speech_list[:-1]

                    pk_vote_list = ""
                    for idx in log.content["vote_queue"]:
                        pk_vote_list += "{}, ".format(idx)
                    pk_vote_list = pk_vote_list[:-1]
                    log_tmp = "{} draw, {} again spoke, {} voted.\n".format(t, pk_speech_list, pk_vote_list)
                elif log.content["vote_outcome"] == "draw in pk":
                    log_tmp = "{} again spoke, draw, directly enter the night.\n".format(t)
                elif type(log.content["vote_outcome"]) == int:
                    log_tmp = "{} voted out {}\n".format(t, log.content["expelled"])
                else:
                    raise ValueError
            elif log.event == "werewolf_team_info":
                wolf_team = ""
                for idx in log.content["wolf_team"]:
                    wolf_team += "{}, ".format(idx)
                wolf_team = wolf_team[:-1]
                log_tmp = "Wolf team members are {}\n".format(wolf_team)
            elif log.event == "self_identity":
                pass
            logs += log_tmp

        return logs

    def get_valid_actions_str(self, valid_actions):
        valid_actions_str = ""
        for action in valid_actions:
            if action[0] == "kill":
                if action[1] == 0:
                    valid_actions_str += "- {'kill':'no'}\n"
                else:
                    valid_actions_str += "- {{'kill':'{0}'}}\n".format(action[1])
            elif action[0] == "check":
                if action[1] == 0:
                    valid_actions_str += "- {'check':'no'}\n"
                else:
                    valid_actions_str += "- {{'check':'{0}'}}\n".format(action[1])
            elif action[0] == "guard":
                if action[1] == 0:
                    valid_actions_str += "- {'guard':'no'}\n"
                else:
                    valid_actions_str += "- {{'guard':'{0}'}}\n".format(action[1])
            elif "witch" in action[0]:
                if action[0] == "witch_pass":
                    valid_actions_str += "- {'heal':'no', 'poison':'no'}\n"
                elif action[0] == "witch_poison":
                    valid_actions_str += "- {{'heal':'no', 'poison':'{0}'}}\n".format(action[1])
                elif action[0] == "witch_heal":
                    valid_actions_str += "- {{'heal':'{0}', 'poison':'no'}}\n".format(action[1])
            elif action[0] == "shoot":
                if action[1] == 0:
                    valid_actions_str += "- {'shoot':'no'}\n"
                else:
                    valid_actions_str += "- {{'shoot':'{0}'}}\n".format(action[1])
            elif action[0] == "vote" or action[0] == "vote_pk":
                if action[1] == 0:
                    valid_actions_str += "- {'vote':'abstain'}\n"
                else:
                    valid_actions_str += "- {{'vote':'{0}'}}\n".format(action[1])

        self.nlp_action_to_env_action = {}
        for (nlp_action, env_action) in zip(valid_actions_str.split("\n"), valid_actions):
            self.nlp_action_to_env_action[nlp_action[2:]] = env_action

        return valid_actions_str

    def reset(self):
        self.observed_notes = {}
        self.belief_states = {}
        self.policy_memory = {}
        self.vote_reason = {}
        self.alive = [f"Player {i}" for i in range(1, 10)]

    def act(self, observation):
        raise NotImplementedError
