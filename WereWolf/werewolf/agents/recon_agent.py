import random
import re
import json
import traceback
from MARBO.wolf.werewolf.agents.prompt_template_v2 import CON
from MARBO.wolf.werewolf.helper.utils import Matcher
from . import agent_registry as AgentRegistry
from MARBO.wolf.werewolf.agents.llm_agent import LLMAgent
from tenacity import retry, stop_after_attempt, RetryError, RetryCallState


def extract_json(text):
    if not isinstance(text, str):
        print("JSON Content Not Found")
        return None

    text = text.strip()
    if not text:
        print("JSON Content Not Found")
        return None

    decoder = json.JSONDecoder()
    candidates = [text]
    fence_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    for match in re.finditer(fence_pattern, text, re.IGNORECASE):
        candidates.append(match.group(1).strip())

    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
        
        for match in re.finditer(r'[\{\[]', candidate):
            snippet = candidate[match.start():].strip()
            try:
                _, end = decoder.raw_decode(snippet)
                return snippet[:end]
            except json.JSONDecodeError:
                continue

    print("JSON Content Not Found")
    return None


def load_json_dict(text):
    json_str = extract_json(text)
    if json_str is None:
        raise json.JSONDecodeError("No JSON content found", text if isinstance(text, str) else str(text), 0)

    content = json.loads(json_str)
    if not isinstance(content, dict):
        raise json.JSONDecodeError("JSON content is not an object", json_str, 0)
    return content


def before_attempts(retry_state: RetryCallState):
    global attempt_number
    attempt_number = retry_state.attempt_number
    if retry_state.outcome is not None and retry_state.outcome.failed:
        exc = retry_state.outcome.exception()
        print(f"[Retry {attempt_number}] {type(exc).__name__}: {exc}")


@AgentRegistry.register(["makto_recon"])
class MaktoAgent(LLMAgent):
    def __init__(self,
                 client=None,
                 tokenizer=None,
                 llm=None,
                 temperature=0.0,
                 log_file=None):
        super().__init__(client=client, tokenizer=tokenizer, llm=llm, temperature=temperature, log_file=log_file)
        self.matcher = Matcher()

        # Memory split:

        self.observed_notes = {}   # {day: str}
        self.belief_states = {}    # {day: {"speech": {...}, "vote": {...}, "night": {...}}}
        self.policy_memory = {}    # {day: {"speech": str, "vote": str, "night": str}}

        self.alive = [f"Player {i}" for i in range(1, 10)]
        self.client = client
        self.vote_reason = {}

    # =========================
    # Memory Helpers
    # =========================
    def _phase_to_key(self, phase: str) -> str:
        phase_l = phase.lower()
        if "night" in phase_l or "skill" in phase_l:
            return "night"
        elif "vote" in phase_l:
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
        content = load_json_dict(raw_action)
        parsed = {}

        valid_roles = {
            "Villager", "Werewolf", "Seer", "Witch", "Guard", "Hunter", "Unknown",
            "村民", "狼人", "预言家", "女巫", "守卫", "猎人", "未知"
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
        day = int(phase.split("_")[0])
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
        day = int(phase.split("_")[0])
        phase_key = self._phase_to_key(phase)
        if day not in self.policy_memory:
            self.policy_memory[day] = {}
        self.policy_memory[day][phase_key] = self.build_policy_memory(day, phase_key)

    def _get_policy_note_logs(self, current_phase: str) -> str:
        current_day = int(current_phase.split("_")[0])
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
        current_day = int(current_phase.split("_")[0])
        chunks = []

        for day in sorted(self.observed_notes.keys()):
            if day <= current_day:
                content = self.observed_notes.get(day, "").strip()
                if content:
                    chunks.append(f"Day {day} Observed Notes:\n{content}")

        return "\n\n".join(chunks).strip()

    # =========================
    # Retry Prompt Helper
    # =========================
    def _append_retry_format_hint(self, messages, phase, valid_action, raw_response, exc):
        if not isinstance(messages, list):
            return

        if messages and isinstance(messages[-1], dict):
            last_msg = str(messages[-1].get("content", ""))
            if "[RETRY FORMAT FIX]" in last_msg:
                return

        valid_action_text = "\n".join(f"- {a}" for a in valid_action)
        err_type = type(exc).__name__
        err_msg = str(exc)
        err_msg = err_msg[:220] + "..." if len(err_msg) > 220 else err_msg
        phase_text = str(phase).lower()

        if "vote" in phase_text:
            schema_hint = (
                'Use keys exactly: "notes", "voting reason", "voting player".\n'
                '"notes" must be a plain string (NOT object/list).\n'
                '"voting player" must be a number or "Abstain".'
            )
        elif "state_reconstruction" in phase_text:
            schema_hint = (
                'Use JSON object only.\n'
                'Each key must be "Player N".\n'
                'Each value must be one of: Villager, Werewolf, Seer, Witch, Guard, Hunter, Unknown.'
            )
        else:
            schema_hint = (
                "Use only valid role-action JSON keys (kill/check/guard/heal/poison/shoot/reason).\n"
                "Do not invent values (e.g., yes/no) unless valid actions contain them."
            )

        hint = (
            "[RETRY FORMAT FIX]\n"
            f"Phase: {phase}\n"
            f"Last error: {err_type}: {err_msg}\n"
            "Re-generate now with strict format:\n"
            "- Output ONLY one JSON object.\n"
            "- No markdown, no explanation text.\n"
            "- Use double quotes for all keys and string values.\n"
            f"- {schema_hint}\n"
            "- Ensure parsed action becomes one of valid actions below when applicable.\n\n"
            f"Valid actions:\n{valid_action_text}"
        )
        messages.append({"role": "user", "content": hint})

    # =========================
    # Log Parsing
    # =========================
    def _parse_log_time(self, time_text):
        pattern = r"(?:Day\s+|第)(\d+)(?:\s+|天)(Daytime|Nighttime|白天|夜晚)"
        match = re.search(pattern, time_text)
        matched_day = int(match.group(1))
        time_of_day = "day" if match.group(2) in ["Daytime", "白天"] else "night"
        return matched_day, time_of_day

    def parse_vote_info(self, game_log, vote_at_day):
        def sort_candidates(candidates):
            sorted_candidates = sorted(candidates, key=lambda candidate: int(re.findall(r'\d+', candidate)[0]))
            return sorted_candidates

        vote_info = ""
        pk_player = []
        vote_out_player = None
        for log in game_log:
            log_turn, log_time_of_day = self._parse_log_time(log.time)
            if log_turn != vote_at_day or "vote" not in log.event:
                continue

            if log.event == 'vote':
                if log.target > 0:
                    vote_info += "Player {} voted for Player {};\n".format(log.source, log.target)
                else:
                    vote_info += "Player {} abstained;\n".format(log.source)
            elif log.event == 'vote_pk':
                if log.target > 0:
                    vote_info += "PK Phase, Player {} voted for Player {};\n".format(log.source, log.target)
                else:
                    vote_info += "PK Phase, Player {} abstained;\n".format(log.source)
            elif log.event == 'end_vote':
                if log.content['vote_outcome'] == 'all abstention':
                    vote_info += "Result: All players abstained, entering night phase directly.\n"
                elif log.content['vote_outcome'] == 'all abstention in pk':
                    vote_info += "Result: All players abstained, entering night phase directly.\n"
                elif log.content['vote_outcome'] == 'draw':
                    pk_speech_list = ''
                    for idx in log.content['speech_queue']:
                        pk_speech_list += 'Player {}, '.format(idx)
                    pk_speech_list = pk_speech_list[:-2]
                    pk_vote_list = ''
                    for idx in log.content['vote_queue']:
                        pk_vote_list += 'Player {}, '.format(idx)
                    pk_vote_list = pk_vote_list[:-2]
                    vote_info += f"Result: Draw, players {pk_speech_list} will speak again for PK, and players {pk_vote_list} will vote.\n"
                    pk_player = sort_candidates(pk_speech_list.split(", "))
                elif log.content['vote_outcome'] == 'draw in pk':
                    vote_info += "Result: Another draw, entering night phase directly.\n"
                elif type(log.content['vote_outcome']) == int:
                    vote_info += "Result: Player {} was voted out.\n".format(log.content['expelled'])
                    vote_out_player = log.content['expelled']
        return vote_info, vote_out_player, pk_player

    # =========================
    # Observation Formatting
    # =========================
    def format_log_with_notes(self, current_phase, game_log):
        if "state_reconstruction" in current_phase:
            note_logs = self._get_recon_note_logs(current_phase)
        else:
            note_logs = self._get_policy_note_logs(current_phase)

        night_log_tmp = ""
        night_action_log_tmp = ""
        speech_log_tmp = ""
        current_turn = int(current_phase.split("_")[0])

        for log in game_log:
            log_turn, log_time_of_day = self._parse_log_time(log.time)
            if "night" in current_phase:
                if log_turn == current_turn:
                    if log.event == 'skill_wolf':
                        night_log_tmp += f"Tonight Werewolf {log.source} is preparing to hunt Player {log.target}.\n"
                    elif log.event == "kill_decision":
                        night_log_tmp += f"Tonight Player {log.target} died.\n"

            elif "day" in current_phase:
                if log.event == 'werewolf_team_info' and "Werewolves are: " not in night_action_log_tmp:
                    wolf_team = ''
                    for idx in log.content['wolf_team']:
                        wolf_team += '{},'.format(idx)
                    wolf_team = wolf_team[:-1]
                    night_action_log_tmp += "Werewolves are: Player {}.\n".format(wolf_team)

                if log_turn == current_turn - 1:
                    if log.event == 'end_night':
                        dead_list = ""
                        for idx in log.content['dead_list']:
                            dead_list += 'Player {}, '.format(idx)
                        if len(dead_list) > 0:
                            dead_list = dead_list[:-2]
                            night_log_tmp = "Last night player {} died;".format(dead_list)
                        else:
                            night_log_tmp = "Last night was a peaceful night, no one died;"
                    if log.event == 'skill_wolf':
                        if "Werewolves voted to kill targets: " not in night_action_log_tmp:
                            night_action_log_tmp += "Werewolves voted to kill targets: "
                        night_action_log_tmp += f"Player {log.source} chose to kill Player {log.target};\n"
                    elif log.event == 'kill_decision':
                        night_action_log_tmp += "Werewolf team killed Player {}.\n".format(log.target)
                    elif log.event == 'skill_seer':
                        checked_identity = log.content.get('checked_identity', log.content.get('cheked_identity'))
                        night_action_log_tmp += "Seer checked Player {}, Player {} is a {}.\n".format(
                            log.target,
                            log.target,
                            'Werewolf' if checked_identity == 'bad' else 'Good Person'
                        )
                    elif log.event == 'skill_guard':
                        if log.target != 0:
                            night_action_log_tmp += "Guard chose to protect Player {} tonight.\n".format(log.target)
                        else:
                            night_action_log_tmp += "Guard chose not to protect anyone tonight.\n"
                    elif log.event == 'skill_witch':
                        if 'heal' in log.content:
                            if log.target != 0:
                                night_action_log_tmp += "Witch used a healing potion on Player {}.\n".format(log.target)
                            else:
                                night_action_log_tmp += "Witch did not use a healing potion.\n"
                        elif 'poison' in log.content:
                            if log.target != 0:
                                night_action_log_tmp += "Witch used a poison potion on Player {}.\n".format(log.target)
                            else:
                                night_action_log_tmp += "Witch did not use a poison potion.\n"
                    elif log.event == 'skill_hunter':
                        night_action_log_tmp += "Player {} is the Hunter; after being killed, they shot Player {}.\n".format(
                            log.source, log.target
                        )
                elif log_turn == current_turn:
                    if log.event == 'speech' or log.event == 'speech_pk':
                        if len(log.content['speech_content']) > 0:
                            speech_log_tmp += "**Player {}**：{};\n".format(log.source, log.content['speech_content'])
                        else:
                            speech_log_tmp += "**Player {}**: Empty;\n".format(log.source)

        return note_logs, night_log_tmp, night_action_log_tmp, speech_log_tmp

    def _process_speaker_order(self, first_speaker, all_speaker):
        first_index = all_speaker.index(f'Player {first_speaker}')
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
            log_turn, log_time_of_day = self._parse_log_time(log.time)

            if log.event == 'werewolf_team_info' and "Werewolves are: " not in wolf_team_info:
                wolf_team = ''
                for idx in log.content['wolf_team']:
                    wolf_team += '{},'.format(idx)
                wolf_team = wolf_team[:-1]
                wolf_team_info += "- Werewolves are: Player {}.\n".format(wolf_team)

            if log.event == 'end_night':
                dead_list = ""
                for idx in log.content['dead_list']:
                    dead_list += 'Player {}, '.format(idx)
                    if f"Player {idx}" in self.alive:
                        self.alive.remove(f"Player {idx}")
                if len(dead_list) > 0:
                    dead_list = dead_list[:-2]
                    night_log_tmp += f"Round {log_turn + 1}, player {dead_list} died; "
                else:
                    night_log_tmp += f"Round {log_turn + 1} was a peaceful night, no one died; "

            elif log_turn < current_turn or ("night" in current_phase and log_turn == current_turn):
                if log.event == 'kill_decision':
                    night_action_log_tmp += f"Round {log_turn + 1}, the Werewolf faction chose to kill Player {log.target}.\n"
                elif log.event == 'skill_seer':
                    checked_identity = log.content.get("checked_identity", log.content.get("cheked_identity"))
                    if checked_identity == "bad":
                        night_action_log_tmp += f"Round {log_turn + 1}, the Seer checked Player {log.target}, and Player {log.target} is a Werewolf.\n"
                    else:
                        night_action_log_tmp += f"Round {log_turn + 1}, the Seer checked Player {log.target}, and Player {log.target} is not a Werewolf.\n"
                elif log.event == 'skill_guard':
                    if log.target != 0:
                        night_action_log_tmp += f"Round {log_turn + 1}, the Guard chose to protect Player {log.target}.\n"
                    else:
                        night_action_log_tmp += f"Round {log_turn + 1}, the Guard chose not to protect anyone.\n"
                elif log.event == 'skill_witch':
                    if 'heal' in log.content:
                        if log.target != 0:
                            night_action_log_tmp += f"Round {log_turn + 1}, the Witch used a healing potion on Player {log.target}.\n"
                        else:
                            night_action_log_tmp += f"Round {log_turn + 1}, the Witch did not use a healing potion.\n"
                    elif 'poison' in log.content:
                        if log.target != 0:
                            night_action_log_tmp += f"Round {log_turn + 1}, the Witch used a poison potion on Player {log.target}.\n"
                        else:
                            night_action_log_tmp += f"Round {log_turn + 1}, the Witch did not use a poison potion.\n"
                    elif 'pass' in log.content:
                        night_action_log_tmp += f"Round {log_turn + 1}, the Witch did not use a healing potion; the Witch did not use a poison potion.\n"
                elif log.event == 'skill_hunter' and log.target != 0:
                    if log_turn == current_turn:
                        night_action_log_tmp += f"Player {log.source} is the Hunter; after being killed by werewolves on Night {current_turn}, they chose to shoot Player {log.target}.\n"
                    else:
                        night_action_log_tmp += f"Player {log.source} is the Hunter; after being voted out on Day {log_turn}, they chose to shoot Player {log.target}.\n"

            if "night" in current_phase:
                if log_turn == current_turn:
                    if log.event == 'skill_wolf':
                        if "Werewolves voted to kill targets: " not in night_log_tmp:
                            night_log_tmp += "Werewolves voted to kill targets: "
                        night_log_tmp += "Player {} chose to kill Player {}; ".format(log.source, log.target)
                    elif log.event == "kill_decision":
                        wolf_killed_at_night = "Tonight, Player {} died, ".format(log.target)

            if log_turn == current_turn:
                if log.event == 'speech':
                    previous_speaker.append(log.source)
                    if len(log.content['speech_content']) > 0:
                        speech_text = log.content['speech_content']
                        if isinstance(speech_text, dict):
                            speech_text = speech_text.get('content', speech_text.get('Speech', speech_text.get('speech', str(speech_text))))
                        speech_log_tmp += "**Player {}**：{}\n".format(log.source, str(speech_text).strip())
                    else:
                        speech_log_tmp += "**Player {}**： Empty.\n".format(log.source)
                elif log.event == 'speech_pk':
                    previous_speaker_pk.append(log.source)
                    if len(log.content['speech_content']) > 0:
                        speech_text = log.content['speech_content']
                        if isinstance(speech_text, dict):
                            speech_text = speech_text.get('content', speech_text.get('Speech', speech_text.get('speech', str(speech_text))))
                        pk_speech_log_tmp += f"**Player {log.source}**：{str(speech_text).strip()}\n"
                    else:
                        pk_speech_log_tmp += f"**Player {log.source}**：Empty.\n"

        return note_logs, night_log_tmp, night_action_log_tmp, wolf_team_info, wolf_killed_at_night, \
            speech_log_tmp, previous_speaker, pk_speech_log_tmp, previous_speaker_pk

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
            else:
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
                subjective_info += f"- Current PK phase, you are the first to speak.\n"

        elif "speech" in phase:
            if len(note_logs.strip()) > 0:
                subjective_info += note_logs + "\n"
            subjective_info += f"- Currently in Round {day}, speeches of players before you in this round:\n{speech_log}\n"

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

        self.alive = [f"Player {i}" for i in range(1, len(observation["roles"]) + 1)]

        identity_info = CON.player_identity_info.format(
            player_idx=observation['current_act_idx'],
            identity=identity,
            identity_ability=CON.identity_abilities[identity]
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
                observation['game_log'], day_i
            )
            vote_info += vote_info_at_day_i

            if vote_out_at_day_i is not None and f"Player {vote_out_at_day_i}" in self.alive:
                self.alive.remove(f"Player {vote_out_at_day_i}")

        if "pk" in phase or "skill" in phase:
            vote_info_last_day, vote_out_last_day, pk_players_at_day_i = self.parse_vote_info(
                observation['game_log'], day - 1
            )
            if vote_info_last_day != "":
                vote_info += f"Round {day - 1} Voting Records: "
            vote_info += vote_info_last_day
            if vote_out_last_day is not None and f"Player {vote_out_last_day}" in self.alive:
                self.alive.remove(f"Player {vote_out_last_day}")

        objective_info = self._format_objective_info(
            phase, observation, note_logs, night_obs, night_action_log,
            wolf_team_info, prev_speaker, prev_speaker_pk, vote_info, pk_players_at_day_i
        )
        subjective_info = self._format_subjective_info(
            phase, observation, note_logs, speech_log, pk_speech_log
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
                instruction_prompt=instruction
            )

        elif "speech" in phase:
            prompt = CON.speech_prompt_v3.format(
                player_identity_info=identity_info,
                objective_info=objective_info,
                subjective_info=subjective_info,
                your_role=brief_identity_description
            )

        elif "vote" in phase:
            prompt = CON.vote_prompt_v3.format(
                player_identity_info=identity_info,
                objective_info=objective_info,
                subjective_info=subjective_info,
                your_role=brief_identity_description
            )

        elif "state_reconstruction" in phase:
            prompt = CON.state_reconstruction_sample_v3.format(
                player_identity_info=identity_info,
                objective_info=objective_info,
                subjective_info=subjective_info,
                your_role=brief_identity_description
            )
        else:
            raise ValueError("Invalid phase: {}".format(phase))

        return prompt

    def get_sys_prompt(self, observation):
        for i, log in enumerate(observation["game_log"]):
            if log.event == 'game_setting':
                total_players = sum([cnt for _, cnt in log.content.items()])
                if total_players == 9:
                    if "Guard" in log.content:
                        return CON.game_description_9p.format(god_description=CON.guard_description)
                    elif "Hunter" in log.content:
                        return CON.game_description_9p.format(god_description=CON.hunter_description)
                elif total_players == 7:
                    if "Guard" in log.content:
                        return CON.game_description_7p.format(god_description=CON.guard_description)
                    elif "Witch" in log.content:
                        return CON.game_description_7p.format(god_description=CON.witch_description)
            if i > 3:
                break
        return CON.game_description

    # =========================
    # Generation
    # =========================
    def __vllm_generate(self, messages):
        def process_messages(messages):
            processed_messages = []
            for message in messages:
                role = message["role"]
                content = message["content"].strip()
                processed_messages.append({"role": role, "content": content})
            return processed_messages

        messages = process_messages(messages)
        response = self.client.chat.completions.create(
            model=self.llm, messages=messages, temperature=self.temperature,
        )
        response_text = response.choices[0].message.content.strip()
        return response_text

    @retry(stop=stop_after_attempt(3), before=before_attempts)
    def _generate_speech(self, observation, messages):
        global attempt_number
        raw_action = self.__vllm_generate(messages)
        action, _, _, _, speech_template = self.parse_speech(raw_action)
        env_action = ('speech', action)
        print("*******************************")
        print("I am Player {}, my identity is {}, current phase: {}".format(
            observation['current_act_idx'], observation['identity'], observation['phase']
        ))
        print("speech env_action: {}".format(env_action))
        return (
            env_action,
            {
                "response": raw_action,
                "action": action,
                "speech_template": speech_template,
                "gen_times": attempt_number
            }
        )

    @retry(stop=stop_after_attempt(3), before=before_attempts)
    def _generate_state_reconstruction(self, observation, messages):
        global attempt_number
        raw_action = self.__vllm_generate(messages)
        raw_action = "" if raw_action is None else str(raw_action).strip()

        # Validate parse early for retry
        try:
            _ = self.parse_state_reconstruction(raw_action)
        except Exception as e:
            print(
                f"[State Reconstruction Parse Retry] Attempt {attempt_number} failed. "
                "Regenerating with stricter output format..."
            )
            self._append_retry_format_hint(messages, observation["phase"], [], raw_action, e)
            raise

        env_action = ('state_reconstruction', raw_action)
        print("*******************************")
        print("I am Player {}, my identity is {}, current phase: {}".format(
            observation['current_act_idx'], observation['identity'], observation['phase']
        ))
        print("state reconstruction env_action: {}".format(env_action))
        return (
            env_action,
            {
                "response": raw_action,
                "action": raw_action,
                "gen_times": attempt_number
            }
        )

    @retry(stop=stop_after_attempt(3), before=before_attempts)
    def _generate_vote(self, observation, messages, valid_action):
        phase = observation['phase']
        global attempt_number
        response_text = self.__vllm_generate(messages)
        print(response_text)

        try:
            note_str, vote_reason, output_vote = self.parse_note_vote_reason(phase, response_text)
            if output_vote == -1:
                action = "{'vote':'abstain'}"
            else:
                action = f"{{'vote':'{output_vote}'}}"
            assert action in valid_action, (
                f"Parsed vote action not in valid_action. parsed={action}, valid_action={valid_action}, "
                f"response_text={response_text}"
            )
            assert response_text is not None and vote_reason is not None
        except Exception as e:
            print(
                f"[Vote Parse Retry] Attempt {attempt_number} failed. "
                "Regenerating with stricter output format..."
            )
            self._append_retry_format_hint(messages, phase, valid_action, response_text, e)
            raise

        print("I am Player {}, my identity is {}, current phase: {}".format(
            observation['current_act_idx'], observation['identity'], observation['phase']
        ))
        print("retry {}, action: {} valid_action: {} response: {}".format(
            attempt_number, action, valid_action, response_text
        ))
        print("vote reason: \n{}".format(vote_reason))

        return (
            action,
            {
                "response": response_text,
                "action": action,
                "phase": phase,
                "note": note_str,
                "vote_reason": vote_reason,
                "gen_times": attempt_number
            }
        )

    @retry(stop=stop_after_attempt(3), before=before_attempts)
    def _generate_action(self, observation, messages, valid_action):
        global attempt_number
        raw_action = self.__vllm_generate(messages)
        if 'None' in raw_action:
            raw_action = raw_action.replace('None', 'No')

        if observation["identity"] == "Witch":
            if "Yes" in raw_action or "是" in raw_action:
                for action_tuple in observation['valid_action']:
                    if action_tuple[0] == 'witch_heal':
                        raw_action = raw_action.replace("Yes", f"{action_tuple[1]}").replace("是", f"{action_tuple[1]}")
                        break

        phase = observation["phase"]
        try:
            action, action_reason = self.parse_night_action(observation["identity"], raw_action)
            assert action in valid_action, (
                f"Parsed action not in valid_action. parsed={action}, valid_action={valid_action}, raw_action={raw_action}"
            )
        except Exception as e:
            print(
                f"[Action Parse Retry] Attempt {attempt_number} failed. "
                "Regenerating with stricter output format..."
            )
            self._append_retry_format_hint(messages, phase, valid_action, raw_action, e)
            raise

        print("*******************************")
        print("I am Player {}, my identity is {}, current phase: {}".format(
            observation['current_act_idx'], observation['identity'], observation['phase']
        ))
        print("retry {}, action: {} valid_action: {} raw_action: {}".format(
            attempt_number, action, valid_action, raw_action
        ))

        return (
            action,
            {
                "response": raw_action,
                "action": action,
                "action_reason": action_reason,
                "gen_times": attempt_number
            }
        )

    # =========================
    # Main Act
    # =========================
    def act(self, observation):
        system_prompt = self.get_sys_prompt(observation)
        input_prompt = self.format_observation(observation)
        print("\n------ PROMPT (w/o game desc.) ------")
        print(input_prompt)

        phase = observation['phase']
        day = int(phase.split("_")[0])

        if "speech" in phase:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt.strip()}
            ]
            try:
                env_action, generation_info = self._generate_speech(observation, messages)
            except RetryError:
                fallback_speech = "I have nothing to say"
                env_action = ('speech', fallback_speech)
                generation_info = {
                    "response": fallback_speech,
                    "action": fallback_speech,
                    "speech_template": "",
                    "gen_times": 4
                }

            if self.has_log:
                self.logger.info(
                    phase,
                    extra={
                        "prompt": input_prompt,
                        "response": generation_info.get("response", ""),
                        "action": generation_info.get("action", "空"),
                        "speech_template": generation_info.get("speech_template", ""),
                        "player_id": observation['current_act_idx'],
                        "role": observation['identity'],
                        "phase": phase,
                        "gen_times": generation_info.get("gen_times", 0)
                    }
                )

        elif "state_reconstruction" in phase:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt.strip()}
            ]
            try:
                env_action, generation_info = self._generate_state_reconstruction(observation, messages)
                parsed_belief = self.parse_state_reconstruction(generation_info["response"])
                self._store_belief_state(phase, parsed_belief)
                self.update_policy_memory(phase)
            except RetryError:
                env_action = ('state_reconstruction', "{}")
                generation_info = {
                    "response": "{}",
                    "action": "{}",
                    "gen_times": 4
                }

            if self.has_log:
                self.logger.info(
                    phase,
                    extra={
                        "prompt": input_prompt,
                        "response": generation_info.get("response", ""),
                        "action": generation_info.get("action", ""),
                        "player_id": observation['current_act_idx'],
                        "role": observation['identity'],
                        "phase": phase,
                        "gen_times": generation_info.get("gen_times", 0)
                    }
                )

        elif "vote" in phase:
            valid_actions_str = self.get_valid_actions_str(observation['valid_action'])
            valid_action = list(self.nlp_action_to_env_action.keys())
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt.strip()}
            ]
            try:
                action, generation_info = self._generate_vote(observation, messages, valid_action)
            except RetryError as e:
                last_exc = e.last_attempt.exception() if e.last_attempt is not None else None
                err_type = type(last_exc).__name__ if last_exc is not None else "UnknownError"
                err_msg = str(last_exc) if last_exc is not None else "No exception captured from retry attempts."
                err_tb = ""
                if last_exc is not None:
                    err_tb = "".join(traceback.format_exception(type(last_exc), last_exc, last_exc.__traceback__))

                print("[Vote RetryError] _generate_vote failed after 3 attempts.")
                print(f"[Vote RetryError] Last exception type: {err_type}")
                print(f"[Vote RetryError] Last exception message: {err_msg}")
                if err_tb:
                    print("[Vote RetryError] Last exception traceback:")
                    print(err_tb)

                action = random.choice(valid_action)
                generation_info = {
                    "response": action,
                    "vote_reason": (
                        "Vote generation error more than 3 times. "
                        f"Last error: {err_type}: {err_msg}. "
                        "Randomly choosing a valid vote or abstention."
                    ),
                    "gen_times": 4,
                    "retry_error_type": err_type,
                    "retry_error_message": err_msg
                }

            env_action = self.nlp_action_to_env_action[action]
            self.vote_reason[day] = generation_info.get("vote_reason", "")

            if self.has_log:
                self.logger.info(
                    phase,
                    extra={
                        "prompt": input_prompt,
                        "response": generation_info.get("response", ""),
                        "action": action,
                        "retry_error_type": generation_info.get("retry_error_type", ""),
                        "retry_error_message": generation_info.get("retry_error_message", ""),
                        "player_id": observation['current_act_idx'],
                        "role": observation['identity'],
                        "phase": phase,
                        "note": generation_info.get("note", ""),
                        "vote_reason": generation_info.get("vote_reason", ""),
                        "gen_times": generation_info.get("gen_times", 4)
                    }
                )

        else:
            valid_actions_str = self.get_valid_actions_str(observation['valid_action'])
            valid_action = list(self.nlp_action_to_env_action.keys())
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt.strip()}
            ]
            try:
                action, generation_info = self._generate_action(observation, messages, valid_action)
            except RetryError as e:
                last_exc = e.last_attempt.exception() if e.last_attempt is not None else None
                err_type = type(last_exc).__name__ if last_exc is not None else "UnknownError"
                err_msg = str(last_exc) if last_exc is not None else "No exception captured from retry attempts."
                err_tb = ""
                if last_exc is not None:
                    err_tb = "".join(traceback.format_exception(type(last_exc), last_exc, last_exc.__traceback__))

                print("[Action RetryError] _generate_action failed after 3 attempts.")
                print(f"[Action RetryError] Last exception type: {err_type}")
                print(f"[Action RetryError] Last exception message: {err_msg}")
                if err_tb:
                    print("[Action RetryError] Last exception traceback:")
                    print(err_tb)

                action = random.choice(valid_action)
                generation_info = {
                    "response": action,
                    "action_reason": (
                        "Action generation error more than 3 times. "
                        f"Last error: {err_type}: {err_msg}. "
                        "Randomly choosing a valid action."
                    ),
                    "gen_times": 4,
                    "retry_error_type": err_type,
                    "retry_error_message": err_msg
                }

            env_action = self.nlp_action_to_env_action[action]

            if self.has_log:
                self.logger.info(
                    phase,
                    extra={
                        "prompt": input_prompt,
                        "response": generation_info.get("response", ""),
                        "action": action,
                        "action_reason": generation_info.get("action_reason"),
                        "retry_error_type": generation_info.get("retry_error_type", ""),
                        "retry_error_message": generation_info.get("retry_error_message", ""),
                        "player_id": observation['current_act_idx'],
                        "role": observation['identity'],
                        "phase": phase,
                        "gen_times": generation_info.get("gen_times", 4)
                    }
                )

        return env_action

    # =========================
    # Parsing Helpers
    # =========================
    def parse_night_action(self, identity, raw_action):
        content = load_json_dict(raw_action)
        action_str = ""
        action_reason = ""

        def get_val(keys, default="no"):
            for k in keys:
                if k in content:
                    return content[k]
            return default

        if identity == "Witch":
            heal = get_val(["heal", "解药"])
            poison = get_val(["poison", "毒药"])
            action_str = f"{{'heal':'{heal}', 'poison':'{poison}'}}"
            action_reason = get_val(["reason", "原因"], "")
        elif identity == "Werewolf":
            kill = get_val(["kill", "杀害"])
            action_str = f"{{'kill':'{kill}'}}"
            action_reason = get_val(["reason", "原因"], "")
        elif identity == "Seer":
            check = get_val(["check", "查验"])
            action_str = f"{{'check':'{check}'}}"
            action_reason = get_val(["reason", "原因"], "")
        elif identity == "Guard":
            guard = get_val(["guard", "守卫"])
            action_str = f"{{'guard':'{guard}'}}"
            action_reason = get_val(["reason", "原因"], "")
        elif identity == "Hunter":
            shoot = get_val(["shoot", "击杀"])
            if shoot in ("否", "no", "No", "NO", None, ""):
                action_str = "no shoot"
            else:
                action_str = f"shoot Player {shoot}"
            action_reason = get_val(["reason", "原因"], "")
        return action_str, action_reason

    def parse_speech(self, raw_action):
        raw_action = "" if raw_action is None else str(raw_action).strip()
        json_str = extract_json(raw_action)
        content = {}
        if json_str is not None:
            try:
                parsed_content = json.loads(json_str)
                if isinstance(parsed_content, dict):
                    content = parsed_content
            except json.JSONDecodeError:
                content = {}

        role_labels_str = ""

        def get_val(keys, default=""):
            for k in keys:
                if k in content:
                    return content[k]
            return default

        speech = get_val(["speech", "Speech", "发言"])
        role_display = get_val(["Identity to Present", "self_present", "想要展示的身份"])
        role_labels = get_val(["Identity Labels", "role_label", "身份标签"], {})
        call_for_vote = get_val(["Vote", "call_for_vote", "归票"])

        if not isinstance(role_labels, dict):
            role_labels = {}

        if len(str(speech).strip()) == 0:
            if content and all(str(player).startswith("Player ") for player in content.keys()):
                speech = json.dumps(content, ensure_ascii=False)
            else:
                speech = raw_action

        if isinstance(speech, dict):
            speech = speech.get("content", speech.get("Speech", speech.get("speech", json.dumps(speech, ensure_ascii=False))))

        if not isinstance(speech, str):
            speech = json.dumps(speech, ensure_ascii=False) if isinstance(speech, dict) else str(speech)

        for player, role in role_labels.items():
            role_labels_str += f"Label {player} as {role}. "

        speech_template = f"Displaying identity as {role_display}. {role_labels_str}Vote direction: {call_for_vote}."
        return speech, role_display, role_labels, call_for_vote, speech_template

    def parse_vote_reponse(self, phase, response_text):
        day = int(phase.split("_")[0])
        response_text = response_text.strip()

        if "```json" in response_text:
            response_text = extract_json(response_text)

        try:
            return self.parse_note_vote_reason(phase, response_text)
        except (TypeError, json.JSONDecodeError):
            pass

        note_str = self.matcher.match_note(response_text, output_str=True)
        if note_str:
            self.append_observed_note(day, note_str)

        vote_reason = self.matcher.match_vote_reason(response_text)

        if "综上" in response_text:
            conclusion = response_text.split("综上")[-1]
            match = re.findall(r'\d+', conclusion)
            if len(match) > 0:
                output_vote = int(match[0])
            else:
                output_vote = -1
        else:
            if "弃票" in response_text:
                output_vote = -1
            else:
                match = re.findall(r'\d+', response_text)
                if len(match) > 0:
                    output_vote = int(match[-1])
                else:
                    output_vote = -1

        return note_str, vote_reason, output_vote

    def parse_note_vote_reason(self, phase, raw_action):
        day = int(phase.split("_")[0])
        content = load_json_dict(raw_action)
        note_str = ""

        def get_val(keys, default=""):
            for k in keys:
                if k in content:
                    return content[k]
            return default

        note_raw = get_val(["notes", "笔记"])
        if isinstance(note_raw, (dict, list)):
            note_str = json.dumps(note_raw, ensure_ascii=False)
        elif note_raw is None:
            note_str = ""
        else:
            note_str = str(note_raw)

        note_str = note_str.strip()
        if note_str:
            self.append_observed_note(day, note_str)

        vote_reason_raw = get_val(["voting reason", "voting_reason", "投票原因"])
        if isinstance(vote_reason_raw, (dict, list)):
            vote_reason = json.dumps(vote_reason_raw, ensure_ascii=False)
        elif vote_reason_raw is None:
            vote_reason = ""
        else:
            vote_reason = str(vote_reason_raw)

        vote_reason = vote_reason.strip()

        vote_to_raw = get_val(["voting player", "voting_player", "投票玩家"])
        vote_to = "" if vote_to_raw is None else str(vote_to_raw).strip()
        vote_to_l = vote_to.lower()

        if vote_to in ("", "None", "Abstain", "弃票", "No", "否") or vote_to_l in ("none", "abstain", "no"):
            output_vote = -1
        else:
            match = re.findall(r'\d+', vote_to)
            if len(match) > 0:
                output_vote = int(match[0])
            else:
                output_vote = -1

        return note_str, vote_reason, output_vote
