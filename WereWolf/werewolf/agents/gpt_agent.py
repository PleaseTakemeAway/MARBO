import json
import random
import re
import time

from tenacity import RetryCallState, RetryError, retry, stop_after_attempt
from MARBO.wolf.werewolf.agents.llm_agent import LLMAgent
from MARBO.wolf.werewolf.agents.prompt_template_v1 import CON
from . import agent_registry as AgentRegistry


attempt_number = 0


def before_attempts(retry_state: RetryCallState):
    global attempt_number
    attempt_number = retry_state.attempt_number


def _extract_json_dict(text):
    if not isinstance(text, str):
        raise ValueError("response is not a string")

    text = text.strip()
    if not text:
        raise ValueError("empty response")

    decoder = json.JSONDecoder()
    candidates = [text]
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    for match in re.finditer(fence_pattern, text, re.IGNORECASE):
        candidates.append(match.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        for match in re.finditer(r"\{", candidate):
            snippet = candidate[match.start():].strip()
            try:
                parsed, end = decoder.raw_decode(snippet)
                if isinstance(parsed, dict):
                    tail = snippet[end:].strip()
                    if not tail:
                        return parsed
            except json.JSONDecodeError:
                continue

    raise ValueError("No valid JSON object found in response")


@AgentRegistry.register(["gpt", "gpt-4", "GPT-4", "gpt4", "o1", "gpt4o", "gpt4o-mini", "gpt5", "gpt5-mini", "gemma", "gemma-4-31B-it", "qwen"])
class GPTAgent(LLMAgent):
    def __init__(self,
                 client,
                 tokenizer=None,
                 llm=None,
                 temperature=1.0,
                 rate_limit=6.0,
                 log_file=None):
        super().__init__(client=client, tokenizer=tokenizer, llm=llm, temperature=temperature, log_file=log_file)
        self.client = client
        self.llm = llm
        self.rate_limit = float(rate_limit)
        self.temperature = temperature
        self.observed_notes = {}
        self.belief_states = {}
        self.policy_memory = {}

    def get_sys_prompt(self, observation):
        for i, log in enumerate(observation.get("game_log", [])):
            if log.event == "game_setting":
                total_players = sum(cnt for _, cnt in log.content.items())
                if total_players == 9:
                    if "Guard" in log.content:
                        return CON.game_description_9p.format(god_description=CON.guard_description)
                    if "Hunter" in log.content:
                        return CON.game_description_9p.format(god_description=CON.hunter_description)
                elif total_players == 7:
                    if "Guard" in log.content:
                        return CON.game_description_7p.format(god_description=CON.guard_description)
                    if "Witch" in log.content:
                        return CON.game_description_7p.format(god_description=CON.witch_description)
            if i > 3:
                break
        return CON.game_description

    def format_observation(self, observation):
        # Delegate prompt construction to LLMAgent so that
        # state_reconstruction prompt stays aligned with Makto-style formatting.
        return super().format_observation(observation)

    def _phase_to_key(self, phase):
        phase_l = phase.lower()
        if "night" in phase_l or "skill" in phase_l:
            return "night"
        if "vote" in phase_l:
            return "vote"
        return "speech"

    def parse_state_reconstruction(self, raw_action):
        content = _extract_json_dict(raw_action)
        parsed = {}

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

        for k, v in content.items():
            key = str(k).strip()
            m = re.search(r"(\d+)", key)
            if m is None:
                continue
            normalized_key = f"Player {int(m.group(1))}"

            val = str(v).strip()
            val = role_alias.get(val, val)
            if val not in valid_roles:
                val = "Unknown"
            parsed[normalized_key] = val

        if not parsed:
            raise ValueError("No player-role mapping parsed from state reconstruction output")
        return parsed

    def _store_belief_state(self, phase, parsed_belief):
        day = int(phase.split("_")[0])
        phase_key = self._phase_to_key(phase)
        if day not in self.belief_states:
            self.belief_states[day] = {}
        self.belief_states[day][phase_key] = parsed_belief

    def labels_to_belief_lines(self, role_map):
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

    def build_policy_memory(self, day, phase_key):
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

    def update_policy_memory(self, phase):
        day = int(phase.split("_")[0])
        phase_key = self._phase_to_key(phase)
        if day not in self.policy_memory:
            self.policy_memory[day] = {}
        self.policy_memory[day][phase_key] = self.build_policy_memory(day, phase_key)

    def _append_state_reconstruction_retry_hint(self, messages, exc):
        if not isinstance(messages, list):
            return
        if messages and isinstance(messages[-1], dict):
            if "[RETRY FORMAT FIX]" in str(messages[-1].get("content", "")):
                return

        err_type = type(exc).__name__
        err_msg = str(exc)
        err_msg = err_msg[:220] + "..." if len(err_msg) > 220 else err_msg
        hint = (
            "[RETRY FORMAT FIX]\n"
            f"Last error: {err_type}: {err_msg}\n"
            "Output ONLY one JSON object.\n"
            "Use keys exactly in format: \"Player N\".\n"
            "Use values only from: Villager, Werewolf, Seer, Witch, Guard, Hunter, Unknown.\n"
            "No markdown, no explanation text."
        )
        messages.append({"role": "user", "content": hint})

    def _chat_generate(self, messages):
        if "o1" in self.llm:
            response = self.client.chat.completions.create(
                model=self.llm, messages=messages, max_tokens=32000
            )
        else:
            response = self.client.chat.completions.create(
                model=self.llm, messages=messages, temperature=self.temperature
            )
        return response.choices[0].message.content.strip()

    @retry(stop=stop_after_attempt(3), before=before_attempts)
    def _generate_state_reconstruction(self, observation, messages):
        global attempt_number

        if self.llm is None:
            raw_action = "{}"
            return (
                ("state_reconstruction", raw_action),
                {"response": raw_action, "action": raw_action, "gen_times": -1},
            )

        raw_action = self._chat_generate(messages)
        raw_action = "" if raw_action is None else str(raw_action).strip()

        try:
            _ = self.parse_state_reconstruction(raw_action)
        except Exception as e:
            self._append_state_reconstruction_retry_hint(messages, e)
            raise

        env_action = ("state_reconstruction", raw_action)
        return (
            env_action,
            {"response": raw_action, "action": raw_action, "gen_times": attempt_number},
        )

    def act(self, observation):
        phase = observation['phase']
        prompt = self.format_observation(observation)
        time.sleep(self.rate_limit)
        if "state_reconstruction" in phase:
            system_prompt = self.get_sys_prompt(observation)
            input_prompt = prompt
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
                self.logger.info(phase,
                                 extra={"prompt": input_prompt,
                                        "response": generation_info.get("response", "{}"),
                                        "action": generation_info.get("action", "{}"),
                                        "player_id": observation['current_act_idx'],
                                        "role": observation['identity'],
                                        "phase": phase,
                                        "gen_times": generation_info.get("gen_times", 0)})
            return env_action

        if 'speech' in phase:
            if self.llm is not None:
                messages = [{'role': 'user', 'content': prompt}]
                if "o1" in self.llm:
                    response = self.client.chat.completions.create(model=self.llm, messages=messages, max_tokens=32000)
                else:
                    response = self.client.chat.completions.create(
                        model=self.llm, messages=messages, temperature=self.temperature
                    )
                raw_action = response.choices[0].message.content.strip()
                checked_action = self.extract_answer(raw_action)
                gen_times = 0
            else:
                raw_action = "aaa"
                gen_times = -1
                checked_action = 'bbb'
            env_action = ('speech', checked_action)

            if self.has_log:
                self.logger.info(phase,
                                 extra={"prompt": prompt,
                                        "response": checked_action,
                                        "action": raw_action,
                                        "player_id": observation['current_act_idx'],
                                        "role": observation['identity'],
                                        "phase": phase,
                                        "gen_times": gen_times})
        else: 
            valid_action = list(self.nlp_action_to_env_action.keys())
            retry_count = 0
            raw_action = None
            if self.llm is not None:
                action = ''
                while action not in valid_action:
                    retry_count += 1
                    if retry_count > 3:
                        raw_action = valid_action[random.randint(0, len(valid_action) - 1)]
                        break
                    messages = [{'role': 'user', 'content': prompt}]
                    if "o1" in self.llm:
                        response = self.client.chat.completions.create(model=self.llm, messages=messages,
                                                                       max_tokens=32000)
                    else:
                        response = self.client.chat.completions.create(
                            model=self.llm, messages=messages, temperature=self.temperature
                        )
                    raw_action = response.choices[0].message.content.strip().strip("- ")
                    try:
                        assert raw_action in valid_action
                        action = raw_action
                    except:
                        action = valid_action[random.randint(0, len(valid_action) - 1)]
            else:
                action = valid_action[random.randint(0, len(valid_action) - 1)]
                print("random choose a valid action, action: {} valid_action: {}".format(action, valid_action))
            env_action = self.nlp_action_to_env_action[action]
            if raw_action is None:
                raw_action = action
            if self.has_log:
                self.logger.info(phase,
                                 extra={"prompt": prompt,
                                        "response": raw_action,
                                        "action": action,
                                        "player_id": observation['current_act_idx'],
                                        "role": observation['identity'],
                                        "phase": phase,
                                        "gen_times": retry_count - 1})
        return env_action

    def extract_answer(self, response):
        pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(pattern, response, re.IGNORECASE)
        json_str = matches[0].strip() if matches else response.strip()
        
        try:
            data = json.loads(json_str)
            for key in ["Speech", "speech", "发言"]:
                if key in data:
                    return str(data[key]).strip()
        except Exception:
            pass

        return response.strip()
