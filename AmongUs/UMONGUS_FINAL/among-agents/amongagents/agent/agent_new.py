import json
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp

from amongagents.agent.belief_prompts import (
    CREWMATE_EXAMPLE,
    CREWMATE_PROMPT,
    IMPOSTOR_EXAMPLE,
    IMPOSTOR_PROMPT,
    ROLE_PREDICTION_SYSTEM_PROMPT,
    PERSONALITY_PROMPT,
    CrewmatePersonalities,
    ImpostorPersonalities,
)
from amongagents.agent.neutral_prompts import (
    CREWMATE_PROMPT as NEUTRAL_CREWMATE_PROMPT,
    CREWMATE_EXAMPLE as NEUTRAL_CREWMATE_EXAMPLE,
    IMPOSTOR_PROMPT as NEUTRAL_IMPOSTOR_PROMPT,
    IMPOSTOR_EXAMPLE as NEUTRAL_IMPOSTOR_EXAMPLE,
    PERSONALITY_PROMPT as NEUTRAL_PERSONALITY_PROMPT,
    CrewmatePersonalities as NeutralCrewmatePersonalities,
    ImpostorPersonalities as NeutralImpostorPersonalities,
)


def _weighted_pick(choices, weights=None):
    if not choices:
        raise ValueError("LLM choices cannot be empty.")
    if weights is None or len(weights) != len(choices):
        return random.choice(choices)
    return random.choices(choices, weights=list(weights), k=1)[0]


def _disable_qwen_thinking(model):
    if os.getenv("LLM_DISABLE_QWEN_THINKING", "1") == "0":
        return False
    return "qwen3.6" in (model or "").lower()


TASK_ACTION_REPAIR_PROMPT = (
    "Your previous [Action] was invalid because it was not exactly one of the "
    "Available actions. The Available actions list in the current prompt is "
    "authoritative; older memory or plans may be outdated.\n\n"
    "Re-read the current location, phase, observations, and Available actions above. "
    "Copy exactly one full action from the Available actions list. Do not paraphrase, "
    "shorten, invent a route, or choose a destination unless the exact full action "
    "appears in Available actions."
)


MEETING_ACTION_REPAIR_PROMPT = (
    "Your previous [Action] was invalid because it was not a valid meeting action. "
    "The Available actions list in the current prompt is authoritative; older memory "
    "or plans may be outdated.\n\n"
    "Re-read the current discussion, observations, and Available actions above. "
    "If SPEAK: ... is available, return SPEAK: followed "
    "by a concrete message grounded in the current discussion. Do not return SPEAK alone."
)


def _clip_repair_text(text: str) -> str:
    limit = int(os.getenv("LLM_REPAIR_LOG_CHARS", "1000"))
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _extract_response_sections(response: str) -> Dict[str, str]:
    pattern = re.compile(
        r"^\s*\[Condensed Memory\]\s*(?P<memory>.*?)"
        r"\s*\[Thinking Process\]\s*(?P<thinking>.*?)"
        r"\s*\[Action\]\s*(?P<action>.*)\s*$",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(response or "")
    if not match:
        return {}
    return {
        "Condensed Memory": match.group("memory").strip(),
        "Thinking Process": match.group("thinking").strip(),
        "Action": match.group("action").strip(),
    }


def _extract_belief_state(response: str) -> str:
    pattern = re.compile(
        r"^\s*\[(?:Relational Belief|Belief State)\]\s*(?P<belief>.*?)(?:\n\s*\[[^\]]+\].*)?$",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(response or "")
    if match:
        return "Relational Belief:\n" + match.group("belief").strip()

    text = (response or "").strip()
    if text.lower().startswith("relational belief:"):
        return text
    if text.lower().startswith("belief state:"):
        return re.sub(
            r"^\s*belief state\s*:",
            "Relational Belief:",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    return ""


def _extract_event_memory(condensed_memory: str) -> str:
    match = re.search(
        r"(?:^|\n)\s*Event Memory:\s*(?P<event>.*)\s*$",
        condensed_memory or "",
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return ""

    event = match.group("event").strip()
    if not event:
        return ""
    return f"Event Memory:\n{event}"


class BeliefStateLLMAgent:
    """LLM agent variant that keeps role beliefs inside Condensed Memory."""

    belief_state = True

    def __init__(
        self,
        player,
        tools,
        game_index,
        agent_config,
        list_of_impostors,
        player_roster: Optional[List[str]] = None,
        preselected_model: Optional[str] = None,
    ):
        self.player = player
        self.tools = tools
        self.game_index = game_index
        self.player_roster = list(player_roster or [player.name])
        self.list_of_impostors = list(list_of_impostors or [])

        if player.identity == "Crewmate":
            system_prompt = CREWMATE_PROMPT.format(name=player.name)
            if player.personality is not None:
                system_prompt += PERSONALITY_PROMPT.format(
                    personality=CrewmatePersonalities[player.personality]
                )
            system_prompt += CREWMATE_EXAMPLE
            model = preselected_model if preselected_model is not None else _weighted_pick(
                agent_config["CREWMATE_LLM_CHOICES"],
                agent_config.get("CREWMATE_LLM_WEIGHTS"),
            )
        elif player.identity == "Impostor":
            system_prompt = IMPOSTOR_PROMPT.format(name=player.name)
            if player.personality is not None:
                system_prompt += PERSONALITY_PROMPT.format(
                    personality=ImpostorPersonalities[player.personality]
                )
            system_prompt += IMPOSTOR_EXAMPLE
            system_prompt += f"\nKnown Impostors: {self.list_of_impostors}\n"
            model = preselected_model if preselected_model is not None else _weighted_pick(
                agent_config["IMPOSTOR_LLM_CHOICES"],
                agent_config.get("IMPOSTOR_LLM_WEIGHTS"),
            )
        else:
            raise ValueError(f"Unsupported player identity: {player.identity}")

        self.system_prompt = system_prompt
        self.model = model
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("LLM_TOP_P", "1.0"))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "10"))

        routes = {}
        raw_routes = os.getenv("LLM_ROUTES_JSON", "")
        if raw_routes:
            try:
                routes = json.loads(raw_routes)
            except Exception as e:
                print(f"[agent_new.py] WARN: LLM_ROUTES_JSON parse failed ({e}); falling back to global URL/KEY.")

        route = routes.get(model)
        if route:
            self.api_urls = [route["url"]]
            self.api_key = route.get("key") or "dummy"
        else:
            self.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "dummy"
            raw_urls = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
            self.api_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
        self.api_url = self.api_urls[0]

        self.summarization = "No thought process has been made."
        self.current_belief_state = self._initial_belief_state()
        self.processed_memory = self._initial_event_memory()
        self.chat_history = []

        experiment_path = os.getenv("EXPERIMENT_PATH", ".")
        self.log_path = os.path.join(experiment_path, "agent-logs.json")
        self.compact_log_path = os.path.join(experiment_path, "agent-logs-compact.json")

    def _format_roster(self) -> str:
        return "\n".join(f"- {player_name}" for player_name in self.player_roster)

    def _initial_belief_state(self) -> str:
        lines = ["Relational Belief:"]
        for player_name in self.player_roster:
            if player_name == self.player.name:
                lines.append(f"- {player_name}: Role={self.player.identity}")
            elif self.player.identity == "Impostor" and player_name in self.list_of_impostors:
                lines.append(f"- {player_name}: Role=Impostor")
            else:
                lines.append(f"- {player_name}: Role=unknown")
        return "\n".join(lines)

    def _initial_event_memory(self) -> str:
        return "Event Memory:\n- No important events observed yet."

    def _combined_memory(self) -> str:
        return f"{self.current_belief_state}\n{self.processed_memory}"

    def log_interaction(self, sysprompt, prompt, original_response, step, interaction_type="action"):
        sections = _extract_response_sections(original_response) if isinstance(original_response, str) else {}
        belief_state = _extract_belief_state(original_response) if isinstance(original_response, str) else ""
        parsed_response = sections if sections else ({"Relational Belief": belief_state} if belief_state else original_response)

        interaction = {
            "game_index": "Game " + str(self.game_index),
            "step": step,
            "type": interaction_type,
            "timestamp": str(datetime.now()),
            "player": {
                "name": self.player.name,
                "identity": self.player.identity,
                "personality": self.player.personality,
                "model": self.model,
                "location": self.player.location,
                "belief_state": True,
            },
            "interaction": {
                "system_prompt": sysprompt,
                "prompt": prompt,
                "response": parsed_response,
                "full_response": original_response,
            },
        }

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.compact_log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            json.dump(interaction, f, indent=2, separators=(",", ": "))
            f.write("\n")
            f.flush()
        with open(self.compact_log_path, "a") as f:
            json.dump(interaction, f, separators=(",", ": "))
            f.write("\n")
            f.flush()

        if os.getenv("AGENT_PROGRESS_DOTS", "0") == "1":
            print(".", end="", flush=True)

    async def send_request(self, messages, max_tokens=None):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if any("localhost" in u or "127.0.0.1" in u for u in self.api_urls):
            payload["repetition_penalty"] = 1
            payload["top_k"] = 0
        if _disable_qwen_thinking(self.model):
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                url = self.api_urls[attempt % len(self.api_urls)] if self.api_urls else self.api_url
                try:
                    async with session.post(url, headers=headers, data=json.dumps(payload)) as response:
                        if response is None:
                            print(f"API request failed: response is None for {self.model}.")
                            continue
                        if response.status != 200:
                            body = await response.text()
                            print(f"[agent_new.py] API {response.status} for {self.model}: {body[:500]}")
                            continue
                        data = await response.json()
                        if "choices" not in data:
                            print(f"API request failed: 'choices' key not in response for {self.model}.")
                            continue
                        if not data["choices"]:
                            print(f"API request failed: 'choices' key is empty in response for {self.model}.")
                            continue
                        return data["choices"][0]["message"]["content"]
                except Exception:
                    print(f"API request failed. Retrying... ({attempt + 1}/{self.max_retries}) for {self.model}.")
                    continue
            return "SPEAK: ..."

    def respond(self, message):
        all_info = self.player.all_info_prompt()
        prompt = f"Player roster:\n{self._format_roster()}\n\n{all_info}\n{message}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.send_request(
            messages,
            max_tokens=int(os.getenv("LLM_RESPONSE_MAX_TOKENS", "512")),
        )

    async def update_belief_state(
        self,
        all_info,
        phase,
        timestep,
        trigger: str = "pre_action",
        source_speech_player: Optional[str] = None,
        source_speech: Optional[str] = None,
        commit: bool = True,
        speech_event_id: Optional[str] = None,
    ):
        previous_belief_state = self.current_belief_state
        system_prompt = ROLE_PREDICTION_SYSTEM_PROMPT.format(
            name=self.player.name,
            role=self.player.identity,
        )
        if self.player.identity == "Impostor":
            system_prompt += f"\nKnown Impostors: {self.list_of_impostors}\n"

        user_prompt = (
            f"Phase: {phase}\n\n"
            f"Belief update trigger: {trigger}\n\n"
            f"Belief measurement only: {not commit}\n\n"
            f"Player roster:\n{self._format_roster()}\n\n"
            f"Previous Relational Belief:\n{previous_belief_state}\n\n"
            f"Previous Event Memory:\n{self.processed_memory}\n\n"
            f"Current observations and messages:\n{all_info}\n\n"
            "Update only the relational belief over every player's role."
        )
        if speech_event_id:
            user_prompt += f"\n\nSpeech event id: {speech_event_id}\n"
        if source_speech:
            user_prompt += (
                "\n\nMost recent speech event for this update:\n"
                f"Speaker: {source_speech_player or 'Unknown'}\n"
                f"Speech: {source_speech or ''}\n"
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.send_request(
            messages,
            max_tokens=int(os.getenv("LLM_BELIEF_MAX_TOKENS", "384")),
        )
        updated_belief_state = _extract_belief_state(response)
        if updated_belief_state and commit:
            self.current_belief_state = updated_belief_state
        returned_belief_state = updated_belief_state or self.current_belief_state

        full_prompt = {
            "Phase": phase,
            "Player Roster": self.player_roster,
            "Previous Relational Belief": previous_belief_state,
            "Previous Event Memory": self.processed_memory,
            "All Info": all_info,
            "Relational Belief Enabled": True,
            "Belief Update Trigger": trigger,
            "Belief Measurement Only": not commit,
            "Belief State Committed": bool(updated_belief_state and commit),
        }
        if speech_event_id:
            full_prompt["Speech Event Id"] = speech_event_id
        if source_speech_player or source_speech:
            full_prompt["Source Speech Player"] = source_speech_player
            full_prompt["Source Speech"] = source_speech
        self.log_interaction(
            sysprompt=system_prompt,
            prompt=full_prompt,
            original_response=response,
            step=timestep,
            interaction_type="role_prediction",
        )
        return returned_belief_state

    def _commit_action_sections(self, sections):
        if sections:
            event_memory = _extract_event_memory(sections["Condensed Memory"])
            if event_memory:
                self.processed_memory = event_memory
            self.summarization = sections["Thinking Process"]

    def _match_available_action(self, output_action, available_actions):
        for action in available_actions:
            action_text = repr(action)
            if "SPEAK: " in action_text and "SPEAK:" in output_action:
                message = output_action.split("SPEAK:", 1)[1].strip()
                if hasattr(action, "provide_message"):
                    action.provide_message(message)
                else:
                    action.message = message
                return action
            if action_text in output_action:
                return action
            if hasattr(action, "message"):
                action.message = "..."
        return None

    def _action_repair_prompt(self, available_actions):
        candidates = "\n".join(
            f"{idx}. {repr(action)}" for idx, action in enumerate(available_actions, 1)
        )
        if any(repr(action).startswith("SPEAK:") for action in available_actions):
            return (
                f"{MEETING_ACTION_REPAIR_PROMPT}\n\n"
                f"Available action candidates:\n{candidates}\n\n"
                "Return only one valid action. Do not include memory, thinking, or the previous invalid answer."
            )
        return (
            f"{TASK_ACTION_REPAIR_PROMPT}\n\n"
            f"Available action candidates:\n{candidates}\n\n"
            "Return exactly one candidate action copied verbatim. Do not include memory, thinking, or the previous invalid answer."
        )

    def _repair_max_tokens(self, available_actions):
        if any(repr(action).startswith("SPEAK:") for action in available_actions):
            return int(os.getenv("LLM_SPEECH_REPAIR_MAX_TOKENS", "512"))
        return int(os.getenv("LLM_REPAIR_MAX_TOKENS", "512"))

    def _action_max_tokens(self, available_actions):
        if any(repr(action).startswith("SPEAK:") for action in available_actions):
            return int(os.getenv("LLM_SPEECH_MAX_TOKENS", "768"))
        return int(os.getenv("LLM_ACTION_MAX_TOKENS", "1536"))

    async def choose_action(self, timestep):
        available_actions = self.player.get_available_actions()
        all_info = self.player.all_info_prompt()
        phase = "Meeting phase" if len(available_actions) == 1 or all(a.name == "VOTE" for a in available_actions) else "Task phase"
        updated_belief_state = await self.update_belief_state(all_info, phase, timestep)

        user_prompt = (
            f"Summarization: {self.summarization}\n\n"
            f"Player roster:\n{self._format_roster()}\n\n"
            f"{all_info}\n\n"
            f"Memory:\n{updated_belief_state}\n{self.processed_memory}"
            f"\n\nPhase: {phase}. Use the updated relational belief as private context. "
            "Do not re-predict, modify, or rewrite roles in this action step. "
            "Update only Event Memory in [Condensed Memory] and return your output."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        full_prompt = {
            "Summarization": self.summarization,
            "Player Roster": self.player_roster,
            "All Info": all_info,
            "Memory": self._combined_memory(),
            "Phase": phase,
            "Relational Belief Enabled": True,
        }
        pending_speech_event_id = getattr(self, "pending_speech_event_id", None)
        if pending_speech_event_id:
            full_prompt["Speech Event Id"] = pending_speech_event_id

        repair_messages = list(messages)
        repair_attempts = []
        response = await self.send_request(
            repair_messages,
            max_tokens=self._action_max_tokens(available_actions),
        )
        final_response = response

        for attempt in range(4):
            sections = _extract_response_sections(final_response)
            output_action = sections["Action"] if sections else final_response.strip()
            matched_action = self._match_available_action(output_action, available_actions)
            if matched_action is not None:
                self._commit_action_sections(sections)
                if repair_attempts:
                    full_prompt["Action Repair Attempts"] = repair_attempts
                self.log_interaction(
                    sysprompt=self.system_prompt,
                    prompt=full_prompt,
                    original_response=final_response,
                    step=timestep,
                )
                return matched_action

            if attempt == 3:
                fallback_action = available_actions[0] if available_actions else None
                fallback_response = (
                    "[Condensed Memory]\n"
                    f"{self.processed_memory}\n"
                    "[Thinking Process]\n"
                    "Deterministic fallback after invalid action repair attempts.\n"
                    f"[Action] {repr(fallback_action) if fallback_action else 'None'}"
                )
                repair_attempts.append(
                    {
                        "attempt": attempt + 1,
                        "invalid_action": _clip_repair_text(output_action),
                        "response": _clip_repair_text(final_response),
                    }
                )
                full_prompt["Action Repair Attempts"] = repair_attempts
                full_prompt["Action Repair Fallback"] = repr(fallback_action) if fallback_action else None
                self.log_interaction(
                    sysprompt=self.system_prompt,
                    prompt=full_prompt,
                    original_response=fallback_response,
                    step=timestep,
                )
                return fallback_action

            repair_attempts.append(
                {
                    "attempt": attempt + 1,
                    "invalid_action": _clip_repair_text(output_action),
                    "response": _clip_repair_text(final_response),
                }
            )
            repair_messages = [
                *messages,
                {"role": "user", "content": self._action_repair_prompt(available_actions)},
            ]
            final_response = await self.send_request(
                repair_messages,
                max_tokens=self._repair_max_tokens(available_actions),
            )

    def choose_observation_location(self, map):
        if isinstance(map, (list, tuple)):
            return random.choice(map)
        return random.choice(list(map))


class BeliefPredictOnlyLLMAgent(BeliefStateLLMAgent):
    """
    Role-prediction ON, action-prompt belief-state OFF.

    Implements:
        b̂_t^{ij} ~ π_θ^i(·|τ_t^i)   (role prediction step, logged as role_prediction)
        a_t^i    ~ π_θ^i(·|τ_t^i)   (action step with neutral/wo-belief prompt)

    Useful for evaluating w/o-belief-KTO and belief-off base models that were
    fine-tuned on the vanilla (no-belief) action format, while still measuring
    their latent role-prediction capability.
    """

    def __init__(
        self,
        player,
        tools,
        game_index,
        agent_config,
        list_of_impostors,
        player_roster: Optional[List[str]] = None,
        preselected_model: Optional[str] = None,
    ):
        super().__init__(
            player, tools, game_index, agent_config, list_of_impostors,
            player_roster, preselected_model,
        )
        # Replace system prompt with neutral (no BELIEF_MEMORY_INSTRUCTION) variant
        if player.identity == "Crewmate":
            system_prompt = NEUTRAL_CREWMATE_PROMPT.format(name=player.name)
            if player.personality is not None:
                system_prompt += NEUTRAL_PERSONALITY_PROMPT.format(
                    personality=NeutralCrewmatePersonalities[player.personality]
                )
            system_prompt += NEUTRAL_CREWMATE_EXAMPLE
        elif player.identity == "Impostor":
            system_prompt = NEUTRAL_IMPOSTOR_PROMPT.format(name=player.name)
            if player.personality is not None:
                system_prompt += NEUTRAL_PERSONALITY_PROMPT.format(
                    personality=NeutralImpostorPersonalities[player.personality]
                )
            system_prompt += NEUTRAL_IMPOSTOR_EXAMPLE
            system_prompt += f"List of impostors: {list_of_impostors}"
        else:
            raise ValueError(f"Unsupported player identity: {player.identity}")
        self.system_prompt = system_prompt
        # Use plain memory string (neutral format, not "Event Memory:" prefixed)
        self.processed_memory = "No memory has been processed."

    def _commit_action_sections(self, sections):
        # Store full condensed memory (neutral format), not just event-memory sub-field
        if sections:
            self.processed_memory = sections["Condensed Memory"]
            self.summarization = sections["Thinking Process"]

    def log_interaction(self, sysprompt, prompt, original_response, step, interaction_type="action"):
        sections = _extract_response_sections(original_response) if isinstance(original_response, str) else {}
        belief_state_str = _extract_belief_state(original_response) if isinstance(original_response, str) else ""
        parsed_response = sections if sections else ({"Relational Belief": belief_state_str} if belief_state_str else original_response)

        interaction = {
            "game_index": "Game " + str(self.game_index),
            "step": step,
            "type": interaction_type,
            "timestamp": str(datetime.now()),
            "player": {
                "name": self.player.name,
                "identity": self.player.identity,
                "personality": self.player.personality,
                "model": self.model,
                "location": self.player.location,
                "belief_state": False,
                "belief_predict_only": True,
            },
            "interaction": {
                "system_prompt": sysprompt,
                "prompt": prompt,
                "response": parsed_response,
                "full_response": original_response,
            },
        }

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.compact_log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            json.dump(interaction, f, indent=2, separators=(",", ": "))
            f.write("\n")
            f.flush()
        with open(self.compact_log_path, "a") as f:
            json.dump(interaction, f, separators=(",", ": "))
            f.write("\n")
            f.flush()

        if os.getenv("AGENT_PROGRESS_DOTS", "0") == "1":
            print(".", end="", flush=True)

    async def choose_action(self, timestep):
        available_actions = self.player.get_available_actions()
        all_info = self.player.all_info_prompt()
        phase = (
            "Meeting phase"
            if len(available_actions) == 1 or all(a.name == "VOTE" for a in available_actions)
            else "Task phase"
        )

        # b̂_t ~ π(·|τ_t): role prediction step (logged as role_prediction)
        await self.update_belief_state(all_info, phase, timestep)

        # a_t ~ π(·|τ_t): action with neutral prompt — no relational belief in user_prompt
        user_prompt = (
            f"Summarization: {self.summarization}\n\n"
            f"{all_info}\n\n"
            f"Memory: {self.processed_memory}"
            f"\n\nPhase: {phase}. Return your output."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        full_prompt = {
            "Summarization": self.summarization,
            "All Info": all_info,
            "Memory": self.processed_memory,
            "Phase": phase,
            "Relational Belief Enabled": False,
            "Belief Predict Only": True,
        }
        pending_speech_event_id = getattr(self, "pending_speech_event_id", None)
        if pending_speech_event_id:
            full_prompt["Speech Event Id"] = pending_speech_event_id

        repair_messages = list(messages)
        repair_attempts = []
        response = await self.send_request(
            repair_messages,
            max_tokens=self._action_max_tokens(available_actions),
        )
        final_response = response

        for attempt in range(4):
            sections = _extract_response_sections(final_response)
            output_action = sections["Action"] if sections else final_response.strip()
            matched_action = self._match_available_action(output_action, available_actions)
            if matched_action is not None:
                self._commit_action_sections(sections)
                if repair_attempts:
                    full_prompt["Action Repair Attempts"] = repair_attempts
                self.log_interaction(
                    sysprompt=self.system_prompt,
                    prompt=full_prompt,
                    original_response=final_response,
                    step=timestep,
                )
                return matched_action

            if attempt == 3:
                fallback_action = available_actions[0] if available_actions else None
                fallback_response = (
                    "[Condensed Memory]\n"
                    f"{self.processed_memory}\n"
                    "[Thinking Process]\n"
                    "Deterministic fallback after invalid action repair attempts.\n"
                    f"[Action] {repr(fallback_action) if fallback_action else 'None'}"
                )
                repair_attempts.append({
                    "attempt": attempt + 1,
                    "invalid_action": _clip_repair_text(output_action),
                    "response": _clip_repair_text(final_response),
                })
                full_prompt["Action Repair Attempts"] = repair_attempts
                full_prompt["Action Repair Fallback"] = repr(fallback_action) if fallback_action else None
                self.log_interaction(
                    sysprompt=self.system_prompt,
                    prompt=full_prompt,
                    original_response=fallback_response,
                    step=timestep,
                )
                return fallback_action

            repair_attempts.append({
                "attempt": attempt + 1,
                "invalid_action": _clip_repair_text(output_action),
                "response": _clip_repair_text(final_response),
            })
            repair_messages = [
                *messages,
                {"role": "user", "content": self._action_repair_prompt(available_actions)},
            ]
            final_response = await self.send_request(
                repair_messages,
                max_tokens=self._repair_max_tokens(available_actions),
            )
