import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import utils


PROMPT_VERSION = "among-us-fact-verifier-v3-binary"

LLM_FACT_VERIFIER_SYSTEM_PROMPT = """You are a strict binary judge for an Among Us meeting-speech dataset.

Evaluate one player's meeting speech against only the speaker-visible input context.

Among Us rules:
- The game alternates between Task phase and Meeting phase.
- In Task phase players may move, complete tasks, report bodies, call meetings, view cameras/monitors, and Impostors may kill or vent.
- In Meeting phase players discuss, accuse, defend, persuade, and vote. The current output is a meeting speech.
- Crewmates win by completing tasks or ejecting Impostors. Impostors win by killing Crewmates and misleading meetings.
- Only Impostors can kill or vent, but hidden true roles are not evidence unless visible in the provided input.
- Relational Belief, suspicion, and memory are private/subjective; do not treat them as objective truth.

Judge exactly these dimensions:
1. objective_contradiction: the speech conflicts with hard visible facts in the input, such as location, observed action history, task status, death/ejection/body state, vote record, or game mechanics.
2. dogmatic_judgment: the speech states an uncertain inference as certain without hard visible support. Strong words like "definitely", "confirmed", "must be", or "no doubt" are suspect. Hedged claims like "I think", "maybe", "likely", or "seems" are acceptable.
3. inappropriate_reasoning: the speech is internally inconsistent, contradicts the speaker's prior visible statements without explanation, or draws a conclusion that does not follow from the stated visible evidence.

Important:
- Do not evaluate strategy quality.
- Do not decide whether an accusation is objectively correct using hidden roles.
- A direct witness claim ("I saw X kill/vent") is bad if the visible context gives no support or makes it impossible.
- A task-completion claim is bad if visible task status says otherwise.
- A role certainty claim is bad unless visible hard evidence supports it.
- Use label false only when there is a clear issue in the three dimensions.
- Use label true when no clear issue is found from the visible context.
- If the input is ambiguous, prefer true. Do not invent hidden evidence.
- For label true, reason_code must be "none".
- For label false, choose exactly one non-none reason_code.

Return exactly one compact JSON object on one line. No markdown. No extra keys.
Schema:
{"label":true|false,"reason_code":"none|objective_contradiction|dogmatic_judgment|inappropriate_reasoning|game_rule_error|self_contradiction"}
"""


@dataclass
class LLMFactVerifierConfig:
    model: str
    api_url: str
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 48
    timeout: float = 60.0
    cache_path: Optional[str] = None


def default_api_url() -> str:
    return (
        os.getenv("LLM_VERIFIER_API_URL")
        or os.getenv("OPENROUTER_API_URL")
        or "https://openrouter.ai/api/v1/chat/completions"
    )


def default_model() -> str:
    return os.getenv("LLM_VERIFIER_MODEL") or "openai/gpt-4o-mini"


def resolve_api_key(api_url: str, api_key_env: Optional[str] = None) -> str:
    if api_key_env:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set for --llm_verifier.")
        return api_key

    lowered = api_url.lower()
    if "localhost" in lowered or "127.0.0.1" in lowered:
        return "dummy"

    if "openrouter.ai" in lowered:
        env_names = ("LLM_VERIFIER_API_KEY", "OPENROUTER_API_KEY", "LLM_API_KEY")
    else:
        env_names = ("LLM_VERIFIER_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")

    for env_name in env_names:
        api_key = os.getenv(env_name)
        if api_key:
            return api_key

    raise RuntimeError(
        "No API key found for --llm_verifier. Set LLM_VERIFIER_API_KEY, "
        "OPENROUTER_API_KEY, LLM_API_KEY, or pass --llm_verifier_api_key_env."
    )


class LLMFactVerifier:
    """
    API-backed verifier for meeting speech fact contradictions.

    The interface matches the rule-based FactVerifier: label() returns False
    only when a contradiction is detected, True when no contradiction is found,
    and None when the verifier is uncertain or cannot parse a verdict.
    """

    def __init__(self, config: LLMFactVerifierConfig) -> None:
        self.config = config
        self.last_result: Optional[Dict[str, Any]] = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_path = Path(config.cache_path).expanduser().resolve() if config.cache_path else None
        if self._cache_path and self._cache_path.exists():
            self._load_cache()

    def label(self, entry: Dict[str, Any], speech: str) -> Optional[bool]:
        if not speech:
            self.last_result = None
            return None

        messages = self._messages(entry, speech)
        cache_key = self._cache_key(messages)
        if cache_key in self._cache:
            result = dict(self._cache[cache_key])
            result["cache_hit"] = True
            self.last_result = result
            return self._label_from_result(result)

        raw_text = self._call_api(messages)
        result = self._parse_result(raw_text)
        result.update(
            {
                "cache_hit": False,
                "model": self.config.model,
                "prompt_version": PROMPT_VERSION,
                "raw_response": raw_text,
            }
        )
        self.last_result = result
        self._cache[cache_key] = result
        self._append_cache(cache_key, result)
        return self._label_from_result(result)

    def _messages(self, entry: Dict[str, Any], speech: str) -> List[Dict[str, str]]:
        prompt = entry.get("interaction", {}).get("prompt", {})
        phase = utils.get_phase(entry)
        all_info = utils.get_all_info(entry)
        memory = utils.memory_text(entry)
        player = utils.get_player_name(entry)
        current_location = utils.current_location(all_info)
        observation_history = utils.observation_block(all_info)
        action_history = utils.action_history_block(all_info)

        user_prompt = f"""Input context for the judge:

[Speaker]
{player}

[Current Phase]
{phase}

[Current Location]
{current_location or "Unknown"}

[Player Roster]
{json.dumps(prompt.get("Player Roster", []), ensure_ascii=False)}

[Speaker-Visible All Info]
{all_info}

[Speaker Memory]
{memory}

[Observation History Extract]
{observation_history}

[Action History Extract]
{action_history}

[Output speech to judge]
{speech}

Return one compact JSON object following the system schema."""
        return [
            {"role": "system", "content": LLM_FACT_VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _cache_key(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "prompt_version": PROMPT_VERSION,
            "model": self.config.model,
            "messages": messages,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        request = urllib.request.Request(self.config.api_url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM verifier API failed with HTTP {exc.code}: {body[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM verifier API request failed: {exc}") from exc

        parsed = json.loads(body)
        try:
            return parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM verifier API response did not contain choices[0].message.content: {body[:1000]}") from exc

    def _parse_result(self, raw_text: str) -> Dict[str, Any]:
        text = str(raw_text or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}

        raw_label = parsed.get("label")
        if isinstance(raw_label, bool):
            label: Optional[bool] = raw_label
        else:
            label_text = str(raw_label or "").strip().lower()
            if label_text in {"true", "consistent", "pass", "valid", "yes"}:
                label = True
            elif label_text in {"false", "contradiction", "fail", "invalid", "no"}:
                label = False
            else:
                label = None

        reason_code = str(parsed.get("reason_code", "") or "").strip().lower()
        if not reason_code:
            reason_code = "none" if label is True else "parse_error"

        if label is False and reason_code != "none":
            contradiction_types = [reason_code]
        else:
            contradiction_types = []

        return {
            "label": label,
            "reason_code": reason_code,
            "contradiction_types": contradiction_types,
        }

    def _label_from_result(self, result: Dict[str, Any]) -> Optional[bool]:
        label = result.get("label")
        if isinstance(label, bool):
            return label
        if label == "contradiction":
            return False
        if label == "consistent":
            return True
        return None

    def _load_cache(self) -> None:
        assert self._cache_path is not None
        with self._cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = item.get("key")
                result = item.get("result")
                if isinstance(key, str) and isinstance(result, dict):
                    self._cache[key] = result

    def _append_cache(self, key: str, result: Dict[str, Any]) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")
