from pathlib import Path
from typing import Any, Optional

from .game_reward import BeliefAwareGameReward
from .llm_fact_verifier import (
    LLMFactVerifier,
    LLMFactVerifierConfig,
    default_api_url,
    default_model,
    resolve_api_key,
)
from .meeting_speech_reward import BeliefMeetingSpeechReward


def add_llm_verifier_args(parser: Any) -> None:
    parser.add_argument(
        "--llm_verifier",
        "--llm_verfier",
        dest="llm_verifier",
        action="store_true",
        help="Use an API-backed LLM fact verifier for Meeting phase SPEAK contradiction filtering.",
    )
    parser.add_argument(
        "--llm_verifier_model",
        default=None,
        help="Verifier model name. Defaults to LLM_VERIFIER_MODEL or openai/gpt-4o-mini for OpenRouter.",
    )
    parser.add_argument(
        "--llm_verifier_api_url",
        default=None,
        help="Chat-completions API URL. Defaults to LLM_VERIFIER_API_URL, OPENROUTER_API_URL, or OpenRouter chat completions.",
    )
    parser.add_argument(
        "--llm_verifier_api_key_env",
        default=None,
        help="Environment variable containing the verifier API key. Defaults to LLM_VERIFIER_API_KEY or OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--llm_verifier_cache_path",
        default=None,
        help="Optional JSONL cache path for verifier results. If omitted, builders choose a cache beside the KTO output.",
    )
    parser.add_argument("--llm_verifier_temperature", type=float, default=0.0)
    parser.add_argument("--llm_verifier_max_tokens", type=int, default=48)
    parser.add_argument("--llm_verifier_timeout", type=float, default=60.0)


def build_reward_from_args(args: Any, default_llm_cache_path: Optional[Path] = None) -> BeliefAwareGameReward:
    if not getattr(args, "llm_verifier", False):
        return BeliefAwareGameReward()

    api_url = args.llm_verifier_api_url or default_api_url()
    config = LLMFactVerifierConfig(
        model=args.llm_verifier_model or default_model(),
        api_url=api_url,
        api_key=resolve_api_key(api_url, args.llm_verifier_api_key_env),
        temperature=args.llm_verifier_temperature,
        max_tokens=args.llm_verifier_max_tokens,
        timeout=args.llm_verifier_timeout,
        cache_path=args.llm_verifier_cache_path or (str(default_llm_cache_path) if default_llm_cache_path else None),
    )
    speech_reward = BeliefMeetingSpeechReward(fact_verifier=LLMFactVerifier(config))
    return BeliefAwareGameReward(speech_reward=speech_reward)
