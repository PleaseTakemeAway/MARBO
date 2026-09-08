import argparse
import json
import ast
try:
    import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x
    tqdm.write = print
import os
from MARBO.wolf.preference_extraction.utils_english import get_system_prompt, create_message, get_game_description
try:
    import openai
except ImportError:
    openai = None
import re
try:
    import anthropic
except ImportError:
    anthropic = None

OPENAI_MODEL_ALIASES = {
    "openai_gpt4": "gpt-4o",
    "openai_gpt4o": "gpt-4o",
    "openai_gpt4o_mini": "gpt-4o-mini",
    "openai_gpt54": "gpt-5.4",
    "openai_gpt54_mini": "gpt-5.4-mini",
    "gpt5-4": "gpt-5.4",
    "gpt5.4": "gpt-5.4",
    "gpt54": "gpt-5.4",
    "gpt5-4-mini": "gpt-5.4-mini",
    "gpt5.4-mini": "gpt-5.4-mini",
    "gpt54-mini": "gpt-5.4-mini",
}

CLAUDE_MODEL_ALIASES = {
    "aws_claude35_sdk_sonnet": "claude-3-5-sonnet-20241022",
    "claude35_sonnet": "claude-3-5-sonnet-20241022",
    "claude_sonnet_4_6": "claude-sonnet-4-5-20250930",
}

_CLIENT_CACHE = {}


def _infer_provider(llm):
    llm_lower = llm.lower()
    if llm_lower.startswith("openai_"):
        return "openai"
    if "claude" in llm_lower or llm_lower.startswith("aws_"):
        return "claude"
    return "openai"


def _normalize_model_name(llm, provider):
    if provider == "claude":
        return CLAUDE_MODEL_ALIASES.get(llm, llm)
    return OPENAI_MODEL_ALIASES.get(llm, llm)


def _build_openai_client():
    if openai is None:
        raise ImportError("openai package is required to use an OpenAI-compatible verifier model.")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY or OPENROUTER_API_KEY is required to use an OpenAI-compatible verifier model."
        )
    kwargs = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENROUTER_API_BASE")
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def _build_claude_client():
    if anthropic is None:
        raise ImportError("anthropic package is required to use a Claude verifier model.")

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("AWS_CLAUDE_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("AWS_CLAUDE_API_BASE")

    if not api_key:
        raise ValueError(
            "Claude verifier requires ANTHROPIC_API_KEY or AWS_CLAUDE_API_KEY. "
            "Do not reuse OPENAI_API_KEY for Claude."
        )

    if base_url and "api.openai.com" in base_url:
        base_url = None

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


def _get_client(provider):
    if provider not in _CLIENT_CACHE:
        if provider == "claude":
            _CLIENT_CACHE[provider] = _build_claude_client()
        else:
            _CLIENT_CACHE[provider] = _build_openai_client()
    return _CLIENT_CACHE[provider]


def _response_to_text(response):
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _anthropic_response_to_text(response):
    parts = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _prepare_anthropic_messages(messages):
    system_parts = []
    anthropic_messages = []

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        if role == "system":
            system_parts.append(content)
        elif role in {"user", "assistant"}:
            anthropic_messages.append({"role": role, "content": content})

    if not anthropic_messages:
        anthropic_messages.append({"role": "user", "content": ""})

    return "\n\n".join(system_parts), anthropic_messages


def get_answer(messages, llm="aws_claude35_sdk_sonnet"):
    provider = _infer_provider(llm)
    model_name = _normalize_model_name(llm, provider)
    client = _get_client(provider)

    if provider == "claude":
        system_prompt, anthropic_messages = _prepare_anthropic_messages(messages)
        kwargs = {
            "model": model_name,
            # Verifier output can be long JSON; 1024 often truncates and breaks parsing.
            "max_tokens": 4096,
            "messages": anthropic_messages,
            "temperature": 0,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        response = client.messages.create(**kwargs)
        return _anthropic_response_to_text(response)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return _response_to_text(response)

def _extract_first_json_object(text):
    if not isinstance(text, str):
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _strip_code_fence(text):
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else stripped


def _parse_json_dict_maybe(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = _strip_code_fence(raw)
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    candidate = _extract_first_json_object(text)
    if candidate:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _normalize_bool_token(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"found", "true", "yes", "1", "contains"}:
        return True
    if token in {"not_found", "not found", "false", "no", "0", "none", "absent"}:
        return False
    return None


def _update_result(result, key, value):
    if value is None:
        return
    result[key] = bool(value)


def _parse_tag_style_judge(text, result):
    pattern = r"<final_judge>(.*?)</final_judge>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return
    judge_text = match.group(1).strip().lower()

    contradiction = None
    arbitrary = None
    improper = None

    if re.search(r"no contradiction with objective (facts|information)", judge_text):
        contradiction = False
    elif re.search(r"(contradicts|contains contradiction with) objective (facts|information)", judge_text):
        contradiction = True

    if re.search(r"no (arbitrary|dogmatic) judgment", judge_text):
        arbitrary = False
    elif re.search(r"contains (arbitrary|dogmatic) judgment", judge_text):
        arbitrary = True

    if re.search(r"no (improper|inappropriate) reasoning", judge_text):
        improper = False
    elif re.search(r"contains (improper|inappropriate) reasoning", judge_text):
        improper = True

    _update_result(result, "Contains contradiction with objective facts", contradiction)
    _update_result(result, "Contains arbitrary judgment", arbitrary)
    _update_result(result, "Contains improper reasoning", improper)


def _parse_json_style_judge(text, result):
    def _find_nested_dict_with_key(obj, target_key):
        if isinstance(obj, dict):
            if target_key in obj and isinstance(obj[target_key], dict):
                return obj[target_key]
            for value in obj.values():
                found = _find_nested_dict_with_key(value, target_key)
                if isinstance(found, dict):
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _find_nested_dict_with_key(item, target_key)
                if isinstance(found, dict):
                    return found
        return None

    parsed = _parse_json_dict_maybe(text)
    if not isinstance(parsed, dict):
        return

    final_verdict = None
    verification = parsed.get("verification")
    if isinstance(verification, dict):
        final_verdict = verification.get("final_verdict")
    if not isinstance(final_verdict, dict):
        maybe_final = parsed.get("final_verdict")
        if isinstance(maybe_final, dict):
            final_verdict = maybe_final
    if not isinstance(final_verdict, dict):
        final_verdict = _find_nested_dict_with_key(parsed, "final_verdict")

    if isinstance(final_verdict, dict):
        _update_result(
            result,
            "Contains contradiction with objective facts",
            _normalize_bool_token(final_verdict.get("objective_contradiction")),
        )
        _update_result(
            result,
            "Contains arbitrary judgment",
            _normalize_bool_token(final_verdict.get("dogmatic_judgment")),
        )
        _update_result(
            result,
            "Contains improper reasoning",
            _normalize_bool_token(final_verdict.get("inappropriate_reasoning")),
        )

    # Fallback: infer from sentence-level failed analyses if final_verdict is missing
    verification = parsed.get("verification")
    step3 = None
    if isinstance(verification, dict):
        step3 = verification.get("step3_speech_analysis")
    if isinstance(step3, list):
        contradiction = None
        arbitrary = None
        improper = None
        for item in step3:
            if not isinstance(item, dict):
                continue
            verdict = str(item.get("verdict", "")).strip().lower()
            if verdict != "fail":
                continue
            error_type = str(item.get("error_type", "")).strip().lower()
            if "contradiction" in error_type:
                contradiction = True
            elif "dogmatic" in error_type or "arbitrary" in error_type:
                arbitrary = True
            elif "inappropriate" in error_type or "improper" in error_type or "reasoning" in error_type:
                improper = True
            else:
                # Unknown fail reason -> conservative fallback to reasoning error
                improper = True
        _update_result(result, "Contains contradiction with objective facts", contradiction)
        _update_result(result, "Contains arbitrary judgment", arbitrary)
        _update_result(result, "Contains improper reasoning", improper)


def _parse_text_fallback_judge(text, result):
    lower = str(text).lower()

    contradiction = None
    arbitrary = None
    improper = None

    if (
        "no contradiction with objective facts" in lower
        or "no contradiction with objective information" in lower
    ):
        contradiction = False
    elif (
        "contains contradiction with objective facts" in lower
        or "contradicts objective information" in lower
        or "objective contradiction: found" in lower
    ):
        contradiction = True

    if "no arbitrary judgment" in lower or "no dogmatic judgment" in lower:
        arbitrary = False
    elif "contains arbitrary judgment" in lower or "contains dogmatic judgment" in lower:
        arbitrary = True

    if "no improper reasoning" in lower or "no inappropriate reasoning" in lower:
        improper = False
    elif "contains improper reasoning" in lower or "contains inappropriate reasoning" in lower:
        improper = True

    # Extract sentence-level error_type markers from partially truncated JSON outputs.
    error_types = re.findall(r'error_type"\s*:\s*"([^"]+)"', lower)
    for et in error_types:
        et = et.strip().lower()
        if "contradiction" in et:
            contradiction = True
        elif "dogmatic" in et or "arbitrary" in et:
            arbitrary = True
        elif "inappropriate" in et or "improper" in et or "reasoning" in et:
            improper = True

    # If there are failed sentence verdicts but no explicit type, treat as reasoning issue.
    has_fail_verdict = bool(re.search(r'verdict"\s*:\s*"fail"', lower))
    if has_fail_verdict and contradiction is None and arbitrary is None and improper is None:
        improper = True

    _update_result(result, "Contains contradiction with objective facts", contradiction)
    _update_result(result, "Contains arbitrary judgment", arbitrary)
    _update_result(result, "Contains improper reasoning", improper)


def _extract_sample_quality_token(text):
    if not isinstance(text, str):
        return None
    lower = text.lower()

    patterns = [
        r'"sample_quality"\s*:\s*"(accept|revise|reject)"',
        r'"final_judge_strict"\s*:\s*"(accept|reject)"',
        r'"final_judge"\s*:\s*"(accept|reject)"',
        r"\bsample[_\s-]?quality\s*[:=]\s*(accept|revise|reject)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            return match.group(1).strip().lower()
    return None


def parse_final_judge(text):
    """
    Parse verifier outputs from either:
      1) legacy <final_judge>...</final_judge> format, or
      2) current JSON final_verdict format.
    """
    result = {}
    _parse_tag_style_judge(text, result)
    _parse_json_style_judge(text, result)
    _parse_text_fallback_judge(text, result)

    # Last-resort fallback using sample_quality/final_judge tokens.
    if not result:
        quality = _extract_sample_quality_token(text)
        if quality == "accept":
            result = {
                "Contains contradiction with objective facts": False,
                "Contains arbitrary judgment": False,
                "Contains improper reasoning": False,
            }
        elif quality == "revise":
            # Usually means no hard objective contradiction, but has fixable issues.
            result = {
                "Contains contradiction with objective facts": False,
                "Contains arbitrary judgment": True,
                "Contains improper reasoning": True,
            }
        elif quality == "reject":
            result = {
                "Contains contradiction with objective facts": True,
                "Contains arbitrary judgment": True,
                "Contains improper reasoning": True,
            }
    return result

def check_conflict(llm_model, sytem_prompt, input_prompt, output_response, game_type):
    game_rule = get_game_description(game_type)
    system_prompt = sytem_prompt.replace("{{GAME_RULE}}", game_rule)
    input_query = f"**Input**:\n{input_prompt}\n\n**Output**:\n{output_response}"
    messages = [
        create_message("system", system_prompt),
        create_message("user", input_query),
    ]
    raw_output = get_answer(messages, llm=llm_model).replace("\n\n", "\n")
    final_judge = parse_final_judge(raw_output)
    return final_judge, raw_output

def give_final_judgement(judge_dict, reason):
    ret = {
        "judge_json": judge_dict,
        "reason": reason,
        "final_judge": None,
        "final_judge_strict": None
    }
    if len(judge_dict) == 0:
        ret["final_judge"] = "reject"
        ret["final_judge_strict"] = "reject"
    else:
        contradiction = bool(judge_dict.get("Contains contradiction with objective facts", True))
        arbitrary = bool(judge_dict.get("Contains arbitrary judgment", True))
        improper = bool(judge_dict.get("Contains improper reasoning", True))

        if contradiction:
            ret["final_judge"] = "reject"
        else:
            ret["final_judge"] = "accept"

        if (
            contradiction
            or arbitrary
            or improper
        ):
            ret["final_judge_strict"] = "reject"
        else:
            ret["final_judge_strict"] = "accept"
    return ret
    

KNOWN_GAME_TYPES = {
    "9p_seer_witch_guard",
    "9p_seer_witch_hunter",
    "7p_seer_guard",
}


def infer_game_type_from_path(input_path):
    parts = os.path.abspath(input_path).split(os.sep)
    for part in reversed(parts):
        if part in KNOWN_GAME_TYPES:
            return part
    basename = os.path.basename(os.path.normpath(input_path))
    inferred = basename.split("-")[0]
    return inferred or None


def iter_speech_sample_files(input_path):
    if os.path.isfile(input_path):
        yield input_path
        return

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Speech sample path does not exist: {input_path}")

    for root, _, filenames in os.walk(input_path):
        for filename in sorted(filenames):
            lower = filename.lower()
            if not lower.endswith((".json", ".jsonl")):
                continue
            if "speech" not in lower:
                continue
            if lower.startswith("reverify_") or lower.startswith("verified_"):
                continue
            yield os.path.join(root, filename)


def write_progress(message):
    writer = getattr(tqdm, "write", None)
    if writer is None and hasattr(tqdm, "tqdm"):
        writer = getattr(tqdm.tqdm, "write", None)
    if writer is None:
        print(message)
    else:
        writer(message)


def sample_key_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def load_speech_sample_file(sample_file):
    with open(sample_file, "r", encoding="utf-8") as f:
        if sample_file.lower().endswith(".jsonl"):
            samples = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    write_progress(f"skip invalid jsonl line in {sample_file}")
            return samples
        return json.load(f)


def iter_good_speech_samples(good_speech, source_file=None):
    if isinstance(good_speech, list):
        for item in good_speech:
            if isinstance(item, dict):
                yield item, source_file
        return

    if isinstance(good_speech, dict):
        for game_path, player_phase_map in good_speech.items():
            file_root_abs = os.path.abspath(game_path)
            if not isinstance(player_phase_map, dict):
                continue
            for player_id, phase_list in player_phase_map.items():
                if isinstance(phase_list, str):
                    phase_list = [phase_list]
                phase_set = set(phase_list or [])
                player_file = os.path.join(file_root_abs, f"Player_{player_id}.jsonl")
                if not os.path.exists(player_file):
                    write_progress(f"missing player file: {player_file}")
                    continue

                with open(player_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            player_game_play_info = json.loads(line.strip())
                        except json.JSONDecodeError:
                            continue

                        phase = player_game_play_info.get("phase")
                        if phase not in phase_set:
                            continue

                        prompt = player_game_play_info.get("prompt")
                        response = player_game_play_info.get("response")
                        if not prompt or not response:
                            continue

                        yield {
                            "game_path": game_path,
                            "player_id": player_id,
                            "phase": phase,
                            "prompt": prompt,
                            "completion": response,
                        }, source_file
        return

    raise TypeError(f"Unsupported good_speech type: {type(good_speech).__name__}")


def iter_speech_samples_from_path(input_path):
    seen = set()
    sample_files = list(iter_speech_sample_files(input_path))
    if not sample_files:
        raise FileNotFoundError(f"No speech sample json/jsonl files found under: {input_path}")

    for sample_file in sample_files:
        good_speech = load_speech_sample_file(sample_file)
        for item, source_file in iter_good_speech_samples(good_speech, sample_file):
            key = (
                item.get("game_path"),
                str(item.get("player_id")),
                item.get("phase"),
                sample_key_value(item.get("prompt")),
                sample_key_value(item.get("completion") or item.get("response")),
            )
            if key in seen:
                continue
            seen.add(key)
            yield item, source_file


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--good_speech_file', '--speech_samples',
                           dest='good_speech_file',
                           type=str, default="./KTO_selected_data_9p_seer_witch_guard/tmp_good-speech.json",
                           help="path to a speech sample json/jsonl file or a directory containing speech sample files")
    argparser.add_argument('--game_type',
                           type=str, default=None,
                           help="choose from: 9p_seer_witch_guard, 9p_seer_witch_hunter, 7p_seer_guard")
    argparser.add_argument('--out_to', type=str, default="reverify_speeches.jsonl")
    argparser.add_argument('--llm_verifier', type=str, default="aws_claude35_sdk_sonnet",
                           help="Examples: aws_claude35_sdk_sonnet, claude35_sonnet, claude_sonnet_4_5")
    args = argparser.parse_args()

    if args.game_type is None:
        args.game_type = infer_game_type_from_path(args.good_speech_file)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    verifier_prompt_path = os.path.join(script_dir, "llm_verifier_sys_prompt.txt")
    with open(verifier_prompt_path, "r", encoding="utf-8") as f:
        verifier_system_prompt = f.read().strip()

    fout = open(args.out_to, "a+", encoding="utf-8")
    cnt = 0
    parse_fail_cnt = 0
    game_system_prompt = get_system_prompt(args.game_type)
    sample_iter = iter_speech_samples_from_path(args.good_speech_file)
    for game_i, (item, source_file) in enumerate(tqdm.tqdm(sample_iter)):
        game_path = item["game_path"]
        player_id = item["player_id"]
        phase = item["phase"]
        input_prompt = item["prompt"]
        response = item.get("completion", item.get("response"))
        # check whether there is conflict
        judge_dict, reason = check_conflict(
            args.llm_verifier, verifier_system_prompt, input_prompt, response, args.game_type
        )
        if len(judge_dict) == 0:
            parse_fail_cnt += 1
        final_judge = give_final_judgement(judge_dict, reason)
        json_out = {
            "game_path": game_path,
            "player_id": player_id,
            "phase": phase,
            "system_prompt": game_system_prompt,
            "prompt": input_prompt,
            "response": response,
            "source_file": source_file,
            "judge_by_llm": final_judge,
        }
        fout.write(json.dumps(json_out, ensure_ascii=False) + "\n")
        cnt += 1

        if cnt % 100 == 0:
            print(f"processed {cnt} samples | parse_fail={parse_fail_cnt}")
    fout.close()
