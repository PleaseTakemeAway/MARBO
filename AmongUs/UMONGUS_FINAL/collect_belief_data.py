#!/usr/bin/env python3
"""Run games for belief-shift data collection.

Belief-action models, i.e. the models we train/evaluate with belief input, use
BeliefStateLLMAgent:

    b_hat_t^i ~ pi_i(. | tau_t^i)
    a_t^i     ~ pi_i(. | tau_t^i, b_hat_t^i)

Opponent/measurement LLM agents use BeliefPredictOnlyLLMAgent:

    b_hat_t^i ~ pi_i(. | tau_t^i)
    a_t^i     ~ pi_i(. | tau_t^i)

During meetings, the environment logs an all-listener belief sweep at meeting
start and immediately after every SPEAK action. These logs support
speech-induced belief-shift rewards.
"""

import argparse
import asyncio
import datetime
import os
import subprocess
import sys
import time
from typing import List

from tqdm.asyncio import tqdm as tqdm_asyncio

sys.path.append(os.path.join(os.path.abspath("."), "among-agents"))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

from amongagents.envs.configs.game_config import SEVEN_MEMBER_GAME
from amongagents.envs.configs.map_config import map_coords
from amongagents.envs.game import AmongUs
from utils import setup_experiment

try:
    from amongagents.UI.MapUI import MapUI
except ImportError:
    MapUI = None


BELIEF_ACTION_GEMMA_MODELS: List[str] = [
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-31B-it",
]

DEFAULT_API_OPPONENT_MODELS: List[str] = [
    "openai/gpt-4o-mini",
]

DEFAULT_MODEL_POOL: List[str] = [
    *BELIEF_ACTION_GEMMA_MODELS,
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen3.6-27B",
    "microsoft/phi-4",
    *DEFAULT_API_OPPONENT_MODELS,
]


ROOT_PATH = os.path.abspath(".")
LOGS_PATH = os.environ.get(
    "EXPERIMENT_LOGS_PATH",
    os.path.join(ROOT_PATH, "expt-logs", "belief_shift_collection"),
)
ASSETS_PATH = os.path.join(ROOT_PATH, "among-agents", "amongagents", "assets")
BLANK_MAP_IMAGE = os.path.join(ASSETS_PATH, "blankmap.png")


def current_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .strip()
            .decode("utf-8")
        )
    except Exception:
        return os.environ.get("COMMIT_HASH", "unknown")


async def multiple_games(args: argparse.Namespace) -> None:
    model_pool = list(dict.fromkeys(args.model_pool))
    belief_action_models = list(dict.fromkeys(args.belief_action_models))
    required_models = list(dict.fromkeys(args.required_models or belief_action_models))

    missing = [model for model in required_models if model not in model_pool]
    if missing:
        raise ValueError(f"required models must be included in --model_pool: {missing}")

    agent_config = {
        "Impostor": "LLM",
        "Crewmate": "LLM",
        "IMPOSTOR_LLM_CHOICES": model_pool,
        "CREWMATE_LLM_CHOICES": model_pool,
    }
    if args.model_weights:
        if len(args.model_weights) != len(model_pool):
            raise ValueError("--model_weights must have the same length as --model_pool")
        agent_config["IMPOSTOR_LLM_WEIGHTS"] = list(args.model_weights)
        agent_config["CREWMATE_LLM_WEIGHTS"] = list(args.model_weights)

    run_args = {
        "game_config": SEVEN_MEMBER_GAME,
        "include_human": False,
        "test": False,
        "personality": args.personality,
        "agent_config": agent_config,
        "belief_action_models": belief_action_models,
        "required_models": required_models,
        "predict_only_opponents": True,
        "post_speech_belief_sweep": True,
        "UI": args.display_ui,
    }

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    experiment_name = setup_experiment(
        args.name,
        LOGS_PATH,
        date,
        current_commit(),
        run_args,
    )
    if args.display_ui and MapUI is None:
        raise RuntimeError("UI requested, but tkinter/MapUI dependencies are unavailable.")
    ui = MapUI(BLANK_MAP_IMAGE, map_coords, debug=False) if args.display_ui else None

    semaphore = asyncio.Semaphore(args.rate_limit)
    start_time = time.time()
    completed = 0
    pbar = tqdm_asyncio(
        total=args.num_games,
        desc=f"Games [{experiment_name}]",
        unit="game",
        dynamic_ncols=True,
        smoothing=0.3,
    )

    async def run_limited_game(game_index: int) -> None:
        nonlocal completed
        async with semaphore:
            game = AmongUs(
                game_config=SEVEN_MEMBER_GAME,
                include_human=False,
                test=False,
                personality=args.personality,
                agent_config=agent_config,
                belief_state=False,
                belief_state_models=belief_action_models,
                belief_predict_only_models=[],
                required_models=required_models,
                predict_only_opponents=True,
                post_speech_belief_sweep=True,
                UI=ui,
                game_index=game_index,
            )
            try:
                await game.run_game()
            finally:
                completed += 1
                elapsed = time.time() - start_time
                avg = elapsed / completed if completed else 0.0
                remaining = avg * (args.num_games - completed)
                pbar.set_postfix(
                    {
                        "avg": f"{avg:.1f}s",
                        "elapsed": f"{elapsed / 60:.1f}m",
                        "ETA": f"{remaining / 60:.1f}m",
                    },
                    refresh=False,
                )
                pbar.update(1)

    try:
        await asyncio.gather(*(run_limited_game(i) for i in range(1, args.num_games + 1)))
    finally:
        pbar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Among Us belief-shift game logs.")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--num_games", type=int, default=20)
    parser.add_argument("--rate_limit", type=int, default=4)
    parser.add_argument("--display_ui", action="store_true")
    parser.add_argument("--personality", action="store_true")
    parser.add_argument("--model_pool", nargs="+", default=DEFAULT_MODEL_POOL)
    parser.add_argument("--model_weights", type=float, nargs="+", default=None)
    parser.add_argument("--belief_action_models", nargs="+", default=BELIEF_ACTION_GEMMA_MODELS)
    parser.add_argument(
        "--required_models",
        nargs="+",
        default=None,
        help="Models that must appear at least once in every game. Defaults to belief_action_models.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(multiple_games(parse_args()))
