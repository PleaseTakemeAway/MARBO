import argparse
import json
import shutil
from pathlib import Path

from .data_builder import build_kto_dataset, dataset_summary as kto_summary, save_dataset
from .game_reward import BeliefAwareGameReward
from .output_paths import (
    add_include_model_scope_to_dataset_dir,
    add_include_model_scope_to_output_root,
)
from .reward_factory import add_llm_verifier_args, build_reward_from_args
from .role_prediction_sft_data_builder import (
    build_role_prediction_sft_dataset,
    dataset_summary as sft_summary,
    resolve_log_paths as resolve_input_log_paths,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build all rewards_with_belief datasets: KTO action/vote/speech data "
            "and role-prediction SFT data."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log_paths", nargs="+", help="agent-logs-compact.json path(s).")
    source.add_argument("--log_root", help="Directory containing agent-logs-compact.json files.")
    parser.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where kto_dataset and role_prediction_sft_dataset are saved.",
    )
    parser.add_argument("--kto_output_dir", default=None)
    parser.add_argument("--sft_output_dir", default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument(
        "--include_sorted",
        action="store_true",
        help="Include duplicate sorted/ log directories. By default they are excluded.",
    )
    parser.add_argument(
        "--include_models",
        nargs="+",
        default=None,
        help=(
            "Only include samples whose player.model contains one of these "
            "case-insensitive substrings. Comma-separated values are also accepted. "
            "The filter is applied to both KTO action rows and SFT role_prediction rows."
        ),
    )
    parser.add_argument(
        "--task_label_mode",
        choices=("strict_observable", "action_space"),
        default="strict_observable",
        help="Task-phase Crewmate SFT label mode.",
    )
    parser.add_argument(
        "--include_previous_event_memory",
        action="store_true",
        help="Also parse Previous Event Memory for SFT hard evidence.",
    )
    parser.add_argument(
        "--num_impostors",
        type=int,
        default=2,
        help="Expected number of Impostors per game for completing oracle SFT labels.",
    )
    add_llm_verifier_args(parser)
    parser.add_argument(
        "--llm_verifier_after_sampling",
        action="store_true",
        help=(
            "When --llm_verifier is set, first build/sample KTO rows with the cheap "
            "rule-based reward, then run the LLM verifier only for sampled meeting-speech rows."
        ),
    )
    parser.add_argument("--skip_kto", action="store_true")
    parser.add_argument("--skip_sft", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_log_paths(args) -> list[str]:
    raw_paths = args.log_paths if args.log_paths else [str(Path(args.log_root).expanduser().resolve())]
    paths = resolve_input_log_paths(raw_paths, exclude_sorted=not args.include_sorted)
    if args.max_files is not None:
        paths = paths[: args.max_files]
    return paths


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    if args.skip_kto and args.skip_sft:
        raise ValueError("Both --skip_kto and --skip_sft were set; nothing to build.")

    log_paths = resolve_log_paths(args)
    if not log_paths:
        raise ValueError("No agent-logs-compact.json files found.")

    output_root = add_include_model_scope_to_output_root(
        Path(args.output_root).expanduser().resolve(),
        args.include_models,
    )
    kto_output_dir = (
        add_include_model_scope_to_dataset_dir(
            Path(args.kto_output_dir).expanduser().resolve(),
            args.include_models,
        )
        if args.kto_output_dir
        else output_root / "kto_dataset"
    )
    sft_output_dir = (
        add_include_model_scope_to_dataset_dir(
            Path(args.sft_output_dir).expanduser().resolve(),
            args.include_models,
        )
        if args.sft_output_dir
        else output_root / "role_prediction_sft_dataset"
    )

    print(f"[data] building from {len(log_paths)} log file(s)")
    if args.include_models:
        print(f"[data] including models matching: {', '.join(args.include_models)}")
    if args.llm_verifier and not args.skip_kto:
        print("[kto] using API-backed LLM fact verifier for meeting speech")

    summaries = {}

    if not args.skip_kto:
        prepare_output_dir(kto_output_dir, args.overwrite)
        print(f"[kto] building -> {kto_output_dir}")
        llm_reward_fn = build_reward_from_args(
            args,
            default_llm_cache_path=kto_output_dir / "llm_fact_verifier_cache.jsonl",
        )
        if args.llm_verifier and args.llm_verifier_after_sampling:
            reward_fn = BeliefAwareGameReward()
            post_sampling_reward_fn = llm_reward_fn
            print("[kto] LLM verifier will run after KTO ratio sampling")
        else:
            reward_fn = llm_reward_fn
            post_sampling_reward_fn = None
        kto_dataset = build_kto_dataset(
            log_paths,
            reward_fn=reward_fn,
            post_sampling_reward_fn=post_sampling_reward_fn,
            include_models=args.include_models,
        )
        summaries["kto"] = kto_summary(kto_dataset)
        save_dataset(kto_dataset, str(kto_output_dir))

    if not args.skip_sft:
        prepare_output_dir(sft_output_dir, args.overwrite)
        print(f"[sft] building -> {sft_output_dir}")
        print(f"[sft] task_label_mode: {args.task_label_mode}")
        sft_dataset = build_role_prediction_sft_dataset(
            log_paths,
            include_models=args.include_models,
            task_label_mode=args.task_label_mode,
            include_previous_event_memory=args.include_previous_event_memory,
            num_impostors=args.num_impostors if args.num_impostors >= 0 else None,
        )
        summaries["sft"] = sft_summary(sft_dataset)
        save_dataset(sft_dataset, str(sft_output_dir))

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
