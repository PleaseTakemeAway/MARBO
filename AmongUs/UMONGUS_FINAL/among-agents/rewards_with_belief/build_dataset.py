import argparse
import json
from pathlib import Path

from .data_builder import build_kto_dataset, dataset_summary, resolve_log_paths as resolve_input_log_paths, save_dataset
from .game_reward import BeliefAwareGameReward
from .output_paths import add_include_model_scope_to_dataset_dir
from .reward_factory import add_llm_verifier_args, build_reward_from_args


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "kto_dataset"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a belief-aware Among Us KTO dataset from compact agent logs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log_paths", nargs="+", help="agent-logs-compact.json path(s).")
    source.add_argument(
        "--log_root",
        help="Directory containing run subdirectories with agent-logs-compact.json files.",
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument(
        "--include_models",
        nargs="+",
        default=None,
        help=(
            "Only include entries whose player.model contains one of these "
            "case-insensitive strings. Comma-separated values are also accepted."
        ),
    )
    add_llm_verifier_args(parser)
    parser.add_argument(
        "--llm_verifier_after_sampling",
        action="store_true",
        help=(
            "When --llm_verifier is set, sample with the cheap rule-based reward first, "
            "then run the LLM verifier only for sampled meeting-speech rows."
        ),
    )
    parser.add_argument(
        "--task_actions_only",
        action="store_true",
        help="Build only Task phase action rows and use task-action-type sampling.",
    )
    parser.add_argument(
        "--include_repair_negatives",
        action="store_true",
        help="Add invalid action-repair attempts as label=False task-action rows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_log_paths(args) -> list[str]:
    if args.log_paths:
        return resolve_input_log_paths(args.log_paths)

    root = Path(args.log_root).expanduser().resolve()
    paths = resolve_input_log_paths([str(root)])
    if args.max_files is not None:
        paths = paths[: args.max_files]
    return paths


def main() -> None:
    args = parse_args()
    output_dir = add_include_model_scope_to_dataset_dir(
        Path(args.output_dir).expanduser().resolve(),
        args.include_models,
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")

    log_paths = resolve_log_paths(args)
    if not log_paths:
        raise ValueError("No agent-logs-compact.json files found.")

    print(f"[data] building from {len(log_paths)} log file(s)")
    if args.include_models:
        print(f"[data] including models matching: {', '.join(args.include_models)}")
    if args.llm_verifier:
        print("[data] using API-backed LLM fact verifier for meeting speech")
    if args.task_actions_only:
        print("[data] building task-action-only dataset")
    if args.include_repair_negatives:
        print("[data] adding invalid repair attempts as negative task-action rows")
    llm_reward_fn = build_reward_from_args(args, default_llm_cache_path=output_dir / "llm_fact_verifier_cache.jsonl")
    if args.llm_verifier and args.llm_verifier_after_sampling:
        reward_fn = BeliefAwareGameReward()
        post_sampling_reward_fn = llm_reward_fn
        print("[data] LLM verifier will run after KTO ratio sampling")
    else:
        reward_fn = llm_reward_fn
        post_sampling_reward_fn = None

    dataset = build_kto_dataset(
        log_paths,
        reward_fn=reward_fn,
        post_sampling_reward_fn=post_sampling_reward_fn,
        include_models=args.include_models,
        task_actions_only=args.task_actions_only,
        include_repair_negatives=args.include_repair_negatives,
    )
    summary = dataset_summary(dataset)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if output_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(output_dir)
    save_dataset(dataset, str(output_dir))
    print(f"[data] saved to {output_dir}")


if __name__ == "__main__":
    main()
