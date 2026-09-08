import argparse
import json
from pathlib import Path

from .role_prediction_sft_data_builder import (
    build_role_prediction_sft_dataset,
    dataset_summary,
    resolve_log_paths as resolve_input_log_paths,
    save_dataset,
)
from .output_paths import add_include_model_scope_to_dataset_dir


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "role_prediction_sft_dataset"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a role-prediction SFT dataset with task-phase rule labels, "
            "meeting-phase oracle labels, and always-oracle Impostor labels."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log_paths", nargs="+", help="agent-logs-compact.json path(s).")
    source.add_argument(
        "--log_root",
        help="Directory containing agent-logs-compact.json files.",
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
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
            "Only include entries whose player.model contains one of these "
            "case-insensitive strings. Comma-separated values are also accepted."
        ),
    )
    parser.add_argument(
        "--task_label_mode",
        choices=("strict_observable", "action_space"),
        default="strict_observable",
        help=(
            "Task-phase Crewmate label mode. strict_observable uses self role "
            "and observed KILL/VENT action evidence only. action_space also "
            "uses available KILL targets as Crewmate labels."
        ),
    )
    parser.add_argument(
        "--include_previous_event_memory",
        action="store_true",
        help=(
            "Also parse Previous Event Memory for hard evidence. Off by default "
            "because that memory is model-generated and can contain hallucinations."
        ),
    )
    parser.add_argument(
        "--num_impostors",
        type=int,
        default=2,
        help=(
            "Expected number of Impostors per game. Used only to complete oracle "
            "labels when a roster player has no compact log row but the missing "
            "role is inferable from the other logged roles. Pass a negative value "
            "to disable this inference."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_log_paths(args) -> list[str]:
    raw_paths = args.log_paths if args.log_paths else [str(Path(args.log_root).expanduser().resolve())]
    paths = resolve_input_log_paths(raw_paths, exclude_sorted=not args.include_sorted)
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
    print(f"[data] task_label_mode: {args.task_label_mode}")
    print("[data] label policy: Impostor=oracle, Meeting=Crewmate oracle, Task=Crewmate rule")
    if args.include_models:
        print(f"[data] including models matching: {', '.join(args.include_models)}")

    dataset = build_role_prediction_sft_dataset(
        log_paths,
        include_models=args.include_models,
        task_label_mode=args.task_label_mode,
        include_previous_event_memory=args.include_previous_event_memory,
        num_impostors=args.num_impostors if args.num_impostors >= 0 else None,
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
