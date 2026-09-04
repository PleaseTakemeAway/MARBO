import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Relabel undesirable samples that have good state reconstruction/internal reasoning "
            "and save pair-wise (chosen/rejected) outputs."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="input json file path or directory containing json files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=(
            "for single input file: output json file path or output directory; "
            "for directory input: output directory"
        ),
    )
    parser.add_argument(
        "--pair_output_path",
        type=str,
        default="",
        help="optional pair-wise output path (single-file mode only)",
    )
    parser.add_argument(
        "--include_all_phases",
        action="store_true",
        help="if set, do not restrict relabel candidates to night_skill/day_vote phases",
    )
    return parser.parse_args()


def _to_samples(raw: Any) -> List[Dict[str, Any]]:
    """
    Accept both:
      1) list-of-samples format (kto_dataset_train_samples.json style)
      2) columnar dict format (kto_dataset_train.json style)
    """
    if isinstance(raw, list):
        samples = [x for x in raw if isinstance(x, dict)]
        return samples

    if isinstance(raw, dict):
        needed = ["prompt", "completion", "label", "role_label", "phase", "game_path"]
        if not all(k in raw for k in needed):
            return []
        n = len(raw.get("prompt", []))
        samples: List[Dict[str, Any]] = []
        for i in range(n):
            sample = {
                "prompt": raw["prompt"][i],
                "completion": raw["completion"][i],
                "label": raw["label"][i],
                "role_label": raw["role_label"][i],
                "phase": raw["phase"][i],
                "game_path": raw["game_path"][i],
                "state_recon": raw.get("state_recon", [None] * n)[i],
            }
            samples.append(sample)
        return samples

    return []


def _load_samples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return _to_samples(raw)


def _iter_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted([p for p in input_path.glob("*.json") if p.is_file()])
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _phase_is_target(phase: str) -> bool:
    text = str(phase)
    return ("night_skill" in text) or ("_day_vote" in text)


def _internal_reasoning_ok(sample: Dict[str, Any]) -> bool:
    # Explicit fields (if present) take precedence.
    for key in ("internal_reasoning_ok", "internal_state_ok", "internal_label"):
        if key in sample:
            return sample.get(key) is True

    meta = sample.get("speech_asymmetric_meta")
    if isinstance(meta, dict):
        for key in ("internal_label", "internal_state_ok", "stage1_internal_label"):
            if key in meta:
                return meta.get(key) is True

    # Fallback for kto_dataset_train_samples style.
    return sample.get("state_recon") is True


def _is_relabel_candidate(sample: Dict[str, Any], include_all_phases: bool) -> bool:
    if sample.get("label") is not False:
        return False
    if sample.get("state_recon") is not True:
        return False
    if (not include_all_phases) and (not _phase_is_target(str(sample.get("phase", "")))):
        return False
    if not _internal_reasoning_ok(sample):
        return False
    return True


def _build_relabel_pair(sample: Dict[str, Any], source_index: int, source_file: str, pair_id: str) -> Dict[str, Any]:
    rejected = copy.deepcopy(sample)
    relabeled = copy.deepcopy(sample)
    relabeled["label"] = True
    relabeled["relabel_meta"] = {
        "pair_id": pair_id,
        "source_file": source_file,
        "source_index": source_index,
        "reason": "outcome_undesirable_but_state_recon_and_internal_reasoning_appropriate",
    }
    return {
        "pair_id": pair_id,
        "source_file": source_file,
        "source_index": source_index,
        "phase": sample.get("phase"),
        "game_path": sample.get("game_path"),
        "rejected": rejected,  # original undesirable
        "chosen": relabeled,   # relabeled desirable
    }


def relabel_samples(samples: List[Dict[str, Any]], source_file: str, include_all_phases: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    augmented = [copy.deepcopy(s) for s in samples]
    pairs: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        if not _is_relabel_candidate(sample, include_all_phases=include_all_phases):
            continue
        pair_id = f"{Path(source_file).stem}_pair_{idx}"
        pair = _build_relabel_pair(sample, source_index=idx, source_file=source_file, pair_id=pair_id)
        pairs.append(pair)
        augmented.append(copy.deepcopy(pair["chosen"]))

    return augmented, pairs, len(pairs)


def _resolve_output_paths(
    input_file: Path,
    output_path: Path,
    pair_output_path: Path | None,
    single_input: bool,
) -> Tuple[Path, Path]:
    if single_input and output_path.suffix == ".json":
        augmented_out = output_path
        if pair_output_path is not None:
            pair_out = pair_output_path
        else:
            pair_out = output_path.with_name(f"{output_path.stem}_pairs.json")
        return augmented_out, pair_out

    # directory mode
    output_path.mkdir(parents=True, exist_ok=True)
    augmented_out = output_path / f"{input_file.stem}_relabel_augmented.json"
    if pair_output_path is not None and single_input:
        pair_out = pair_output_path
    else:
        pair_out = output_path / f"{input_file.stem}_relabel_pairs.json"
    return augmented_out, pair_out


def _save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def main():
    args = parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    pair_output_path = Path(args.pair_output_path).expanduser().resolve() if args.pair_output_path else None

    input_files = _iter_input_files(input_path)
    if not input_files:
        raise FileNotFoundError(f"No JSON files found under: {input_path}")

    single_input = len(input_files) == 1 and input_files[0].is_file()
    total_pairs = 0

    for in_file in input_files:
        samples = _load_samples(in_file)
        if not samples:
            print(f"[SKIP] {in_file}: unsupported/empty format")
            continue

        augmented, pairs, pair_count = relabel_samples(
            samples=samples,
            source_file=str(in_file),
            include_all_phases=args.include_all_phases,
        )
        total_pairs += pair_count

        augmented_out, pair_out = _resolve_output_paths(
            input_file=in_file,
            output_path=output_path,
            pair_output_path=pair_output_path,
            single_input=single_input,
        )
        _save_json(augmented_out, augmented)
        _save_json(pair_out, pairs)

        print(
            f"[relabel] file={in_file.name} "
            f"original={len(samples)} relabeled_pairs={pair_count} augmented={len(augmented)}"
        )
        print(f"[saved] augmented -> {augmented_out}")
        print(f"[saved] pairs     -> {pair_out}")

    print(f"[done] total_relabel_pairs={total_pairs}")


if __name__ == "__main__":
    main()
